"""Single-arm GJB ACT protocol: original core, FP32, masked full-mean L1+10 KL.

Only the data/robot interface differs: S9/A7, overview and recorded-row targets.
"""
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .aligned_data import SingleArmData, SingleArmDataset, file_hash
from .aligned_model import ACTConfig, ACTPolicy, compute_loss
from .aligned_state import StatefulBatchSampler, seed_everything, rng_state, restore_rng

VERSION = 'single-arm-gjb-act-v1'
GJB_COMMIT = '782e270'


def atomic_save(path, payload):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_suffix('.tmp')
    torch.save(payload,pending)
    os.replace(pending,path)


def canonical(value):
    return json.dumps(value,sort_keys=True,allow_nan=False)


@torch.no_grad()
def validate(model,loader,normalizer,device):
    model.eval()
    error=0.; count=0; xyz=0.; grip=0.; real=0
    for batch in loader:
        images,state=batch['images'].to(device),batch['state'].to(device)
        # No targets are passed to the model: exactly GJB's z=0 validation.
        prediction=model(images,state)['action'].float().cpu()
        valid=(~batch['is_pad'])[...,None].expand_as(prediction)
        error+=float((prediction-batch['action']).abs()[valid].sum())
        count+=int(valid.sum())
        a=normalizer.decode_action(prediction.numpy())
        b=normalizer.decode_action(batch['action'].numpy())
        mask=~batch['is_pad'].numpy()
        xyz+=float(np.linalg.norm(a[...,:3]-b[...,:3],axis=-1)[mask].sum())
        grip+=float(np.abs(a[...,6]-b[...,6])[mask].sum())
        real+=int(mask.sum())
    model.train()
    if not count or not math.isfinite(error):
        raise ValueError('Validation has no finite valid action targets')
    return {'prior_action_l1':error/count,'valid_action_coordinates':count,
        'xyz_error_mm':xyz/max(1,real)*1000,
        'gripper_target_mae_command_units':grip/max(1,real), 'prior_only':True,
        'success_rate':None}


def config_for(args):
    return {'steps':args.steps,'batch_size':args.batch_size,'num_workers':args.num_workers,
        'seed':args.seed,'learning_rate':args.learning_rate,'backbone_learning_rate':args.learning_rate,
        'weight_decay':1e-4,'precision':args.precision,'gradient_clip_norm':None,
        'lr_scheduler':'constant','warmup_steps':0,'validation_every':args.eval_steps,
        'save_every':args.save_freq,'candidate_save_every':args.candidate_save_every,
        'save_best':args.save_best,'cpu_threads':args.cpu_threads,
        'model':{'state_dim':9,'action_dim':7,'num_cameras':1,'chunk_size':args.chunk_size,
            'hidden_dim':512,'heads':8,'feedforward_dim':3200,'encoder_layers':4,
            'decoder_layers':7,'posterior_layers':4,'latent_dim':32,'dropout':.1}}


def run_aligned(args):
    config=config_for(args)
    checkpoint=None
    resume=args.resume
    continuation=args.continue_from
    if resume and continuation: raise ValueError('Choose --resume or --continue-from')
    if resume or continuation:
        if Path(resume or continuation).name!='last.pt':
            raise ValueError('Use last.pt for resume/continuation; best.pt is an inference selection')
        checkpoint=torch.load(resume or continuation,map_location='cpu',weights_only=True)
        if checkpoint.get('version')!=VERSION:
            raise ValueError('Old LeRobot/GJB dual-arm weights cannot resume into this single-arm model')
        frozen=checkpoint['config']
        if checkpoint['step']>frozen['steps']:raise ValueError('Checkpoint step exceeds its budget')
        if resume:
            best_path=Path(resume).parent/'best.pt'
            if best_path.exists() and torch.load(best_path,map_location='cpu',weights_only=True)['step']>checkpoint['step']:
                raise ValueError('best.pt is newer than last.pt; recover the interrupted checkpoint pair explicitly')
        if resume:
            config=frozen
        else:
            if checkpoint['step']!=frozen['steps']:
                raise ValueError('Use --resume for an unfinished run; continuation requires a completed budget')
            if args.steps<=checkpoint['step']:raise ValueError('Continuation needs a larger cumulative --steps')
            config={**frozen,'steps':args.steps}
    if args.smoke_test:
        if checkpoint:raise ValueError('Smoke tests cannot resume formal runs')
        config.update(steps=1,batch_size=1,num_workers=0,validation_every=1,save_every=1,
                      candidate_save_every=0)
        config['model']={**config['model'],'hidden_dim':32,'heads':4,'feedforward_dim':64,
            'encoder_layers':1,'decoder_layers':1,'posterior_layers':1,'latent_dim':8,'chunk_size':2}
    for key in ('steps','batch_size','validation_every','save_every','cpu_threads'):
        if config[key]<1:raise ValueError(f'{key} must be positive')
    if config['num_workers']<0 or config['candidate_save_every']<0:raise ValueError('Invalid workers/candidate cadence')
    if config['save_best'] and config['validation_every']%config['save_every']:
        raise ValueError('Each validation must also save last.pt: validation_every must be a multiple of save_every')
    if config['precision'] not in ('fp32','bf16'):raise ValueError('Unsupported precision')
    if config['model']['chunk_size']!=30 and not args.smoke_test:
        raise ValueError('GJB-aligned formal baseline fixes chunk_size=30')
    if not math.isfinite(config['learning_rate']) or config['learning_rate']<=0:raise ValueError('Invalid learning rate')
    output=Path(resume).resolve().parent.parent if resume else (
        args.output_dir or Path(__file__).resolve().parents[2]/'outputs/act_variation_baseline')
    output=Path(output).expanduser().resolve()
    if args.print_command:
        print(json.dumps({'engine':VERSION,'config':config,'output':str(output),
            'resume':str(resume) if resume else None},ensure_ascii=False,indent=2));return
    if output.exists() and not resume:raise ValueError(f'Output already exists: {output}')
    if args.device=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA is unavailable; select --device cpu or mps')
    device=torch.device(args.device)
    if config['precision']=='bf16' and (device.type!='cuda' or not torch.cuda.is_bf16_supported()):
        raise ValueError('BF16 requires supported CUDA; choose fp32 explicitly')
    torch.set_num_threads(config['cpu_threads']);seed_everything(config['seed'])
    data=SingleArmData(args.dataset_root,args.eval_split,args.split_file)
    if checkpoint:
        if (checkpoint['dataset_identity']!=data.identity or checkpoint['split']!=data.split
                or checkpoint['normalization']!=data.stats):
            raise ValueError('Resume cannot change dataset content, split or normalization; pass the original split file')
    weights=None
    if not checkpoint and not args.smoke_test and not args.no_pretrained_backbone:
        if args.backbone_weights:
            weights=Path(args.backbone_weights).expanduser().resolve()
        else:
            from torchvision.models import ResNet18_Weights
            ResNet18_Weights.IMAGENET1K_V1.get_state_dict(progress=True,check_hash=True)
            weights=Path(torch.hub.get_dir())/'checkpoints/resnet18-f37072fd.pth'
    model=ACTPolicy(ACTConfig(**config['model'],backbone_weights=str(weights) if weights else None)).to(device)
    backbone_ids={id(p) for p in model.backbone.parameters()}
    optimizer=torch.optim.AdamW([
        {'params':[p for p in model.parameters() if p.requires_grad and id(p) not in backbone_ids], 'lr':config['learning_rate']},
        {'params':[p for p in model.backbone.parameters() if p.requires_grad], 'lr':config['backbone_learning_rate']}],weight_decay=config['weight_decay'])
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
    train=SingleArmDataset(data,'train',config['model']['chunk_size'],args.repo_id)
    heldout=SingleArmDataset(data,'val',config['model']['chunk_size'],args.repo_id)
    sampler=StatefulBatchSampler(len(train),config['batch_size'],config['seed'],prefetch_safe=True)
    generator=torch.Generator().manual_seed(config['seed']+1)
    loader=DataLoader(train,batch_sampler=sampler,num_workers=config['num_workers'],
        pin_memory=device.type=='cuda',generator=generator,
        **({'prefetch_factor':2,'persistent_workers':True} if config['num_workers'] else {}))
    val_loader=DataLoader(heldout,batch_size=config['batch_size'],shuffle=False,num_workers=0,generator=generator)
    step=0;best=math.inf
    if checkpoint:
        model.load_state_dict(checkpoint['model'],strict=True)
        optimizer.load_state_dict(checkpoint['optimizer']);scheduler.load_state_dict(checkpoint['scheduler'])
        sampler.load_state_dict(checkpoint['sampler']);restore_rng(checkpoint['rng'])
        generator.set_state(checkpoint['loader_generator'].cpu())
        step=checkpoint['step'];best=checkpoint['best_prior_action_l1']
    output.mkdir(parents=True,exist_ok=bool(resume))
    (output/'checkpoints').mkdir(exist_ok=bool(resume))
    provenance={'version':VERSION,'gjb_reference_commit':GJB_COMMIT,
        'model_config':config['model'],'config':config,'dataset_root':str(data.root),
        'dataset_identity':data.identity,'split':data.split,'split_unit':data.split_unit,
        'normalization':data.stats,'image_key':'observation.images.overview',
        'runtime':{'device':str(device),'torch_version':str(torch.__version__),
                   'cuda_version':torch.version.cuda},
        'image_shape':data.info['features']['observation.images.overview']['shape'],
        'action_alignment':'existing_recorded_row_target_no_shift',
        'gripper_units':'existing_recorded_actuator_target; not dual-arm metres',
        'source_sha256':{str(p.relative_to(Path(__file__).parent)):file_hash(p)
            for p in sorted(Path(__file__).parent.rglob('*.py'))},
        'backbone_weights_sha256':file_hash(weights) if weights else checkpoint['provenance'].get('backbone_weights_sha256') if checkpoint else None,
        'engineering_smoke':args.smoke_test,
        'parent_checkpoint':str(continuation) if continuation else None}
    if resume:
        old=checkpoint['provenance']
        if old['source_sha256']!=provenance['source_sha256']:
            raise ValueError('Exact resume rejected: training/model source changed')
        # Ignore abandoned logs newer than the latest complete checkpoint.
        metrics_path=output/'metrics.jsonl'
        if metrics_path.exists():
            lines=metrics_path.read_text().splitlines()
            keep=[line for line in lines if json.loads(line)['step']<=step]
            if keep!=lines:
                (output/f'metrics_abandoned_after_{step}_{time.time_ns()}.jsonl').write_text('\n'.join(lines)+'\n')
                metrics_path.write_text('\n'.join(keep)+'\n')
    else:
        (output/'run.json').write_text(json.dumps(provenance,indent=2,ensure_ascii=False)+'\n')
        for relative in provenance['source_sha256']:
            destination=output/'source_snapshot'/relative
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(Path(__file__).parent/relative,destination)
        if continuation:
            source_best=Path(continuation).parent/'best.pt'
            if config['save_best'] and not source_best.is_file():
                raise ValueError('Continuation requires the parent best.pt together with last.pt')
            if source_best.is_file():
                best_payload=torch.load(source_best,map_location='cpu',weights_only=True)
                if (best_payload['version']!=VERSION or best_payload['dataset_identity']!=data.identity
                        or best_payload['config']!=checkpoint['config'] or best_payload['step']>checkpoint['step']):
                    raise ValueError('Parent best checkpoint does not match the continued run')
                atomic_save(output/'checkpoints/best.pt',best_payload)
    iterator=iter(loader)
    if checkpoint:generator.set_state(checkpoint['loader_generator'].cpu())

    def save(name):
        atomic_save(output/'checkpoints'/name,{'version':VERSION,'config':config,'provenance':provenance,
            'step':step,'best_prior_action_l1':best,'model':model.state_dict(),
            'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
            'sampler':sampler.state_dict(),'rng':rng_state(),'loader_generator':generator.get_state(),
            'dataset_identity':data.identity,'split':data.split,'normalization':data.stats})

    print(f'{VERSION}: S9/A7, one overview, {len(train)}/{len(heldout)} train/val frames, steps={config["steps"]}')
    print(f'Split: {data.split_unit}. Recorded-row action timing preserved; this is not a dual-arm Safe Stitch Pack.')
    model.train()
    try:
        while step<config['steps']:
            batch=next(iterator)
            batch={k:v.to(device,non_blocking=True) for k,v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=config['precision']=='bf16'):
                result=model(batch['images'],batch['state'],batch['action'],batch['is_pad'])
                losses=compute_loss(result,{'action':batch['action']},{'action':~batch['is_pad']},variant='baseline')
            if not torch.isfinite(losses['total']):raise FloatingPointError('Nonfinite training loss')
            losses['total'].backward()
            norm=torch.nn.utils.get_total_norm([p.grad for p in model.parameters() if p.grad is not None],error_if_nonfinite=True)
            optimizer.step();scheduler.step();sampler.mark_consumed();step+=1
            if step%25==0 or step==config['steps'] or args.smoke_test:
                with (output/'metrics.jsonl').open('a') as stream:
                    stream.write(json.dumps({'step':step,'event':'training',
                        'total':losses['total'].item(),'action':losses['action'].item(),
                        'kl':losses['kl'].item(),'gradient_norm':float(norm),
                        'lr':optimizer.param_groups[0]['lr']},allow_nan=False)+'\n')
                print(f'step {step}: loss={losses["total"].item():.6f} action={losses["action"].item():.6f} KL={losses["kl"].item():.6f}')
            if step%config['validation_every']==0 or step==config['steps']:
                metrics=validate(model,val_loader,data.normalizer,device)
                with (output/'metrics.jsonl').open('a') as stream:
                    stream.write(json.dumps({'step':step,'event':'validation',**metrics},allow_nan=False)+'\n')
                if config['save_best'] and metrics['prior_action_l1']<best:
                    best=metrics['prior_action_l1'];save('best.pt')
                print(f'validation step {step}: prior valid L1={metrics["prior_action_l1"]:.6f}')
            candidate=config['candidate_save_every'] and (step%config['candidate_save_every']==0 or step==config['steps'])
            if step%config['save_every']==0 or step==config['steps'] or candidate:
                save('last.pt')
            if candidate:save(f'step_{step:06d}.pt')
            if args.stop_after and step>=args.stop_after:
                save('last.pt');break
    except KeyboardInterrupt:
        print('Interrupted: resume last complete checkpoint; partial optimizer steps are not saved.')
        raise
    (output/'completion.json').write_text(json.dumps({'step':step,'target_steps':config['steps'],
        'completed':step==config['steps'],'engineering_smoke':args.smoke_test})+'\n')
    return output
