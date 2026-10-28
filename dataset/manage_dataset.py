"""唯一入口：单视角原件 → 事实裁剪 → auto/reviewed → Prepared → Training Pack。

Every run has its own output. Raw recordings and historical results are
immutable. The single-view pack has an independent reader; ACT adaptation is separate.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.task3 import prepare_paper_dataset as pipeline
from dataset.task3.curation_io import clear_input_caches, discover_episodes
from dataset.task3.review_server import serve_review
from dataset.task3.curation_rules import load_config, validate_config
from dataset.task3.rebuild_curation import build, sha, validate_source

POLICY = "papera-intent-aware"


def stamp():
    return datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")


def write_navigation(path, value):
    """Publish navigation atomically; UTF-8 JSON readable by legacy PowerShell.

    Unicode escapes keep these small pointers ASCII-compatible, avoiding ANSI
    decoding of UTF-8 Chinese text in Windows PowerShell 5.1. Artifact writers
    and their fingerprints are deliberately unchanged.
    """
    payload = json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False) + '\n'
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='\n',
                dir=path.parent, prefix=path.name+'.', suffix='.partial', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class DatasetWorkflow:
    def __init__(self, task_id, root=ROOT, *, engineering=False):
        from tasks.registry import get
        self.task = get(task_id)
        self.root = Path(root).resolve()
        self.engineering = engineering
        self.base = self.root / ('.local/checks/pipeline' if engineering else '.')
        self.source = self.base / 'datasets' / task_id
        self.output = self.base / 'outputs/dataset' / task_id

    def list_runs(self, kind="curation", *, history=False):
        if kind not in ("audits", "curation"):
            raise ValueError("kind must be audits or curation")
        files = set(self.output.glob("*/manifest.json"))
        files.update((self.output / kind).glob("*/manifest.json"))
        found = []
        for p in files:
            try:
                m = json.loads(p.read_text(encoding="utf-8"))
                if not m.get("episodes") or m.get("version") != pipeline.VERSION:
                    continue
                current = m.get("selection_policy") == POLICY
                evidence = m.get("pipeline_stage") == "evidence"
                if history:
                    if current or evidence:
                        continue
                elif kind == "curation" and not current:
                    continue
                elif kind == "audits" and not evidence:
                    continue
                found.append({"path": str(p.parent.resolve()), "manifest": m,
                              "read_only": not current})
            except (OSError, ValueError):
                continue
        return sorted(found, key=lambda x: (x["manifest"].get("created_at", ""), x["path"]))

    def catalog(self):
        runs = self.list_runs()
        self.output.mkdir(parents=True, exist_ok=True)
        columns = ["run", "window_frames", "minimum_static_frames", "intent_stable_frames", "before_frames",
                   "after_frames", "episodes", "raw_frames", "kept_frames",
                   "filtered_frames", "output_directory", "review_file", "training_ready"]
        temporary = self.output / "CURATION_CATALOG.csv.partial"
        with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for item in runs:
                p, m = Path(item["path"]), item["manifest"]
                cfg = m["config"]
                writer.writerow({
                    "run": m["run_id"], "window_frames": cfg["sa"]["window_frames"],
                    "minimum_static_frames": cfg["sa"].get("minimum_static_frames", "historical_no_confirmation"),
                    "intent_stable_frames": cfg["intent"]["stable_frames"],
                    "before_frames": cfg["context"]["before_frames"],
                    "after_frames": cfg["context"]["after_frames"],
                    **{k: m["summary"].get(k) for k in
                       ("episodes", "raw_frames", "kept_frames", "filtered_frames")},
                    "output_directory": str(p), "review_file": str(p / "reviews.json"),
                    "training_ready": False,
                })
        temporary.replace(self.output / "CURATION_CATALOG.csv")
        return runs

    def _remember(self, paths):
        self.output.mkdir(parents=True, exist_ok=True)
        path = self.output / "CURATION_CURRENT.json"
        write_navigation(path, {"selected_version": Path(paths[-1]).resolve().as_posix(),
            "recent_versions": [Path(p).resolve().as_posix() for p in paths],
            "selection_policy": POLICY, "training_ready": False})
        self.catalog()

    def reclip(self, audit_run, config=None, *, candidate=False):
        cfg = load_config(config) if not isinstance(config, dict) else validate_config(config)
        name = f"curation_{'candidate_' if candidate else ''}{stamp()}"
        output, _ = build(audit_run, self.output / "curation" / name, cfg)
        if not candidate:
            self._remember([output])
        return [output]

    def _matching_audit(self, source, refs):
        expected = {(r["key"], str(Path(r["dataset_root"]).resolve()), r["episode_index"]) for r in refs}
        for item in reversed(self.list_runs("audits")):
            m = item["manifest"]
            if Path(m["source_root"]).resolve() != source or m.get("partial_scope"):
                continue
            actual = {(e["key"], str(Path(e["dataset_root"]).resolve()), e["episode_index"]) for e in m["episodes"]}
            if actual != expected:
                continue
            try:
                validate_source(item["path"], check_raw=True)
            except (OSError, ValueError):
                continue
            return Path(item["path"])
        return None

    def process(self, source=None, config=None, fresh=False, workers=2, *, candidate=False):
        cfg = load_config(config) if not isinstance(config, dict) else validate_config(config)
        clear_input_caches()
        source = Path(source).resolve() if source is not None else self.source
        refs = discover_episodes(source, engineering=self.engineering)
        if not refs:
            raise ValueError(f"没有找到可处理的已保存记录：{source}")
        from tasks.artifacts import source_contract
        contract = source_contract(refs)
        if contract["task_id"] != self.task.id: raise ValueError("Cross-task source rejected")
        original = None if fresh else self._matching_audit(source, refs)
        if original:
            print(f"原件指纹未变，复用完整性与视频检查：{original}", flush=True)
        else:
            original = self.output / "audits" / f"evidence_{stamp()}"
            manifest = pipeline.audit(source, original, config=cfg, workers=workers, engineering=self.engineering)
            if not all(e["structural_pass"] and e["video_verified"] for e in manifest["episodes"]):
                raise ValueError(f"检查未全部通过，未生成裁剪结果。查看：{original / 'summary.csv'}")
        return self.reclip(original, cfg, candidate=candidate)

    def _version(self, run, *, allow_history=False):
        run = Path(run).resolve()
        from tasks.artifacts import check_artifact
        manifest = check_artifact(run, self.task.id, self.engineering)
        if manifest.get("version") != pipeline.VERSION:
            raise ValueError("请选择裁剪或历史审核结果；Prepared是导出物，不能重新作为审核来源。")
        current = manifest.get("selection_policy") == POLICY and manifest.get("pipeline_stage") == "curation"
        if manifest.get("selection_policy") == POLICY and not current:
            raise ValueError("当前策略缺少完整curation阶段标记，不能打开或导出。")
        if not current and not allow_history:
            raise ValueError("请选择当前意图与物理证据裁剪结果；历史结果仅支持只读查看。")
        if not manifest.get("episodes"):
            raise ValueError("处理清单没有 episode。")
        for e in manifest["episodes"]:
            if sha(pipeline.inside(run, run / e["index_path"])) != e["index_sha256"]:
                raise ValueError("索引已被修改；请重新处理，不要直接编辑结果文件。")
        return run, current

    def review(self, run, port=8025, review_source=None, *, candidate=False):
        run, current = self._version(run, allow_history=True)
        if current:
            if not candidate:
                self._remember([run])
            print(f"审核保存到：{run / 'reviews.json'}", flush=True)
        else:
            print("历史结果只读：不保存审核、不生成新的裁剪预览。", flush=True)
        options = {'review_source': review_source} if review_source is not None else {}
        serve_review(run, port=port, read_only=not current, **options)

    def preview(self, run=None, port=8026, *, open_browser=True):
        """Inspect frozen cuts, including older scenes; never approve or export."""
        if run is None:
            pointer = self.output / 'CURATION_CURRENT.json'
            if not pointer.is_file():
                raise ValueError('尚无 CURATION_CURRENT.json，请先 process，或用 --run 指定裁剪结果')
            try:
                selected = json.loads(pointer.read_text(encoding='utf-8-sig'))['selected_version']
            except (ValueError, KeyError) as exc:
                raise ValueError(f'导航文件无效：{pointer}；不能按时间猜测版本') from exc
            if not isinstance(selected, str) or not selected.strip() or not Path(selected).is_absolute():
                raise ValueError('selected_version 必须为非空绝对路径')
            run = selected
        run = Path(run).resolve()
        manifest = json.loads((run/'manifest.json').read_text(encoding='utf-8'))
        # Preview does not load a current scene or turn old provenance into new
        # training data. Formal prepare/pack retain their exact binding checks.
        if (manifest.get('task_id') != self.task.id or
                manifest.get('task_binding', {}).get('task_id') != self.task.id):
            raise ValueError('Cross-task preview rejected')
        if bool(manifest.get('engineering_only')) != bool(self.engineering):
            raise ValueError('Engineering and human preview scopes cannot be mixed')
        if (manifest.get('version') != pipeline.VERSION or
                manifest.get('pipeline_stage') != 'curation' or not manifest.get('episodes')):
            raise ValueError('请选择包含记录的 Curation 结果，不能预览 Audit/Prepared/Pack')
        for episode in manifest['episodes']:
            index = pipeline.inside(run, run/episode['index_path'])
            if sha(index) != episode.get('index_sha256'):
                raise ValueError('裁剪索引指纹不一致，拒绝生成预览')
        print(f'只读自动裁剪预览：{run}\n使用原版本保留帧；不改变训练资格或 CURRENT。', flush=True)
        serve_review(run, port=port, read_only=True, frozen_preview=True, open_browser=open_browser)

    def prepare(self, run, review_mode='reviewed', output=None, *, candidate=False):
        run, _ = self._version(run)
        target = Path(output).resolve() if output is not None else self.output / "prepared" / f"prepared_{'candidate_' if candidate else ''}{stamp()}"
        pipeline.prepare(run, target, review_mode=review_mode)
        if not candidate:
            write_navigation(self.output / 'PREPARED_CURRENT.json', {
                'path': target.as_posix(), 'curation': run.as_posix(), 'review_mode': review_mode,
                'training_started': False})
        print(f"已导出 {review_mode} Prepared：{target}\n下一步使用 pack；这一步不启动训练。", flush=True)
        return target

    def pack(self, prepared, output=None, *, split_file=None, width=640, height=480,
             resume=False, continue_on_error=False, candidate=False):
        from dataset.task3.training_pack import build as build_pack
        from tasks.artifacts import check_artifact
        check_artifact(prepared, self.task.id, self.engineering)
        if output is None:
            manifest = json.loads((Path(prepared) / 'manifest.json').read_text(encoding='utf-8'))
            review_mode = manifest.get('review_mode')
            if review_mode not in ('auto', 'reviewed'):
                raise ValueError('Prepared must declare its auto/reviewed selection mode')
            target = self.output / 'training_packs' / f'{"candidate_" if candidate else ""}{review_mode}_{stamp()}'
        else:
            target = Path(output).resolve()
        if resume and split_file is None and (target/'build.json').is_file():
            split_file = Path(json.loads((target/'build.json').read_text(encoding='utf-8'))['binding']['split_source'])
        report = build_pack(prepared, target, split_file=split_file or self.create_split(prepared),
            width=width, height=height, resume=resume, continue_on_error=continue_on_error)
        if report.get('complete') and not candidate:
            write_navigation(self.output / 'TRAINING_PACK_CURRENT.json', {
                'path': target.as_posix(), 'prepared': Path(prepared).resolve().as_posix(),
                'reader': 'dataset.task3.training_pack.read_episode', 'training_started': False})
        return report


    def create_split(self, prepared):
        import hashlib
        m=json.loads((Path(prepared)/'manifest.json').read_text(encoding='utf-8'))
        sessions=sorted({e['session'] for e in m['episodes']},key=lambda s:hashlib.sha256(('20261016:'+s).encode()).hexdigest())
        if len(sessions)<2:raise ValueError('Need at least two collection sessions for an independent train/val split')
        n=max(1,round(len(sessions)*.2))
        doc={'task_id':self.task.id,'seed':20261016,'unit':'whole session','val_sessions':sessions[:n],'train_sessions':sessions[n:]}
        folder=self.output/'splits';folder.mkdir(parents=True,exist_ok=True)
        path=folder/('split_'+stamp()+'.json');pipeline.write_json(path,doc)
        return path

def preview_menu(task_id=None, port=8026, *, open_browser=True):
    from tasks.registry import discover
    import socket
    tasks = [task for task in discover().values() if task.manifest['status'] == 'ready']
    if task_id is None:
        for number, task in enumerate(tasks, 1):
            print(f'{number}. {task.id} — {task.manifest["display_name"]}')
        choice = input('选择任务编号或 task_id（0 退出）：').strip()
        if choice == '0': return
        task_id = tasks[int(choice)-1].id if choice.isdigit() and 1 <= int(choice) <= len(tasks) else choice
    workflow = DatasetWorkflow(task_id)
    print(f'数据：{workflow.source}\n输出：{workflow.output}')
    print('1. 预览 CURRENT 指定的自动裁剪结果\n2. 选择已有裁剪版本预览\n3. 运行正式自动裁剪，再预览新结果\n0. 退出')
    choice = input('请选择：').strip()
    if choice == '0': return
    run = None
    if choice == '2':
        runs = workflow.list_runs()
        if not runs: raise ValueError('没有可选择的裁剪版本，请先运行自动裁剪')
        for number, item in enumerate(runs, 1): print(f'{number}. {item["path"]}')
        selected = input('选择版本编号：').strip()
        if not selected.isdigit() or not 1 <= int(selected) <= len(runs): raise ValueError('无效版本编号')
        run = runs[int(selected)-1]['path']
    elif choice == '3':
        run = workflow.process(workers=2)[-1]
    elif choice != '1': raise ValueError('无效菜单选项')
    # Never terminate/reuse another process or discard its unsaved review.
    for candidate in range(port, min(port+20, 65536)):
        try:
            with socket.socket() as sock:
                if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
                sock.bind(('127.0.0.1', candidate))
            break
        except OSError: continue
    else: raise ValueError('附近端口均不可用，请通过 -Port 指定空闲端口')
    if candidate != port: print(f'端口 {port} 已占用，使用 {candidate}；旧服务保持不变。')
    workflow.preview(run, candidate, open_browser=open_browser)


def main(argv=None):
    p=argparse.ArgumentParser(description='Task-scoped Audit -> Safe Stitch -> Prepared -> Training Pack')
    p.add_argument('--task')
    p.add_argument('--engineering',action='store_true',help='Only .local/checks/pipeline scripted-test data; never human training')
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('locations');sub.add_parser('list')
    a=sub.add_parser('process');a.add_argument('--config',type=Path);a.add_argument('--fresh',action='store_true');a.add_argument('--workers',type=int,default=2)
    a=sub.add_parser('reclip');a.add_argument('--audit',type=Path,required=True);a.add_argument('--config',type=Path)
    env=json.loads((ROOT/'config/environments.json').read_text(encoding='utf-8'))
    for command in ('preview', 'preview-menu'):
        a=sub.add_parser(command)
        a.add_argument('--port',type=int,default=env['review_port'])
        a.add_argument('--no-browser',action='store_true')
        if command == 'preview': a.add_argument('--run',type=Path)
    a=sub.add_parser('review');a.add_argument('--run',type=Path,required=True);a.add_argument('--port',type=int,default=env['review_port'])
    a=sub.add_parser('prepare');a.add_argument('--run',type=Path,required=True);a.add_argument('--review-mode',choices=['auto','reviewed'],default='auto')
    a=sub.add_parser('pack');a.add_argument('--prepared',type=Path,required=True);a.add_argument('--split-file',type=Path);a.add_argument('--output',type=Path);a.add_argument('--resume',action='store_true')
    a=sub.add_parser('verify-pack');a.add_argument('--pack',type=Path,required=True)
    args=p.parse_args(argv)
    if args.command in ('preview','preview-menu') and not 1 <= args.port <= 65535:
        p.error('--port must be within 1..65535')
    if args.command == 'preview-menu':
        if args.engineering: p.error('Use preview --run with --engineering; the menu is for human datasets')
        preview_menu(args.task,args.port,open_browser=not args.no_browser)
        return 0
    if not args.task: p.error('--task is required for this command')
    w=DatasetWorkflow(args.task,engineering=args.engineering)
    if args.command=='locations':print(json.dumps({'raw':str(w.source),'output':str(w.output),'config':str(ROOT/'config/curation.json')},ensure_ascii=False,indent=2))
    elif args.command=='list':
        for item in w.catalog(): print(item['path'])
    elif args.command=='process':print(w.process(config=args.config,fresh=args.fresh,workers=args.workers))
    elif args.command=='reclip':
        from tasks.artifacts import check_artifact
        check_artifact(args.audit,args.task,args.engineering);print(w.reclip(args.audit,args.config))
    elif args.command=='review':w.review(args.run,port=args.port)
    elif args.command=='preview':w.preview(args.run,port=args.port,open_browser=not args.no_browser)
    elif args.command=='prepare':print(w.prepare(args.run,args.review_mode))
    elif args.command=='pack':
        m=w.pack(args.prepared,args.output,split_file=args.split_file,resume=args.resume)
        print(json.dumps({'complete':m['complete'],'episodes':m['episode_count'],'frames':m['frame_count']},ensure_ascii=False))
    elif args.command=='verify-pack':
        from tasks.artifacts import check_artifact
        from dataset.task3.training_pack import verify
        check_artifact(args.pack,args.task,args.engineering);m=verify(args.pack)['manifest']
        print(json.dumps({'verified':True,'task_id':m['task_id'],'episodes':m['episode_count'],'frames':m['frame_count'],'engineering_only':m['engineering_only']},ensure_ascii=False))
    return 0

if __name__=='__main__':raise SystemExit(main())
