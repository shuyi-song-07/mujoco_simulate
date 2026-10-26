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
        temporary = path.with_suffix(".partial")
        pipeline.write_json(temporary, {"selected_version": str(Path(paths[-1]).resolve()),
            "recent_versions": [str(Path(p).resolve()) for p in paths],
            "selection_policy": POLICY, "training_ready": False})
        temporary.replace(path)
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

    def prepare(self, run, review_mode='reviewed', output=None, *, candidate=False):
        run, _ = self._version(run)
        target = Path(output).resolve() if output is not None else self.output / "prepared" / f"prepared_{'candidate_' if candidate else ''}{stamp()}"
        pipeline.prepare(run, target, review_mode=review_mode)
        if not candidate:
            pipeline.write_json(self.output / 'PREPARED_CURRENT.json', {
                'path': str(target), 'curation': str(run), 'review_mode': review_mode,
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
            pipeline.write_json(self.output / 'TRAINING_PACK_CURRENT.json', {
                'path': str(target), 'prepared': str(Path(prepared).resolve()),
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

def main(argv=None):
    p=argparse.ArgumentParser(description='Task-scoped Audit -> Safe Stitch -> Prepared -> Training Pack')
    p.add_argument('--task',required=True)
    p.add_argument('--engineering',action='store_true',help='Only .local/checks/pipeline scripted-test data; never human training')
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('locations');sub.add_parser('list')
    a=sub.add_parser('process');a.add_argument('--config',type=Path);a.add_argument('--fresh',action='store_true');a.add_argument('--workers',type=int,default=2)
    a=sub.add_parser('reclip');a.add_argument('--audit',type=Path,required=True);a.add_argument('--config',type=Path)
    env=json.loads((ROOT/'config/environments.json').read_text(encoding='utf-8'))
    a=sub.add_parser('review');a.add_argument('--run',type=Path,required=True);a.add_argument('--port',type=int,default=env['review_port'])
    a=sub.add_parser('prepare');a.add_argument('--run',type=Path,required=True);a.add_argument('--review-mode',choices=['auto','reviewed'],default='auto')
    a=sub.add_parser('pack');a.add_argument('--prepared',type=Path,required=True);a.add_argument('--split-file',type=Path);a.add_argument('--output',type=Path);a.add_argument('--resume',action='store_true')
    a=sub.add_parser('verify-pack');a.add_argument('--pack',type=Path,required=True)
    args=p.parse_args(argv);w=DatasetWorkflow(args.task,engineering=args.engineering)
    if args.command=='locations':print(json.dumps({'raw':str(w.source),'output':str(w.output),'config':str(ROOT/'config/curation.json')},ensure_ascii=False,indent=2))
    elif args.command=='list':
        for item in w.catalog(): print(item['path'])
    elif args.command=='process':print(w.process(config=args.config,fresh=args.fresh,workers=args.workers))
    elif args.command=='reclip':
        from tasks.artifacts import check_artifact
        check_artifact(args.audit,args.task,args.engineering);print(w.reclip(args.audit,args.config))
    elif args.command=='review':w.review(args.run,port=args.port)
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
