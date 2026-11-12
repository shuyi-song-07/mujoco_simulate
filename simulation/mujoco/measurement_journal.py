"""Single-arm adaptation of papera_gjb's independent raw measurement journal.

The threaded HTTP handler acknowledges a batch only after flush/fsync. Retries
are idempotent. The physics/control thread never waits on this journal's lock.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time
import queue


def jsonable(value):
    """Snapshot numpy values; never fabricate finite values for missing data."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class MeasurementJournal:
    def __init__(self, root, session, provenance):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=False)
        self.session=session;self.lock=threading.Lock();self.clients={};self.written=0;self.error=None;self.metadata_error=None
        self.configs={};self.seen={};self.client_written={};self.client_base_written={};self.provenance=provenance
        self.path=self.root/'measurements.jsonl'
        self.handle=self.path.open('x',encoding='utf-8',newline='\n')
        self._summary()

    def status(self):
        return {'schema':'single-hand-journal-v1','path':str(self.root),'written':self.written,
                'error':self.error or self.metadata_error,'clients':dict(self.clients)}

    def _summary(self):
        data={**self.status(),'session':self.session,'provenance':self.provenance,'configurations':self.configs}
        temp=self.root/'manifest.tmp'
        temp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        os.replace(temp,self.root/'manifest.json')

    def append(self,payload,receipt):
        client=payload.get('client_id');rows=payload.get('rows');report=payload.get('report')
        if not isinstance(client,str) or not 1<=len(client)<=160 or not isinstance(rows,list) or not 1<=len(rows)<=4 or not isinstance(report,dict):
            raise ValueError('invalid_measurement_batch')
        for key in ('completed','dropped_measurements','dropped','pending'):
            value=report.get(key,0)
            if type(value) is not int or value<0:raise ValueError('invalid_journal_counters')
        previous=0
        for row in rows:
            seq=row.get('sequence') if isinstance(row,dict) else None
            if type(seq) is not int or seq<=previous or row.get('kind') not in ('measurement','dispatch','camera_event','control_result'):
                raise ValueError('invalid_journal_sequence_or_kind')
            previous=seq
            if row['kind']=='measurement':
                m=row.get('measurement',{})
                if not isinstance(m,dict) or not isinstance(m.get('measurement_id'),str) or not isinstance(m.get('hands'),list):
                    raise ValueError('invalid_measurement_identity')
            json.dumps(row,allow_nan=False)
        with self.lock:
            if self.error:raise OSError(self.error)
            try:
                self.client_base_written.setdefault(client,report.get('written',0))
                for row in rows:
                    key=(client,row['sequence'])
                    digest=hashlib.sha256(json.dumps(row,sort_keys=True,allow_nan=False).encode()).hexdigest()
                    if key in self.seen:
                        if self.seen[key]!=digest:raise ValueError('journal_retry_content_changed')
                        continue
                    context=row.get('context') or {}
                    config_id=hashlib.sha256(json.dumps(context,sort_keys=True,allow_nan=False).encode()).hexdigest()
                    self.configs[config_id]=context
                    record={k:v for k,v in row.items() if k!='context'}
                    record.update(client_id=client,configuration_id=config_id,receipt=receipt,
                                  server_monotonic_ns=time.monotonic_ns(),server_epoch_ms=time.time()*1000)
                    self.handle.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
                    self.seen[key]=digest
                    self.written+=row['kind']=='measurement'
                    self.client_written[client]=self.client_written.get(client,0)+(row['kind']=='measurement')
                self.handle.flush();os.fsync(self.handle.fileno())
                self.clients[client]={**report,'server_written_measurements':self.client_written.get(client,0),
                    'unaccounted_measurements':report.get('completed',0)-self.client_base_written[client]-self.client_written.get(client,0)-report.get('dropped_measurements',0)}
            except OSError as exc:
                self.error=str(exc);raise
            # Rows are already durable. A Windows rename/sharing failure on the
            # small manifest must be retryable without duplicating those rows.
            self.metadata_error=None
            try:
                self._summary()
            except OSError as exc:
                self.metadata_error=str(exc);raise
        return {'ok':True,'last_sequence':rows[-1]['sequence'],'written':self.written}

    def validate_boundary(self, barrier):
        """A browser ACK is checked against durable server rows before saving."""
        if not isinstance(barrier,dict) or barrier.get('version')!='hand-save-boundary-v1':
            raise ValueError('hand_save_boundary_required')
        seq=barrier.get('last_sequence');client=barrier.get('client_id')
        end=barrier.get('end_capture_epoch_ms')
        if (type(seq) is not int or seq<0 or not isinstance(client,str)
                or type(end) not in (int,float) or not math.isfinite(end)
                or barrier.get('acknowledged_sequence')!=seq or barrier.get('pending')!=0
                or barrier.get('error') or barrier.get('dropped')!=0):
            raise ValueError('hand_save_boundary_incomplete')
        with self.lock:
            if self.error or self.metadata_error:raise OSError(self.error or self.metadata_error)
            prefix=[s for c,s in self.seen if c==client and 1<=s<=seq]
            # A browser can reconnect to a restarted backend with a continuing
            # sequence. Its first durable row defines this session's prefix.
            if seq and (not prefix or max(prefix)!=seq or len(prefix)!=seq-min(prefix)+1):
                raise ValueError('hand_save_boundary_not_durable')
        return float(end)

    def export(self,destination,epoch,start_ms,end_ms,barrier=None):
        """Copy this episode's measured capture interval; no resampling/filtering."""
        sealed=barrier and barrier.get('version')=='hand-save-boundary-v1'
        if sealed:end_ms=min(end_ms,self.validate_boundary(barrier))
        with self.lock:
            self.handle.flush()
            destination=Path(destination);count=0;ids=set();selected=[]
            if barrier and not sealed and isinstance(barrier.get('last_acknowledged_capture_ms'),(int,float)):
                end_ms=min(end_ms,barrier['last_acknowledged_capture_ms'])
            with self.path.open(encoding='utf-8') as source:
                for line in source:
                    row=json.loads(line);m=row.get('measurement',{})
                    if barrier and row['client_id']==barrier.get('client_id') and row['sequence']>barrier.get('acknowledged_sequence',0):continue
                    t=m.get('capture_epoch_ms')
                    if row['kind']=='measurement' and m.get('source_session')==self.session and m.get('source_control_epoch')==epoch and isinstance(t,(float,int)) and math.isfinite(t) and start_ms<=t<=end_ms:
                        selected.append(row);ids.add(m['measurement_id']);count+=1
                    elif row['kind']!='measurement' and start_ms<=row['browser_epoch_ms']<=end_ms:
                        selected.append(row)
            destination.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected),encoding='utf-8')
            summary={**self.status(),'scope':'capture_epoch_ms_in_episode','start_ms':start_ms,'end_ms':end_ms,
                'save_barrier':barrier,'source_control_epoch':epoch,'measurements':count,'unique_measurement_ids':len(ids),
                'sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),
                'provenance':self.provenance,'configurations':self.configs,
                'completeness':'inspect client completed/written/pending/dropped and dispatch/execution logs; forced browser/process termination is not recoverable'}
            destination.with_suffix('.manifest.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
            return summary

    def export_for_operation(self,destination,epoch,start_ms,end_ms,barrier=None):
        """Research failure must not strand a saved robot episode before reset.

        Keep the session journal as recovery source and explicitly flag the
        episode; a successful robot save does not certify complete hand labels.
        """
        warning=None
        try:
            summary=self.export(destination,epoch,start_ms,end_ms,barrier)
            if summary.get('error') or (barrier and (barrier.get('error') or barrier.get('pending') or barrier.get('dropped'))):
                warning='hand_measurement_upload_incomplete'
        except OSError as exc:
            warning=f'hand_measurement_export_failed: {exc}'
            summary={'measurements':None,'sha256':None,'path':str(self.root),
                     'source_control_epoch':epoch,'start_ms':start_ms,'end_ms':end_ms,'save_barrier':barrier}
        summary['research_warning']=warning
        summary['requires_review']=bool(warning)
        if warning:
            # Main episode metadata also retains this warning if even the
            # research sidecar cannot be written.
            try:
                Path(destination).with_suffix('.manifest.json').write_text(
                    json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
            except OSError:
                pass
        return summary

    def close(self):
        with self.lock:
            try:
                self.handle.flush();os.fsync(self.handle.fileno());self._summary()
            finally:
                self.handle.close()


class SingleArmResearchJournal(MeasurementJournal):
    """Durable browser measurements plus a bounded, nonblocking physics log."""

    def __init__(self, root, session, provenance):
        self.control_epoch = None
        super().__init__(root, session, provenance)
        self.events_path = self.root / "server_events.jsonl"
        self.events = queue.Queue(maxsize=8192)
        self.event_dropped = 0
        self.event_error = None
        self.event_sequence = 0
        self.worker = threading.Thread(target=self._write_events, daemon=True)
        self.worker.start()

    def record(self, kind, **fields):
        # Conversion freezes mutable arrays before physics takes another step.
        self.event_sequence += 1
        row = jsonable({"kind": kind, "server_sequence": self.event_sequence,
                        "server_epoch_ms": time.time() * 1000,
                        "server_monotonic_ns": time.monotonic_ns(), **fields})
        try:
            self.events.put_nowait(row)
        except queue.Full:
            self.event_dropped += 1

    def status(self):
        return {**super().status(),
                "server_event_dropped": getattr(self, "event_dropped", 0),
                "server_event_error": getattr(self, "event_error", None)}

    def _write_events(self):
        try:
            with self.events_path.open("x", encoding="utf-8") as output:
                while True:
                    row = self.events.get()
                    try:
                        if row is None:
                            output.flush(); os.fsync(output.fileno())
                            return
                        if row.get("kind") == "_flush":
                            output.flush(); os.fsync(output.fileno())
                            row["done"].set()
                        else:
                            output.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                            output.flush()
                    finally:
                        self.events.task_done()
        except Exception as exc:
            self.event_error = str(exc)

    def flush_events(self):
        done = threading.Event()
        self.events.put({"kind": "_flush", "done": done}, timeout=5)
        if not done.wait(10) or self.event_error:
            raise OSError(self.event_error or "server event log flush timed out")

    def export_episode(self, root, episode_index, epoch, start_ms, end_ms, barrier=None):
        self.flush_events()
        directory = Path(root) / "research"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"episode_{episode_index:06d}_hands.jsonl"
        if barrier is not None:
            self.validate_boundary(barrier)
        summary = self.export(path, epoch, start_ms, end_ms, barrier)
        event_path = directory / f"episode_{episode_index:06d}_events.jsonl"
        count = 0
        with self.events_path.open(encoding="utf-8") as source, event_path.open("w", encoding="utf-8") as out:
            for line in source:
                event = json.loads(line)
                if (event.get("control_epoch") == epoch
                        and start_ms <= event["server_epoch_ms"] <= end_ms):
                    out.write(line); count += 1
        summary.update(
            schema="single-arm-raw-research-v1", events=count,
            events_file=event_path.name,
            events_sha256=hashlib.sha256(event_path.read_bytes()).hexdigest(),
            server_event_dropped=self.event_dropped, server_event_error=self.event_error,
            requires_review=bool(not barrier or self.event_dropped or self.event_error
                                 or summary.get("error") or not summary["measurements"]),
            boundary_verified=bool(barrier),
            note="No browser boundary (e.g. keyboard/headless save) requires completeness review" if not barrier else None,
        )
        path.with_suffix(".manifest.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return summary

    def close(self):
        try:
            if self.worker.is_alive():
                self.flush_events()
                self.events.put(None, timeout=5)
                self.worker.join(timeout=10)
        finally:
            super().close()

