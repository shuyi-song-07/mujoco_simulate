"""Cross-stage provenance checks; task identity never comes from directory time."""
import copy
import json
from pathlib import Path
from tasks.registry import ROOT, validate_binding, hook
from tasks.recorded_collection import PROOF_FIELD, recorded_collection_provenance

UNION_FIELDS = ('dataset_collection_compatibility', 'source_task_bindings', 'source_prepared_manifests')


def _validate_prepared_sources(manifest):
    if manifest.get('pipeline_stage') == 'prepared' and any(k in manifest for k in UNION_FIELDS):
        from dataset.task3.merge_prepared import validate_manifest
        validate_manifest(manifest)

def source_contract(refs):
    bindings=[]; engineering=[]
    for ref in refs:
        p=Path(ref['metadata_path'])
        m=json.loads(p.read_text(encoding='utf-8'))
        binding=m.get('task_binding')
        validate_binding(binding)
        if m.get('task_id')!=binding['task_id'] or m['config'].get('task_binding')!=binding:
            raise ValueError(f'Task identity mismatch: {p}')
        hook(m['config'],'dataset_adapter')().validate(m)
        bindings.append(binding);engineering.append(bool(ref.get('engineering')))
    if not bindings or any(b!=bindings[0] for b in bindings) or len(set(engineering))!=1:
        raise ValueError('Dataset must contain one exact task version and one provenance kind')
    result = {'task_id':bindings[0]['task_id'],'task_binding':bindings[0],
              'engineering_only':engineering[0]}
    proof = recorded_collection_provenance(bindings[0])
    if proof is not None:
        result[PROOF_FIELD] = proof
    return result

def fields(manifest):
    result={k:manifest[k] for k in ('task_id','task_binding','engineering_only')}
    validate_binding(result['task_binding'],result['task_id'])
    if PROOF_FIELD in manifest:
        result[PROOF_FIELD] = copy.deepcopy(manifest[PROOF_FIELD])
    _validate_prepared_sources(manifest)
    for key in UNION_FIELDS:
        if key in manifest:
            result[key] = copy.deepcopy(manifest[key])
    return result

def check_artifact(path,task_id,engineering=False):
    p=Path(path).resolve()
    m=json.loads((p/'manifest.json').read_text(encoding='utf-8'))
    validate_binding(m.get('task_binding'),task_id)
    if m.get('task_id')!=task_id:raise ValueError('Cross-task manifest rejected')
    if bool(m.get('engineering_only'))!=bool(engineering):
        raise ValueError('Engineering and human datasets cannot be mixed; select the explicit engineering scope')
    _validate_prepared_sources(m)
    return m
