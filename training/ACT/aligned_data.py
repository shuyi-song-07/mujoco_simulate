"""LeRobot single-arm reader around GJB's train-only circular-RPY normalization.

Existing recorded-row action timing and gripper command units are preserved.
This is not the dual-arm Safe Stitch Pack contract.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

IMAGE_KEY = 'observation.images.overview'


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def wrap_actions(values, reference):
    result = np.asarray(values, dtype=np.float64).copy()
    result[..., 3:6] = reference[3:6] + (result[..., 3:6]-reference[3:6]+np.pi) % (2*np.pi)-np.pi
    return result


class Normalizer:
    def __init__(self, stats):
        self.stats = stats
        for key in ('state_mean','state_std','action_mean','action_std','action_angle_reference'):
            setattr(self, key, np.asarray(stats[key], dtype=np.float64))

    def encode_state(self, values):
        return ((values-self.state_mean)/self.state_std).astype(np.float32)

    def encode_action(self, values):
        return ((wrap_actions(values,self.action_angle_reference)-self.action_mean)/self.action_std).astype(np.float32)

    def decode_action(self, values):
        result = np.asarray(values)*self.action_std+self.action_mean
        result[...,3:6] = (result[...,3:6]+np.pi) % (2*np.pi)-np.pi
        return result.astype(np.float32)


class SingleArmData:
    def __init__(self, root, eval_split=.1, split_file=None, fingerprint=True):
        self.root = Path(root).resolve()
        self.info = json.loads((self.root/'meta/info.json').read_text())
        features = self.info['features']
        if (features['observation.state']['shape'] != [9] or features['action']['shape'] != [7]
                or IMAGE_KEY not in features):
            raise ValueError('Aligned single-arm ACT requires S9/A7 and overview RGB')
        paths = sorted((self.root/'data').rglob('*.parquet'))
        table = pa.concat_tables([pq.read_table(p,columns=['index','episode_index','observation.state','action']) for p in paths])
        rows = table.sort_by('index').to_pydict()
        self.state = np.asarray(rows['observation.state'], dtype=np.float64)
        self.action = np.asarray(rows['action'], dtype=np.float64)
        self.episode = np.asarray(rows['episode_index'], dtype=np.int64)
        if (not np.array_equal(rows['index'],np.arange(len(self.state)))
                or not np.isfinite(self.state).all() or not np.isfinite(self.action).all()):
            raise ValueError('Dataset indices must be contiguous; state/action must be finite')
        episodes = sorted(set(self.episode.tolist()))
        if split_file:
            split = json.loads(Path(split_file).read_text())
            self.split = {key:list(split[key]) for key in ('train','val')}
            self.split_unit = split.get('unit','explicit_episode_membership')
        else:
            if not 0 < eval_split < 1 or len(episodes) < 2:
                raise ValueError('Need at least two episodes and 0 < eval_split < 1')
            n = math.ceil(len(episodes)*eval_split)
            self.split = {'train':episodes[:-n], 'val':episodes[-n:]}
            self.split_unit = 'episode_last_holdout; not a certified session split'
        train,val = (self.split[k] for k in ('train','val'))
        if (not train or not val or set(train)&set(val) or len(set(train))!=len(train)
                or len(set(val))!=len(val) or set(train+val)!=set(episodes)):
            raise ValueError('Split must partition all episodes into nonempty disjoint train/val')
        mask = np.isin(self.episode,train)
        action = self.action[mask]
        reference = np.zeros(7)
        reference[3:6] = np.arctan2(np.sin(action[:,3:6]).mean(0),np.cos(action[:,3:6]).mean(0))
        circular = wrap_actions(action,reference)
        self.stats = {'fit_split':'train','std_floor':.01,
            'state_mean':self.state[mask].mean(0).tolist(), 'state_std':np.maximum(self.state[mask].std(0),.01).tolist(),
            'action_mean':circular.mean(0).tolist(), 'action_std':np.maximum(circular.std(0),.01).tolist(),
            'action_angle_reference':reference.tolist()}
        self.normalizer = Normalizer(self.stats)
        self.bounds = {}
        for ep in episodes:
            ids = np.flatnonzero(self.episode==ep)
            if not np.array_equal(ids,np.arange(ids[0],ids[-1]+1)):
                raise ValueError('Episode frames are not contiguous')
            self.bounds[ep]=(int(ids[0]),int(ids[-1])+1)
        self.identity = None
        if fingerprint:
            files = [self.root/'meta/info.json',*paths,*sorted((self.root/'videos').rglob('*.mp4')),
                     *sorted((self.root/'images').rglob('*.png')),*sorted((self.root/'images').rglob('*.jpg'))]
            self.identity = [{'path':str(p.relative_to(self.root)), 'sha256':file_hash(p)} for p in files]


class SingleArmDataset(Dataset):
    def __init__(self, data, split, chunk_size, repo_id=None):
        self.data,self.chunk_size,self.repo_id = data,chunk_size,repo_id
        self.indices = np.flatnonzero(np.isin(data.episode,data.split[split]))
        self.base = None

    def __getstate__(self):
        return {**self.__dict__, 'base':None}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        if self.base is None:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            self.base = LeRobotDataset(self.repo_id or f'local/{self.data.root.name}',root=self.data.root,video_backend='pyav')
            # Reader metadata is in-memory only: decode the sole policy camera,
            # rather than paying to decode the two unused recorded views.
            features = self.base.meta.features
            for key in list(features):
                if key.startswith('observation.images.') and key!=IMAGE_KEY:
                    del features[key]
        anchor = int(self.indices[index])
        _,end = self.data.bounds[int(self.data.episode[anchor])]
        positions = np.arange(anchor,anchor+self.chunk_size)
        pad = positions>=end
        action = self.data.normalizer.encode_action(self.data.action[np.minimum(positions,end-1)])
        action[pad]=0
        image = self.base[anchor][IMAGE_KEY]
        if image.ndim!=3 or image.shape[0]!=3 or not torch.isfinite(image).all():
            raise ValueError('Expected finite CHW RGB input')
        return {'images':image.unsqueeze(0),
                'state':torch.from_numpy(self.data.normalizer.encode_state(self.data.state[anchor])),
                'action':torch.from_numpy(action),'is_pad':torch.from_numpy(pad)}
