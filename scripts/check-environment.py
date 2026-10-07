"""Check dependencies/CUDA and replace one report in logs; no camera access."""
import json
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
import sys

import torch
import torchvision

packages = {
    'mujoco': 'mujoco', 'numpy': 'numpy', 'lerobot': 'lerobot',
    'Flask': 'flask', 'flask-cors': 'flask_cors', 'av': 'av',
    'pyarrow': 'pyarrow', 'datasets': 'datasets', 'pandas': 'pandas',
    'matplotlib': 'matplotlib', 'accelerate': 'accelerate', 'wandb': 'wandb',
    'torch': 'torch', 'torchvision': 'torchvision', 'torchcodec': 'torchcodec',
}
versions = {}
for distribution, module in packages.items():
    import_module(module)
    versions[distribution] = version(distribution)

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDataset

assert sys.version_info[:2] == (3, 12), sys.version
assert Path(sys.prefix).name == 'mujoco_repro', sys.prefix
assert torch.cuda.is_available(), 'CUDA is unavailable'
a = torch.ones((64, 64), device='cuda')
assert torch.equal(a @ a, torch.full_like(a, 64))
boxes = torch.tensor([[0., 0., 10., 10.], [0., 0., 10., 10.]], device='cuda')
kept = torchvision.ops.nms(boxes, torch.tensor([0.9, 0.8], device='cuda'), 0.5)
assert kept.tolist() == [0]
torch.cuda.synchronize()

report = {
    'python': sys.version, 'executable': sys.executable, 'prefix': sys.prefix,
    'versions': versions, 'cuda': torch.version.cuda,
    'gpu': torch.cuda.get_device_name(0), 'gpu_capability': torch.cuda.get_device_capability(0),
    'cuda_matmul': 'passed', 'torchvision_cuda_nms': 'passed',
    'act_import': ACTPolicy.__name__, 'dataset_import': LeRobotDataset.__name__,
}
output = Path(__file__).resolve().parents[1] / 'logs/environment-check.json'
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(report, ensure_ascii=False, indent=2))
