"""push_r: reuse the unchanged paired-Panda execution math."""
from rollout.common.dual_executor import PolicyExecutor

def create_executor(controller, config=None):
    return PolicyExecutor(controller, config)
