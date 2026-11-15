"""Task-local collection wiring; no duplicated controller implementation."""
from collection.common.adapters.dual_panda_v1 import (
    controller_id, page, backend_module, config_patch_scope, schema, create_controller, create_model,
)

# Contact supplies pushing force, not permission to latch a rigid grasp.
contact_semantics = 'nonprehensile_push'
