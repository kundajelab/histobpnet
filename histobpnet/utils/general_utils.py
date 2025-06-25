from datetime import datetime
import torch
import os
import lightning as L

def get_curr_datetime_str():
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime('%Y%m%d_%H%M%S')
    return formatted_datetime

def get_timestamped_str(base_str: str):
    return f"instance-{get_curr_datetime_str()}_{base_str}"

def get_instance_id():
    return f"instance-{get_curr_datetime_str()}"

def set_random_seed(seed: int, skip_scvi: bool = True):
    L.seed_everything(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if using multi-GPU

    if not skip_scvi:
        import scvi
        scvi.settings.seed = seed
    else:
        # scvi does this so I'm just calling it here if we're not using scvi
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # ensure deterministic CUDA operations for Jax (see https://github.com/google/jax/issues/13672)
        if "XLA_FLAGS" not in os.environ:
            os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"
        else:
            os.environ["XLA_FLAGS"] += " --xla_gpu_deterministic_ops=true"