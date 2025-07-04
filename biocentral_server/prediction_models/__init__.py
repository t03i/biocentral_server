import torch.multiprocessing as torch_mp

from .biotrainer_task import BiotrainerTask
from .prediction_models_endpoint import prediction_models_service_route

# Spawn new cuda contexts instead of forking:
# https://stackoverflow.com/questions/72779926/gunicorn-cuda-cannot-re-initialize-cuda-in-forked-subprocess
torch_mp.set_start_method("spawn", force=True)

__all__ = [
    "BiotrainerTask",
    "prediction_models_service_route",
]
