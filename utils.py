import time
from contextlib import contextmanager
from typing import List

import intel_extension_for_pytorch
import torch
from torch.cpu.amp import autocast as autocast_cpu
from torch.cuda.amp import autocast as autocast_cuda
from torch.xpu.amp import autocast as autocast_xpu
from transformers import StoppingCriteria


@contextmanager
def autocast_context(device_type: str, dtype: torch.dtype):
    if dtype != torch.float32:
        if device_type == "xpu":
            with autocast_xpu():
                yield
        elif device_type == "gpu":
            with autocast_cuda():
                yield
        else:
            with autocast_cpu():
                yield
    else:
        yield


def infer(model, dtype: torch.dtype, device_type: str, input_text) -> torch.Tensor:
    print(" thinking...")
    t1 = time.time()
    with autocast_context(device_type, dtype):
        output = model.generate(**input_text)
    if device_type == "xpu":
        torch.xpu.synchronize()
    t2 = time.time()
    print(f"time for generation: {t2 - t1}")
    return output


class StopOnTokens(StoppingCriteria):
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        return input_ids.ravel()[-1].item() in self.stop_token_ids

    def __init__(self, stop_token_ids: List[int]):
        self.stop_token_ids = stop_token_ids
