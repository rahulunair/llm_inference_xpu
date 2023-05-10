import argparse
import os
import warnings

import psutil

num_processes = psutil.cpu_count(logical=False)

os.environ["OMP_NUM_THREADS"] = str(num_processes)
os.environ["KMP_BLOCKTIME"] = "1"
os.environ["OMP_DYNAMIC"] = "false"
os.environ["KMP_AFFINITY"] = "compact,granularity=fine,1,0"
warnings.filterwarnings("ignore")

import intel_extension_for_pytorch as ipex
import torch
from torch.cpu.amp import autocast as autocast_cpu
from torch.cuda.amp import autocast as autocast_cuda
from torch.xpu.amp import autocast as autocast_xpu
from transformers import AutoModelForCausalLM, AutoTokenizer

from chat import chat_with_llm
from chat import benchmark_chat_with_llm

print(f"IPEX version: {ipex.__version__}")

torch.manual_seed(12345)
ipex.xpu.seed_all()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", default="xpu", help="Choose device: cpu, xpu, or gpu"
    )
    parser.add_argument(
        "--dtype", default="float32", help="Choose data type: float32 or float16"
    )
    parser.add_argument(
        "--model-name",
        default="togethercomputer/RedPajama-INCITE-Chat-3B-v1",
        help="LLM model name",
    )
    return parser.parse_args()


def determine_device_and_dtype(args):
    device_type = args.device.lower()
    if device_type == "xpu":
        device_type = "xpu"
    elif device_type == "gpu":
        device_type = "cuda"
    else:
        device_type = "cpu"
    dtype = args.dtype
    if dtype == "float16":
        dtype = torch.float16
    elif dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    return device_type, dtype


def load_model(args):
    print(f"Running LLM in {args.device} mode with IPEX optimization")
    print(f"Using data type {args.dtype}")
    print(f"Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=eval(f"torch.{args.dtype}"), trust_remote_code=True
    )
    model = model.to(args.device)
    print(f"Device used when loading model: {args.device}, dtype: {args.dtype}")
    model = ipex.optimize(model.eval(), dtype=eval(f"torch.{args.dtype}"))
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    stop_token_ids = frozenset(tokenizer.convert_tokens_to_ids(["<"]))
    model_end_tokens = frozenset(["<"])
    print(
        f"Model [{args.model_name}] loaded. Stop Token Ids: {stop_token_ids}. In {model.device} mode!"
    )

    return model, tokenizer, stop_token_ids, model_end_tokens


def warm_up_model(model, tokenizer, device_type, dtype, steps=2):
    warm_up_input = tokenizer(
        "This is a warm-up sentence.", return_tensors="pt"
    ).input_ids.to(device_type)
    for _ in range(steps):
        if dtype != torch.float32:
            if device_type == "xpu":
                with autocast_xpu():
                    _ = model.generate(warm_up_input)
            elif device_type == "gpu":
                with autocast_cuda():
                    _ = model.generate(warm_up_input)
            else:
                with autocast_cpu():
                    _ = model.generate(warm_up_input)
        else:
            _ = model.generate(warm_up_input)
    return model
    print("Model warmed up!")


def main():
    args = parse_arguments()
    device_type, dtype = determine_device_and_dtype(args)
    model, tokenizer, stop_token_ids, model_end_tokens = load_model(args)
    # model = warm_up_model(model, tokenizer, device_type, dtype)
    benchmark_chat_with_llm(model, tokenizer, stop_token_ids, device_type, dtype)
    #chat_with_llm(model, tokenizer, stop_token_ids, device_type, dtype)


if __name__ == "__main__":
    main()
