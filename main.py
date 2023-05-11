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
        "--device", default="xpu", help="Choose device: cpu, xpu, or cuda"
    )
    parser.add_argument(
        "--dtype", default="float32", help="Choose data type: float32 or float16"
    )
    parser.add_argument(
        "--model-name",
        default="togethercomputer/RedPajama-INCITE-Chat-3B-v1",
        help="LLM model name",
    )
    parser.add_argument(
        "--do_sample", action="store_true", help="Enable sampling in generation"
    )
    parser.add_argument(
        "--max_generated_tokens",
        type=int,
        default=50,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature for generation",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top p for nucleus sampling",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=80,
        help="Top k for k-top sampling",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.5,
        help="Repetition penalty for generation",
    )
    parser.add_argument(
        "--bench",
        action="store_true",
        help="benchmark llm",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="deploy an interactive chatbot",
    )
    return parser.parse_args()


def determine_device_and_dtype(device, dtype):
    device_type = device.lower()
    if device_type == "xpu":
        device_type = "xpu"
    elif device_type == "cuda":
        device_type = "cuda"
    else:
        device_type = "cpu"
    dtype = dtype
    if dtype == "float16":
        dtype = torch.float16
    elif dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    return device_type, dtype


def load_model(model_name, device, dtype):
    print(f"Running LLM in {device} mode with IPEX optimization")
    print(f"Using data type {dtype}")
    print(f"Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, trust_remote_code=True
    )
    model = model.to(device)
    print(f"Device used when loading model: {device}, dtype: {dtype}")
    model = ipex.optimize(model.eval(), dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    stop_token_ids = frozenset(tokenizer.convert_tokens_to_ids(["<"]))
    model_end_tokens = frozenset(["<"])
    print(
        f"Model [{model_name}] loaded. Stop Token Ids: {stop_token_ids}. In {device} mode!"
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
    device_type, dtype = determine_device_and_dtype(args.device, args.dtype)
    model, tokenizer, stop_token_ids, _ = load_model(
        args.model_name, device_type, dtype
    )
    # model = warm_up_model(model, tokenizer, args.device, args.dtype, steps=2)
    # benchmark_chat_with_llm(model, tokenizer, stop_token_ids, args.device, args.dtype)
    chat_with_llm(
        model,
        tokenizer,
        stop_token_ids,
        args.device,
        args.dtype,
        args.do_sample,
        args.max_generated_tokens,
        args.temperature,
        args.top_p,
        args.top_k,
        args.repetition_penalty,
    )


if __name__ == "__main__":
    main()
