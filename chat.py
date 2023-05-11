import timeit
import intel_extension_for_pytorch
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList

from utils import StopOnTokens, infer

HUMAN_START = "<human>: "
BOT_START = "<bot>: "
END_CONVO = "\n"
MAX_PRINT_LENGTH = 70


g_conversation = ""


def gradio_chat_with_llm(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    stop_token_ids: list,
    user_input,
    device="xpu",
    dtype=torch.float32,
    do_sample=True,
    max_generated_tokens=50,
    temperature=0.1,
    top_p=0.95,
    top_k=80,
    repetition_penalty=1.5,
    reset=False,
):
    global g_conversation
    if reset:
        g_conversation = ""
    g_conversation += f"{HUMAN_START}{user_input}{END_CONVO}{BOT_START}"
    input_ids = tokenizer(g_conversation).input_ids
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    generation_kwargs = dict(
        do_sample=do_sample,
        input_ids=input_ids,
        max_new_tokens=max_generated_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        pad_token_id=0,
        stopping_criteria=StoppingCriteriaList([StopOnTokens(stop_token_ids)]),
    )
    generated_tokens = infer(model, dtype, device, generation_kwargs)
    generated_text = tokenizer.decode(
        generated_tokens[0][-max_generated_tokens:], skip_special_tokens=True
    )
    return generated_text


def chat_with_llm(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    stop_token_ids: frozenset,
    device_type: str,
    dtype: torch.dtype,
    do_sample: bool = True,
    max_generated_tokens: int = 50,
    temperature: float = 0.1,
    top_p: float = 0.95,
    top_k: int = 80,
    repetition_penalty: float = 1.5,
):
    generation_kwargs = dict(
        do_sample=do_sample,
        input_ids=None,
        max_new_tokens=max_generated_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        pad_token_id=0,
    )

    print("=" * MAX_PRINT_LENGTH)
    print("=" * MAX_PRINT_LENGTH)
    print("Type something then press ENTER! Type exit to stop the chat!")
    print("user >> ) ", end="")
    conversation = ""
    while (prompt := input()) != "exit":
        conversation += f"{HUMAN_START}{prompt}{END_CONVO}{BOT_START}"
        input_ids = tokenizer(conversation).input_ids
        input_ids = torch.tensor(input_ids).to(device_type).unsqueeze(0)
        generation_kwargs["input_ids"] = input_ids
        generation_kwargs["stopping_criteria"] = StoppingCriteriaList(
            [StopOnTokens(stop_token_ids)]
        )
        generated_text = ""
        print("🤖🤖>> ) ", end="")
        generated_tokens = infer(model, dtype, device_type, generation_kwargs)
        generated_text = tokenizer.decode(
            generated_tokens[0][-max_generated_tokens:], skip_special_tokens=True
        )
        print(generated_text, end="")
        conversation += f"{generated_text}{END_CONVO}"
        print("\nQ)? ", end="")


def benchmark_chat_with_llm(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    stop_token_ids: frozenset,
    device_type: str,
    dtype: torch.dtype,
    max_generated_tokens: int = 50,
    temperature: float = 0.1,
    top_p: float = 0.95,
    top_k: int = 80,
    repetition_penalty: float = 1.5,
    num_runs: int = 10,
):
    def _run_chat_with_llm():
        prompt = "Hello, how are you?"
        conversation = f"{HUMAN_START}{prompt}{END_CONVO}{BOT_START}"
        input_ids = tokenizer(conversation).input_ids
        input_ids = torch.tensor(input_ids).to(device_type).unsqueeze(0)
        generation_kwargs = dict(
            do_sample=True,
            input_ids=input_ids,
            max_new_tokens=max_generated_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            pad_token_id=0,
            stopping_criteria=StoppingCriteriaList([StopOnTokens(stop_token_ids)]),
        )
        generated_tokens = infer(model, dtype, device_type, generation_kwargs)

    total_time = timeit.timeit(_run_chat_with_llm, number=num_runs)
    average_time = total_time / num_runs
    print(f"Average time per run: {average_time:.6f} seconds")
