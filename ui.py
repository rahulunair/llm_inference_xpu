import gradio as gr
import torch
from main import determine_device_and_dtype, load_model
from chat import gradio_chat_with_llm

model = None
tokenizer = None
stop_token_ids = None
device_type = None
dtype = None
g_conversation_history = ""


def reset_conversation():
    global g_conversation
    g_conversation = ""


def end_chat():
    global model
    model = None


def main_chat(
    model_name,
    device,
    dtype_choice,
    do_sample,
    max_generated_tokens,
    temperature,
    top_p,
    top_k,
    repetition_penalty,
    user_input,
    reset_conversation=False,
    reset_model=False,
):
    global model, tokenizer, stop_token_ids, device_type, dtype, g_conversation_history
    print("chocies selected...")
    print(f"user_input: {user_input}")
    print(f"model_name: {model_name}")
    print(f"device: {device}")
    print(f"dtype_choice: {dtype_choice}")
    print(f"do_sample: {do_sample}")
    print(f"max_generated_tokens: {max_generated_tokens}")
    print(f"temperature: {temperature}")
    print(f"top_p: {top_p}")
    print(f"top_k: {top_k}")
    print(f"repetition_penalty: {repetition_penalty}")
    print(f"reset_conversation: {reset_conversation}")
    print(f"reset_model: {reset_model}")
    top_k = int(top_k)
    max_generated_tokens = int(max_generated_tokens)
    if model is None:
        device_type, dtype = determine_device_and_dtype(device, dtype_choice)
        model, tokenizer, stop_token_ids, _ = load_model(model_name, device_type, dtype)
        return "Model loaded successfully!"

    if reset_model:
        end_chat()
        return "Model has been reset. Please load the model again using the 'Load Model' button."
    if reset_conversation:
        reset_conversation()
        g_conversation_history = ""
    else:
        response = gradio_chat_with_llm(
            model=model,
            tokenizer=tokenizer,
            stop_token_ids=stop_token_ids,
            user_input=user_input,
            device=device_type,
            dtype=dtype,
            do_sample=do_sample,
            max_generated_tokens=max_generated_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
        g_conversation_history += f"User: {user_input}\nLLM Bot: {response}\n"
        return g_conversation_history


model_choices = gr.inputs.Dropdown(
    [
        "togethercomputer/RedPajama-INCITE-Chat-3B-v1",
        "cerebras/Cerebras-GPT-2.7B",
        "cerebras/Cerebras-GPT-1.3B",
        "EleutherAI/gpt-neo-2.7B",
    ],
    label="Model",
)
device_choices = gr.inputs.Dropdown(["cpu", "xpu", "cuda"], label="Device")
dtype_choices = gr.inputs.Dropdown(
    ["float32", "float16", "bfloat16"], label="Data Type"
)
do_sample = gr.inputs.Checkbox(True, label="Enable Sampling")
max_generated_tokens = gr.inputs.Number(50, label="Max Generated Tokens")
temperature = gr.inputs.Slider(0.1, 1.0, step=0.1, label="Temperature")
top_p = gr.inputs.Slider(0.5, 1.0, step=0.05, label="Top P")
top_k = gr.inputs.Number(80, label="Top K")
repetition_penalty = gr.inputs.Slider(1.0, 2.0, step=0.1, label="Repetition Penalty")
reset_conversation_button = gr.inputs.Checkbox(False, label="Reset Conversation")
reset_model_button = gr.inputs.Checkbox(False, label="End Chat")


gr.Interface(
    fn=main_chat,
    inputs=[
        model_choices,
        device_choices,
        dtype_choices,
        do_sample,
        max_generated_tokens,
        temperature,
        top_p,
        top_k,
        repetition_penalty,
        gr.inputs.Textbox(lines=5, placeholder="Type your message here..."),
        reset_conversation_button,
        reset_model_button,
    ],
    outputs=gr.outputs.Textbox(label="Chatbot Response"),
    title="Large Language Model Chat Interface",
).launch()
