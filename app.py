import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

def chatbot(message, history):

    prompt = ""

    for user, bot in history:
        prompt += f"User: {user}\nAssistant: {bot}\n"

    prompt += f"User: {message}\nAssistant:"

    result = pipe(
        prompt,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

    text = result[0]["generated_text"]
    answer = text[len(prompt):].strip()

    return answer

demo = gr.ChatInterface(
    fn=chatbot,
    title="🤖 AI Chatbot",
    description="AI Chatbot chạy trên Hugging Face",
    theme="soft"
)

demo.launch()
