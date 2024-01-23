import torch
from transformers import pipeline

pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")

# We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
messages = [
    {
        "role": "system",
        "content": """Your task is to answer in a consistent style. You a funny chatbot that writes with Australian slangs.
        """,
    },
    {"role": "user", "content": "What you like to do?"},
]
prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
outputs = pipe(prompt, max_new_tokens=1024, do_sample=True, temperature=0.3, top_k=20, top_p=0.90)
print(outputs[0]["generated_text"].replace(prompt, ""))
