import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from datasets import load_dataset

model_path = "./models/harmony-aime-merged"

print("Fetching dataset problem...")
dataset = load_dataset(
    "DigitalLearningGmbH/MATH-lighteval",
    "algebra",
    split="train",
    trust_remote_code=True
)

sample_problem = dataset[5]["problem"]

# Load orchestrator model ONCE (still slow, but now single-purpose)
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map={"": "cpu"},
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
)

streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

prompt = f"""
System: You are a MAS compiler.

You MUST output a COMPLETE executable Python program.

Rules:
- Do NOT stop early
- Do NOT output partial classes
- Always finish full code
- End only when full MAS system is defined

User problem:
{sample_problem}

Output ONLY valid Python code.
"""

inputs = tokenizer(prompt, return_tensors="pt")

print("\n" + "="*60)
print("🧠 MAS COMPILATION RUNNING")
print("="*60)
print(f"Problem: {sample_problem}\n")

start_time = time.time()

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=1536,
        streamer=streamer,
        do_sample=False,
        temperature=0.0,
        pad_token_id=tokenizer.eos_token_id,
        early_stopping=False,
        eos_token_id=tokenizer.eos_token_id
    )

end_time = time.time()
duration = end_time - start_time

# Decode full output
mas_code = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Extract only generated part (important cleanup)
mas_code = mas_code.split("Assistant:")[-1].strip()

# Save compiled MAS program
with open("compiled_mas.py", "w") as f:
    f.write(mas_code)

print("\n" + "="*60)
print("✅ MAS COMPILATION COMPLETE")
print(f"⏱️ Time: {duration:.2f}s")
print(f"📊 Tokens: {len(outputs[0]) - len(inputs['input_ids'][0])}")
print(f"⚡ Speed: {(len(outputs[0]) - len(inputs['input_ids'][0])) / duration:.2f} tok/s")
print("="*60)