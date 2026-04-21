import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from datasets import load_dataset

model_path = "./models/harmony-aime-merged"

# 1. Fetch a problem from MATH-lighteval
print("Fetching dataset problem...")
dataset = load_dataset("DigitalLearningGmbH/MATH-lighteval", "algebra", split="train", trust_remote_code=True)
sample_problem = dataset[5]["problem"]

# 2. Load Model and Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map={"": "cpu"}, 
    torch_dtype=torch.bfloat16, 
    low_cpu_mem_usage=True,
    trust_remote_code=True
)

# 3. Setup Streamer
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# 4. Prepare Prompt
prompt = f"System: You are a mathematical reasoning assistant. Solve the following problem step-by-step.\nUser: {sample_problem}\nAssistant:"
inputs = tokenizer(prompt, return_tensors="pt")

print(f"\n--- Solving Dataset Problem ---")
print(f"Problem: {sample_problem}")
print("\nThinking (Streaming output below)...")
print("-" * 30)

# 5. GENERATION WITH TIMER
start_time = time.time() # Start the clock

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        streamer=streamer,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )

end_time = time.time() # Stop the clock
total_duration = end_time - start_time
tokens_generated = len(outputs[0]) - len(inputs["input_ids"][0])
tokens_per_second = tokens_generated / total_duration

print("-" * 30)
print(f"Inference Complete.")
print(f"Total Time: {total_duration:.2f} seconds")
print(f"Tokens Generated: {tokens_generated}")
print(f"Average Speed: {tokens_per_second:.2f} tokens/sec")