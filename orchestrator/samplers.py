# import os
# import dotenv
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from groq import Groq
# from openai import OpenAI


# # -----------------------
# # QWEN LOCAL MODEL
# # -----------------------
# # qwen_model_name = "Qwen/Qwen2.5-7B-Instruct"

# # qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_model_name)
# # qwen_model = AutoModelForCausalLM.from_pretrained(
# #     qwen_model_name,
# #     device_map="auto",
# #     torch_dtype=torch.float16
# # )

# # async def qwen_sampler(msg, temperature, output_fields):
# #     prompt = "\n".join([m["content"] for m in msg])
# #     inputs = qwen_tokenizer(prompt, return_tensors="pt").to(qwen_model.device)

# #     with torch.no_grad():
# #         outputs = qwen_model.generate(
# #             **inputs,
# #             max_new_tokens=512,
# #             temperature=temperature or 0.7,
# #             pad_token_id=qwen_tokenizer.eos_token_id
# #         )

# #     return qwen_tokenizer.decode(outputs[0], skip_special_tokens=True)


# # -----------------------
# # GROQ API MODEL
# # -----------------------

# dotenv.load_dotenv()

# api_key = os.getenv("GROQ_API_KEY")

# if not api_key:
#     raise ValueError("GROQ_API_KEY is not set in environment variables")

# class GroqChatCompletionSampler:
#     def __init__(self, model, temperature=0.7):
#         self.model = model
#         self.temperature = temperature
#         self.client = OpenAI(
#             api_key=os.getenv("GROQ_API_KEY"),
#             base_url="https://api.groq.com/openai/v1"  # 👈 KEY TRICK
#         )

#     async def __call__(self, msg, temperature=None, output_fields=None):
#         response = self.client.chat.completions.create(
#             model=self.model,
#             messages=msg,
#             temperature=temperature or self.temperature,
#         )
#         return response.choices[0].message.content