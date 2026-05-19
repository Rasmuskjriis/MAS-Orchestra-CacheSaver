import time
from datasets import load_dataset
from openai import OpenAI

import argparse
import asyncio

from openai import AsyncOpenAI as _AsyncOpenAI
from openai import OpenAI as _OpenAI
from cachesaver.models.openai import AsyncOpenAI as _CacheSaverAsyncOpenAI
from cachesaver.models.openai import OpenAI as _CacheSaverOpenAI
from orchestrator.prompts.agent_prompts import agent_prompt

from orchestrator.utils.utils import calculate_saved_tokens, make_dummy_metadata

def create_chat_completion(client, model, messages):
     response = client.chat.completions.create(
        model=f"{model}",
        messages=messages,
        temperature=0.7,
        max_tokens=4096
    )
     return response

def create_chat_completion_with_cs(client, model, messages):
    (response, metadata) = client.chat.completions.create(
        model=f"{model}",
        messages=messages,
        temperature=0.7,
        max_tokens=4096,
        metadata = True
    )
    return (response, metadata)

async def main(agent_type, problems, model, use_cachesaver):
  # =========================
  # 1. MAS PROMPT
  # =========================


#   agent_type = "CoTAgent"  
#   agent_type = "SCAgent"
#   agent_type = "DebateAgent"
#   agent_type = "ReflexionAgent" # uses early exit right now
#   agent_type = "All"

    MATH_SYSTEM_PROMPT, MATH_USER_PROMPT_TEMPLATE, MATH_USER_SUFFIX = agent_prompt(agent_type)

    def build_math_messages(question):
        return [
          {"role": "system", "content": MATH_SYSTEM_PROMPT},
          {"role": "user", "content": MATH_USER_PROMPT_TEMPLATE.replace("{question}", question)},
          {"role": "user", "content": MATH_USER_SUFFIX},
      ]

    BASE_URL = "https://discern-stroller-recycling.ngrok-free.dev/v1"

    if use_cachesaver:
        client = _CacheSaverOpenAI(
            base_url=BASE_URL,
            api_key="dummy",  # required but ignored by server
            namespace="",
            cachedir="./cache"
        )
    else:
        client = _OpenAI(
            base_url=BASE_URL,
            api_key="dummy"  # required but ignored by server
        )

    # =========================
    # 1. Load dataset problem
    # =========================

    print("Fetching dataset...")
    dataset = load_dataset(
        "HuggingFaceH4/aime_2024",
        "default",
        split="train"
    )

    # =========================
    # 4. Generate XML
    # =========================

    print("\n==============================")
    print("🧠 GENERATING MAS XML PLAN")
    print("==============================\n")

    prompt_tokens_used = 0
    prompt_tokens_saved = 0
    completion_tokens_used = 0
    completion_tokens_saved = 0
    api_calls = 0

    start = time.time()

    print("Dataset: ", dataset)
    # print("Dataset length: ", len(dataset[:1]))
    print("problems", dataset["problem"])
    print("answer: ", dataset["answer"])

    if problems == "all":
        problems = len(dataset["problem"])

    print("problem amount", problems)

    for i in range(problems):
        
        problem = dataset["problem"][-1]
        
        # problem += "Please use exactly  3 agents to debate this"
        
        messages = build_math_messages(problem)

        if use_cachesaver:
            (response, metadata) = create_chat_completion_with_cs(client, model, messages)
            print("METADATA: ", metadata)
        else:
            response = create_chat_completion(client, model, messages)
            metadata = make_dummy_metadata()

        usage = getattr(response, "usage", None)
        tokens = calculate_saved_tokens(usage, metadata)

        prompt_tokens_saved += tokens["prompt_tokens_saved"]                
        prompt_tokens_used += tokens["prompt_tokens_used"]
        completion_tokens_saved += tokens["completion_tokens_saved"]
        completion_tokens_used += tokens["completion_tokens_used"]
        if tokens.get('api_call'):
            api_calls += 1

        print(f"Model: {response.model}")
        print(f"Tokens: {response.usage.prompt_tokens} prompt, {response.usage.completion_tokens} completion")
        print(f"\n--- Response ---\n")
        print(response.choices[0].message.content)

        output = response.choices[0].message.content

        end_tag = "</answer>"
        end_idx = output.rfind(end_tag)

        if end_idx != -1:
            xml_content = output[:end_idx + len(end_tag)]
        else:
            # fallback if model is broken
            xml_content = output


        # Save output to file
        OUTPUT_XML = f"orchestrator/orchestrated_plans/aime24_{i+1}.xml"

        with open(OUTPUT_XML, "w") as f:
            f.write(xml_content)

    end = time.time()

    print("\n==============================")
    print("✅ XML PLANS SAVED")
    print(f"⏱️ Time: {end - start:.2f}s")
    print("==============================")
    
    print("prompt_tokens_used_orc", prompt_tokens_used)
    print("completion_tokens_used_orc", completion_tokens_used)
    
    return {
        "prompt_tokens_saved_orc": prompt_tokens_saved,
        "prompt_tokens_used_orc": prompt_tokens_used,
        "completion_tokens_saved_orc": completion_tokens_saved,
        "completion_tokens_used_orc": completion_tokens_used,
        "api_calls_orc": api_calls
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-a","--agent_type", type=str, default="CoTAgent")
    parser.add_argument("-p","--problems", type=int, default="all")
    parser.add_argument("-m","--model", type=str, default="math")
    parser.add_argument("-c","--cachesaver", action="store_true", dest="use_cachesaver")

    args = parser.parse_args()

    asyncio.run(
        main(
            agent_type=args.agent_type,
            problems=args.problems, 
            model=args.model,
            use_cachesaver=args.use_cachesaver
        )
    )