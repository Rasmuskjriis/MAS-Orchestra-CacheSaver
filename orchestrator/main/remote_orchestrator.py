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

# Helper functions that allow the agents to talk to the LLM with or without CacheSaver
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

    # Create a prompt to the orchestrator based on the templates in the "agents_prompts.py" file
    # These include the orchestrator having access to all sub-agents or only one at a time
    MATH_SYSTEM_PROMPT, MATH_USER_PROMPT_TEMPLATE, MATH_USER_SUFFIX = agent_prompt(agent_type)

    # Helper function to collect into one prompt
    def build_math_messages(question):
        return [
          {"role": "system", "content": MATH_SYSTEM_PROMPT},
          {"role": "user", "content": MATH_USER_PROMPT_TEMPLATE.replace("{question}", question)},
          {"role": "user", "content": MATH_USER_SUFFIX},
      ]

    # The URL that hosts the orchestrator LLM
    BASE_URL = "https://discern-stroller-recycling.ngrok-free.dev/v1"

    # Create a client based on whether CacheSaver should be used or not
    if use_cachesaver:
        client = _CacheSaverOpenAI(
            base_url=BASE_URL,
            api_key="dummy",
            namespace="",
            cachedir="./cache"
        )
    else:
        client = _OpenAI(
            base_url=BASE_URL,
            api_key="dummy"
        )

    # Load the dataset
    dataset = load_dataset(
        "HuggingFaceH4/aime_2024",
        "default",
        split="train"
    )

    # Initialize logging metrics
    prompt_tokens_used = 0
    prompt_tokens_saved = 0
    completion_tokens_used = 0
    completion_tokens_saved = 0
    api_calls_saved = 0
    api_calls_used = 0

    start = time.time()

    if problems == "all":
        problems = len(dataset["problem"])

    for i in range(problems):
        
        problem = dataset["problem"][i]
        
        # Build the message to the orchestrator using a helper function
        messages = build_math_messages(problem)

        if use_cachesaver:
            (response, metadata) = create_chat_completion_with_cs(client, model, messages)
        else:
            response = create_chat_completion(client, model, messages)
            metadata = make_dummy_metadata() # Make dummy metadata if CacheSaver is not used

        # Log tokens and number of API calls
        usage = getattr(response, "usage", None)
        tokens = calculate_saved_tokens(usage, metadata)

        prompt_tokens_saved += tokens["prompt_tokens_saved"]                
        prompt_tokens_used += tokens["prompt_tokens_used"]
        completion_tokens_saved += tokens["completion_tokens_saved"]
        completion_tokens_used += tokens["completion_tokens_used"]
        api_calls_saved += tokens["api_calls_saved"]
        api_calls_used += tokens["api_calls_used"]

        output = response.choices[0].message.content

        # Only take the agent plan part of the output from the orchestrator
        end_tag = "</answer>"
        end_idx = output.rfind(end_tag)

        if end_idx != -1:
            xml_content = output[:end_idx + len(end_tag)]
        else:
            xml_content = output


        # Save output to file
        OUTPUT_XML = f"orchestrator/orchestrated_plans/aime24_{i+1}.xml"

        with open(OUTPUT_XML, "w") as f:
            f.write(xml_content)

    end = time.time()
    
    # Return the metrics for our experiments
    return {
        "prompt_tokens_saved_orc": prompt_tokens_saved,
        "prompt_tokens_used_orc": prompt_tokens_used,
        "completion_tokens_saved_orc": completion_tokens_saved,
        "completion_tokens_used_orc": completion_tokens_used,
        "api_calls_saved_orc": api_calls_saved,
        "api_calls_used_orc": api_calls_used,
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