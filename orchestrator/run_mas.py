from mas_r1_reasoner.agents.agent_system_async import AsyncAgentSystem
from mas_r1_reasoner.agents.agent_system import Info
from mas_r1_reasoner.agents.shared_vars import set_global
from mas_r1_reasoner.agents.code_sanity import validate_python_code
from mas_r1_reasoner.rewards.utils.harmony_parser.minimal import extract_harmony_code_from_response
from mas_r1_reasoner.rewards.utils.string_match_score import MathScorer

from mas_r1_reasoner.agents.sampler.chat_completion_sampler import ChatCompletionSampler
from mas_r1_reasoner.agents.sampler.groq_completion_sampler import GroqCompletionSampler
from mas_r1_reasoner.agents.sampler.cs_groq_completion_sampler import CSGroqCompletionSampler

from datasets import load_dataset
import argparse
import asyncio
import ray
import os
import numpy as np

async def main(problems, agent_model, use_cachesaver):
    # Set up global variables required for MAS execution
    set_global("global_max_ray_workers", 4)

    set_global("global_node_model", f"{agent_model}")

    if use_cachesaver:
        model_sampler_map = {
            f"{agent_model}": CSGroqCompletionSampler(
                model=f"{agent_model}",
                temperature=1.0,
                mock_output=False
            )
        }
    else:
        model_sampler_map = {
            f"{agent_model}": GroqCompletionSampler(
                model=f"{agent_model}",
                temperature=1.0,
                mock_output=False
            )
        }

    print("model_sampler_map: ", model_sampler_map)

    set_global("global_model_sampler_map", model_sampler_map)

    # Set other required global variables with defaults
    set_global("global_max_round", 1)
    set_global("global_max_sc", 1)
    set_global("global_decompose_only", False)
    set_global("global_architecture_only", False)
    set_global("global_architecture_only_sequential", False)
    set_global("global_enable_tree_architecture", False)
    set_global("global_init_archive", ["COT", "COT_SC", "Reflexion", "LLM_debate"])
    set_global("global_include_blocks", False)
    set_global("global_add_judge", False)
    set_global("global_eval_building_blocks", False)
    set_global("global_known_prompt", None)
    set_global("global_multiply_processes", None)
    set_global("global_FORMAT_INST", lambda request_keys: f"""Reply EXACTLY with the following XML format.\n{str(request_keys)}\nDO NOT MISS ANY REQUEST FIELDS and ensure that your response is a well-formed XML object!\n\n""")
    set_global("global_output_description", "If the question is asked for a numeric result, Return ONLY an integer and DO NOT return anything other than the integer answer; If the question is asked for more than numeric results, Return what the question asked and make sure the answer is complete.")
    set_global("global_cot_instruction", "Please think step by step and then solve the task.")

    assert os.getenv("GROQ_API_KEY") is not None, "Missing GROQ_API_KEY"
   
    ray.init()

    system = AsyncAgentSystem.create_with_globals()

    print("Fetching dataset...")
    dataset = load_dataset(
        "HuggingFaceH4/aime_2024",
        "default",
        split="train"
    )

    # print("dataset", dataset)

    api_calls = 0

    prompt_tokens_used = 0
    # prompt_tokens_saved = 0
    completion_tokens_used = 0
    # completion_tokens_saved = 0

    scores = []
    
    if problems == "all":
        problems = len(dataset["problem"])

    print("problem amount", problems)
    try:
        for i in range(problems):
            problem = dataset["problem"][i]
            answer = dataset["answer"][i]

            print("problem: ", problem)
            print("answer: ", answer)

            print("END OF SOLUTION")

            task_info = Info(
                name="task",
                author="user",
                content=problem,
                msg=None,
                sub_tasks=[],
                agents=[],
                iteration_idx=-1,
                final_answer=None
            )

            with open(f"orchestrator/orchestrated_plans/aime24_{i+1}.xml") as f:
                xml_plan = f.read()

            code, name, thought = extract_harmony_code_from_response(
                xml_plan,
                validate_python_code,
                logger=None
            )

            # print(f"Extracted code: {code}")
            # print(f"Extracted thought: {thought}")
            # print(f"Extracted name: {name}")

            if code.startswith("direct_answer"):
                print("No executable agent plan found:", thought)
            else:
                # Run the MAS-plan
                completion = await system.execute_mas_batch_async(
                    [code],
                    [task_info]
                )

                # print("completion: ", completion)
                # print("completion[0]: ", completion[0])

                (result, success, error_message, tokens) = completion[0]
            
                print("result: ", result)
                print("success: ", success)
                print("error_message: ", error_message)
                print("tokens: ", tokens)
                
                prompt_tokens_used += tokens["total_prompt_tokens"]
                completion_tokens_used += tokens["total_completion_tokens"]
                
                print("api_calls", tokens["api_calls"])
                api_calls += tokens["api_calls"]

                print("actual answer: ", answer)

                mathscorer = MathScorer()

                print("Result: ", result)
                print("Answer: ", answer)
                correct = mathscorer.grade_answer(result, answer)

                print("correct: ", correct)
                
                if correct:
                    scores.append(1)
                else:
                    scores.append(0)
                
        accuracy = np.mean(scores)
        print("Accuracy: ", accuracy)
        
        system.cleanup()
            
        return {
            "accuracy": accuracy,
            "prompt_tokens_used_agents": prompt_tokens_used,
            "completion_tokens_used_agents": completion_tokens_used,
            "api_calls_agents": api_calls      
        }
    finally:
        system.cleanup()
        ray.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-p","--problems", type=int, default="all")
    parser.add_argument("-m","--agent_model", type=str, default="meta-llama/llama-4-scout-17b-16e-instruct")
    parser.add_argument("-c","--cachesaver", action="store_true", dest="use_cachesaver")

    args = parser.parse_args()

    asyncio.run(
        main(
            problems=args.problems, 
            agent_model=args.agent_model,
            use_cachesaver=args.use_cachesaver
        )
    )