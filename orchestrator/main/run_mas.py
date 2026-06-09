from mas_r1_reasoner.agents.agent_system_async import AsyncAgentSystem
from mas_r1_reasoner.agents.agent_system import Info
from mas_r1_reasoner.agents.shared_vars import set_global
from mas_r1_reasoner.agents.code_sanity import validate_python_code
from mas_r1_reasoner.rewards.utils.harmony_parser.minimal import extract_harmony_code_from_response
from mas_r1_reasoner.rewards.utils.string_match_score import MathScorer

from mas_r1_reasoner.agents.sampler.chat_completion_sampler import ChatCompletionSampler
from mas_r1_reasoner.agents.sampler.groq_completion_sampler import GroqCompletionSampler
from mas_r1_reasoner.agents.sampler.cs_groq_completion_sampler import CSGroqCompletionSampler
from mas_r1_reasoner.agents.sampler.cs_chat_completion_sampler import CSChatCompletionSampler

from datasets import load_dataset
import argparse
import asyncio
import ray
import os
import numpy as np

async def main(problems, agent_model, use_cachesaver, max_debate_round, num_repeated_samples, max_reflection_round):
    
    # Create samplers depending on whether CacheSaver should be used or not    
    if use_cachesaver:
        model_sampler_map = {
            f"{agent_model}": CSChatCompletionSampler(
                model=f"{agent_model}",
                temperature=1.0,
                mock_output=False
            )
        }
    else:
        model_sampler_map = {
            f"{agent_model}": ChatCompletionSampler(
                model=f"{agent_model}",
                temperature=1.0,
                mock_output=False
            )
        }
        
    # Set up global variables required for MAS execution
    set_global("global_max_ray_workers", 4)
    set_global("global_node_model", f"{agent_model}")
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
    set_global("global_max_debate_round", max_debate_round)
    set_global("global_num_repeated_samples", num_repeated_samples)
    set_global("global_max_reflection_round", max_reflection_round)
    set_global("global_model_sampler_map", model_sampler_map)
   
   # Initialize Ray and create an asynchronous AgentSystem
    ray.init()
    system = AsyncAgentSystem.create_with_globals()

    # Load the AIME24 dataset
    dataset = load_dataset(
        "HuggingFaceH4/aime_2024",
        "default",
        split="train"
    )

    # Initialize logging metrics
    prompt_tokens_saved = 0
    prompt_tokens_used = 0
    completion_tokens_saved = 0
    completion_tokens_used = 0
    api_calls_saved = 0
    api_calls_used = 0
    
    # Scores contains a list of correct/incorrect answers
    scores = []
    
    if problems == "all":
        problems = len(dataset["problem"])
    try:
        for i in range(problems):
            problem = dataset["problem"][i]
            answer = dataset["answer"][i]

            # Create a task_info object for the execution
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

            # Open the generated plan by the orchestrator
            with open(f"orchestrator/orchestrated_plans/aime24_{i+1}.xml") as f:
                xml_plan = f.read()

            # Convert said XML plan into valid executable Python code
            code, name, thought = extract_harmony_code_from_response(
                xml_plan,
                validate_python_code,
                logger=None
            )

            if code.startswith("direct_answer"):
                print("No executable agent plan found:", thought)
            else:
                # Executing the agent plan
                completion = await system.execute_mas_batch_async(
                    [code],
                    [task_info]
                )

                (result, success, error_message, tokens) = completion[0]

                # Log the metrics
                prompt_tokens_saved += tokens["total_prompt_tokens_saved"]                
                prompt_tokens_used += tokens["total_prompt_tokens_used"]
                completion_tokens_saved += tokens["total_completion_tokens_saved"]
                completion_tokens_used += tokens["total_completion_tokens_used"]
                api_calls_saved += tokens["total_api_calls_saved"]
                api_calls_used += tokens["total_api_calls_used"]

                # Compute whether the generated answer in correct or not and append to scores
                mathscorer = MathScorer()
                correct = mathscorer.grade_answer(result, answer)                
                if correct:
                    scores.append(1)
                else:
                    scores.append(0)
                
        accuracy = np.mean(scores)
        
        system.cleanup()
        
        # Return metrics for the experiments
        return {
            "accuracy": accuracy,
            "prompt_tokens_saved_agents": prompt_tokens_saved,
            "prompt_tokens_used_agents": prompt_tokens_used,
            "completion_tokens_saved_agents": completion_tokens_saved,
            "completion_tokens_used_agents": completion_tokens_used,
            "api_calls_saved_agents": api_calls_saved,
            "api_calls_used_agents": api_calls_used  
        }
    finally:
        system.cleanup()
        ray.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-p","--problems", type=int, default="all")
    parser.add_argument("-m","--agent_model", type=str, default="gpt-5-nano-2025-08-07")
    parser.add_argument("-c","--cachesaver", action="store_true", dest="use_cachesaver")
    parser.add_argument("--max_debate_round", type=int, default=1, help="Maximum number of debate rounds for the LLM_debate")
    parser.add_argument("--num_repeated_samples", type=int, default=5, help="Number of repeated samples for the SCAgent")
    parser.add_argument("--max_reflection_round", type=int, default=5, help="Maximum reflection rounds for the ReflexionAgent")

    args = parser.parse_args()

    asyncio.run(
        main(
            problems=args.problems, 
            agent_model=args.agent_model,
            use_cachesaver=args.use_cachesaver,
            max_debate_round=args.max_debate_round,
            num_repeated_samples=args.num_repeated_samples,
            max_reflection_round=args.max_reflection_round
        )
    )