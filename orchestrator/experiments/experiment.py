import unittest
import asyncio
import pandas as pd
import time

import orchestrator.main.remote_orchestrator as orchestrator
import orchestrator.main.run_mas as run
from orchestrator.utils.utils import tokens_to_cost

class Test(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        asyncio.get_running_loop().set_debug(False)
        self.results = []
    
    async def experiment(self, agent_type, problems, model, agent_model, use_cachesaver, max_debate_round, num_repeated_samples, max_reflection_round):
        start_time = time.time()
        result_orc = await orchestrator.main(agent_type, problems, model, use_cachesaver)
        result_agents = await run.main(problems, agent_model, use_cachesaver, max_debate_round, num_repeated_samples, max_reflection_round)
        end_time = time.time()
        
        runtime = end_time - start_time
        
        input_cost_saved_orc, output_cost_saved_orc, total_cost_saved_orc = tokens_to_cost(result_orc["prompt_tokens_saved_orc"], result_orc["completion_tokens_saved_orc"], agent_model)
        input_cost_used_orc, output_cost_used_orc, total_cost_used_orc = tokens_to_cost(result_orc["prompt_tokens_used_orc"], result_orc["completion_tokens_used_orc"], agent_model)
        input_cost_saved_agents, output_cost_saved_agents, total_cost_saved_agents = tokens_to_cost(result_agents["prompt_tokens_saved_agents"], result_agents["completion_tokens_saved_agents"], agent_model)
        input_cost_used_agents, output_cost_used_agents, total_cost_used_agents = tokens_to_cost(result_agents["prompt_tokens_used_agents"], result_agents["completion_tokens_used_agents"], agent_model)
        
        row = {
            # Metrics
            "max_debate_round": max_debate_round,
            "num_repeated_samples": num_repeated_samples,
            "max_reflection_round": max_reflection_round,
            "problems": problems,
            "use_cachesaver": use_cachesaver,
            "accuracy": round(result_agents["accuracy"], 2),
            
            # Orchestrator - Saved
            "prompt_tokens_saved_orc": result_orc["prompt_tokens_saved_orc"],
            "input_cost_saved_orc ($)": input_cost_saved_orc,
            "completion_tokens_saved_orc": result_orc["completion_tokens_saved_orc"],
            "output_cost_saved_orc ($)": output_cost_saved_orc,
            
            # Orchestrator - Used
            "prompt_tokens_used_orc": result_orc["prompt_tokens_used_orc"],
            "input_cost_used_orc ($)": input_cost_used_orc,
            "completion_tokens_used_orc": result_orc["completion_tokens_used_orc"],
            "output_cost_used_orc ($)": output_cost_used_orc,
            
            # Agents - Saved
            "prompt_tokens_saved_agents": result_agents["prompt_tokens_saved_agents"],
            "input_cost_saved_agents ($)": input_cost_saved_agents,
            "completion_tokens_saved_agents": result_agents["completion_tokens_saved_agents"],
            "output_cost_saved_agents ($)": output_cost_saved_agents,
            
            # Agents - Used
            "prompt_tokens_used_agents": result_agents["prompt_tokens_used_agents"],
            "input_cost_used_agents ($)": input_cost_used_agents,
            "completion_tokens_used_agents": result_agents["completion_tokens_used_agents"],
            "output_cost_used_agents ($)": output_cost_used_agents,
            
            # Api calls and runtime
            "api_calls_saved_orc" : result_orc["api_calls_saved_orc"],
            "api_calls_used_orc" : result_orc["api_calls_used_orc"],
            "api_calls_saved_agents" : result_agents["api_calls_saved_agents"],
            "api_calls_used_agents" : result_agents["api_calls_used_agents"],
            "runtime (s)": round(runtime, 2)
        }

        self.results.append(row)
        
    # "meta-llama/llama-4-scout-17b-16e-instruct"
    # "gpt-5-nano-2025-08-07"
        
    async def test_experiment(self, agent_type, samples, problems):
        
        model = "meta-llama/llama-4-scout-17b-16e-instruct"
        
        use_cachesaver = True
        
        # max_debate_round = 2
        
        for i in range(samples):
            if agent_type == "DebateAgent":
                await self.experiment(agent_type, problems, "math", model, use_cachesaver, i+1, 0, 0)
            elif agent_type == "SCAgent":
                await self.experiment(agent_type, problems, "math", model, use_cachesaver, 0, i+1, 0)
            elif agent_type == "ReflexionAgent":
                await self.experiment(agent_type, problems, "math", model, use_cachesaver, 0, 0, i+1)
        
        dataframe = pd.DataFrame(self.results)
        
        print("\nResults with CacheSaver:")
        print(dataframe)
        
        dataframe = dataframe.T
        
        dataframe.to_excel(f"orchestrator/results/aime24_results_groq.xlsx", index=True)
        
    async def run_experiment(self):
        await self.test_experiment("DebateAgent", 3, 1)
        await self.test_experiment("SCAgent", 3, 1)
        await self.test_experiment("ReflexionAgent", 3, 1)
        
if __name__ == '__main__':
    test = Test()

    async def main():
        await test.asyncSetUp()
        await test.run_experiment()

    asyncio.run(main())