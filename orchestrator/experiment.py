import unittest
import asyncio
import pandas as pd
import time

import orchestrator.remote_orchestrator as orchestrator
import orchestrator.run_mas as run
from orchestrator.utils import tokens_to_cost

class Test(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        asyncio.get_running_loop().set_debug(False)
        self.results = []
    
    async def experiment(self, problems, model, agent_model, use_cachesaver):
        start_time = time.time()
        metrics = await orchestrator.main(problems, model, use_cachesaver)
        result = await run.main(problems, agent_model, use_cachesaver)
        end_time = time.time()
        
        runtime = end_time - start_time
        
        input_cost_used_orc, output_cost_used_orc, total_cost_used_orc = tokens_to_cost(metrics["prompt_tokens_used_orc"], metrics["completion_tokens_used_orc"], agent_model)
        input_cost_used_agents, output_cost_used_agents, total_cost_used_agents = tokens_to_cost(result["prompt_tokens_used_agents"], result["completion_tokens_used_agents"], agent_model)
        
        row = {
            "problems": problems,
            "use_cachesaver": use_cachesaver,
            "accuracy": round(result["accuracy"], 2),
            
            "prompt_tokens_used_orc": metrics["prompt_tokens_used_orc"],
            "input_cost_orc ($)": input_cost_used_orc,
            "completion_tokens_used_orc": metrics["completion_tokens_used_orc"],
            "output_cost_orc ($)": output_cost_used_orc,
            "total_cost_orc ($)": total_cost_used_orc,
            
            "prompt_tokens_used_agents": result["prompt_tokens_used_agents"],
            "input_cost_agents ($)": input_cost_used_agents,
            "completion_tokens_used_agents": result["completion_tokens_used_agents"],
            "output_cost_agents ($)": output_cost_used_agents,
            "total_cost_agents ($)": total_cost_used_agents,
            
            "api_calls_orc" : metrics["api_calls_orc"],
            "api_calls_agents" : result["api_calls_agents"],
            "runtime (s)": round(runtime, 2)
        }

        self.results.append(row)
        
    async def test_experiment(self):
        
        use_cachesaver = False
        
        await self.experiment(1, "math", "meta-llama/llama-4-scout-17b-16e-instruct", use_cachesaver)
        
        dataframe = pd.DataFrame(self.results)
        
        print("\nResults with CacheSaver:")
        print(dataframe)
        
        dataframe = dataframe.T
        
        dataframe.to_excel("orchestrator/experiments/aime24_results.xlsx", index=True)
        
if __name__ == '__main__':
    unittest.main()