from mas_r1_reasoner.agents.agent_system_async import AsyncAgentSystem
from mas_r1_reasoner.agents.agent_system import Info
from mas_r1_reasoner.agents.shared_vars import set_global
from mas_r1_reasoner.agents.code_sanity import validate_python_code
from mas_r1_reasoner.agents.sampler.chat_completion_sampler import ChatCompletionSampler
from mas_r1_reasoner.rewards.utils.harmony_parser.minimal import extract_harmony_code_from_response
from datasets import load_dataset
import asyncio
import ray
import os
from orchestrator.samplers import GroqChatCompletionSampler

# Set up global variables required for MAS execution
set_global("global_max_ray_workers", 4)

set_global("global_node_model", "gpt-4o")

model_sampler_map = {
    "gpt-4o": ChatCompletionSampler(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0.5,
        mock_output=False
    )
}

set_global("global_model_sampler_map", model_sampler_map)

# Set other required global variables with defaults
set_global("global_max_round", 5)
set_global("global_max_sc", 5)
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
set_global("global_debate_role", ['Math Professor', 'Grade School Teacher'])

dataset = load_dataset(
    "DigitalLearningGmbH/MATH-lighteval",
    "algebra",
    split="train",
    trust_remote_code=True
)

problem = dataset[5]["problem"]
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

with open("mas_plan.xml") as f:
    xml_plan = f.read()

code, name, thought = extract_harmony_code_from_response(
    xml_plan,
    validate_python_code,
    logger=None
)

print(f"Extracted code: {code}")
print(f"Extracted thought: {thought}")
print(f"Extracted name: {name}")

if code.startswith("direct_answer"):
    print("No executable agent plan found:", thought)
else:
    print("Check if API key is present")
    assert os.getenv("GROQ_API_KEY") is not None, "Missing GROQ_API_KEY"
    ray.init()

    system = AsyncAgentSystem.create_with_globals()

    async def run():
        result = await system.execute_mas_batch_async(
            [code],
            [task_info]
        )
        print(result[0])

    asyncio.run(run())