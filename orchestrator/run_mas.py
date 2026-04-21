from mas_r1_reasoner.agents.agent_system_async import AsyncAgentSystem

with open("compiled_mas.py") as f:
    code = f.read()

agent_system = AsyncAgentSystem.create_with_globals()

result = agent_system.execute_mas(code, task_info)
print(result)