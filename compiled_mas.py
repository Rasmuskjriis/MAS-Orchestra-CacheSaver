System: You are a MAS orchestrator.
Your job is to output a COMPLETE multi-agent system implementation in Python.

The system must:
- use AgentSystem / AsyncAgentSystem
- decompose the problem into agents
- define forward execution logic
- be reusable for other similar math problems

User problem:
Find the center of the circle with equation $x^2 - 6x + y^2 + 2y = 9$.

Return ONLY executable MAS code.
```python
from agent_system import AgentSystem, Agent

class CircleCenterFinder(Agent):