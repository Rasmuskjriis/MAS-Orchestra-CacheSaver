MAS Orchestra
===============================
### Verification
* [X] Add verification process to check if the returned answer is actually correct
* [X] Add average correctness of answers in run_mas.py

### Program
* [X] Refactor run_mas.py to run mutiple problems in a for loop

### Samplers
* [X] Add groq sampler
* [X] Add cachesaver with groq sampler
* [X] Add dummy metadata to groq/openai sampler
* [X] Add cachesaver with openai sampler
* [] Update token counting in cs samplers to reflect cached/deduplicated responses

### Trained orchestrator
* [X] Add cachesaver to the orchestrator

### Agents
* [X] Change agent naming from randomized to standardized

### Bugs
* [X] Investigate possible bug with only the first round's responses being taken from the cache (see agents)

### Tokens
* [] Update remote_orchestrator.py to return saved tokens as well
* [] Update experiment.py to handle saved tokens



# Notes:

## Answers
- Sometimes cannot detect identify answers as being correct even if they are like with the SCAgent which answers: "minutes.\n\n204'" and the actual answer being "204". Mathscorer explicitly looks bor boxed answers and therefore misses these kinds of answers.

## Parameter tuning:
### CoTAgent
- [Non-applicable] CoTAgent does not really have any parameters

### SCAgent
- [Great] SCAgent works well as it is a breadth type of agent with multiple identical calls
- When we increase the number of repeated samples (agents) by 1 using CacheSaver, we increase the number of api calls by 1

### DebateAgent
- [Great] DebateAgent works similar to LLMDebate with configurable amount of debate rounds. As for the amound of agents I am still unsure how to configure this
- When we increase the number of debate rounds by 1 using CacheSaver, we increase the number of api calls by 3, since there is one extra call per agent plus the evaluator meaning 2 + 1 = 3

### ReflexionAgent
- [Ok] ReflexionAgent can increase the number of rounds for reflexion, however it will early exit if it believes that the answer is correct. Therefore even if a max-round number is set, the agent is not guaranteed to reach it. A solution could potentially be to enforce this
- Works by initially having a CoTAgent answer the question. Then in the first round that answer is critiqued by another critic_agent, after which the CoTAgent gets it's own previous response and the critique from the critic_agent and tries to improve upon it's answer. Subsequent rounds follow this pattern of critique and then updated response from the critic_agent and CoTAgent respectively from all previous rounds.
- This means that when we increase the number of reflexion rounds by 1 using CacheSaver, we increase the number of api calls by 2, since there is one extra call for the critic_agent and one for the CoTAgent
- It seems quite unlikely that accuracy monotonically increases with this setup, since deep into reflexion rounds the agents will only see answers and correctness like "The answer is 24" or "This seems correct" instead of reasoning, however this is yet to be tested.
- It seems as if the agent gets cut off when trying to provide reasoning in later rounds like in the example below


Example of the critic_agent being confused by previous answers/responses:

(RayAgentWorker pid=14779) ### thinking #2 by Chain-of-Thought LLM Agent 1:
(RayAgentWorker pid=14779) I will re-evaluate the problem to ensure accuracy and completeness in my solution.
(RayAgentWorker pid=14779) 
(RayAgentWorker pid=14779) ### answer #2 by Chain-of-Thought LLM Agent 1:
(RayAgentWorker pid=14779) 204
(RayAgentWorker pid=14779) 
(RayAgentWorker pid=14779) Please review the answer above and criticize on where it might be wrong.
(RayAgentWorker pid=14779) ====================
(RayAgentWorker pid=14779) 
(RayAgentWorker pid=14779) Parsed response: {'feedback': 'The answer provided seems to be a numerical value without any explanation or justification. To assess its correctness, I need to re-solve the problem and verify if the answer matches.', 'correct': 'unknown'}