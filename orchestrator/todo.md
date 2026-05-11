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
* [] Add cachesaver with openai sampler
* [] Update token counting in cs samplers to reflect cached/deduplicated responses

### Trained orchestrator
* [X] Add cachesaver to the orchestrator

### Agents
* [X] Change agent naming from randomized to standardized

### Bugs
* [X] Investigate possible bug with only the first round's responses being taken from the cache (see agents)