import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from datasets import load_dataset
from openai import OpenAI

BASE_URL = "https://discern-stroller-recycling.ngrok-free.dev/v1"
MODEL = "math"

OUTPUT_XML = "mas_plan.xml"

client = OpenAI(
    base_url=BASE_URL,
    api_key="dummy"  # required but ignored by server
)

# =========================
# 1. Load dataset problem
# =========================
print("Fetching dataset problem...")
dataset = load_dataset(
    "DigitalLearningGmbH/MATH-lighteval",
    "algebra",
    split="train",
    trust_remote_code=True
)

problem = dataset[5]["problem"]

# print("problem: ", problem)
# print()

# =========================
# 3. MAS PROMPT (IMPORTANT)
# =========================

MATH_SYSTEM_PROMPT = """You are a helpful assistant.

MASness (How much Multi-Agent System-ness): Minimal

Valid Channels: thinking, agent, answer

Model: harmony

An agent is a pre-configured AI personalities that can delegate tasks to. Each subagent:
1. Has a specific purpose and expertise area
2. Uses its own context window separate from the main conversation
3. (Optional) Can be configured with specific tools it's allowed to use
4. Includes a custom system prompt that guides its behavior

An agent can only call within the <thinking> channel, it should be defined in tag <agent>. Each agent must contain <agent_name>, <agent_description>, <required_arguments>, <agent_output_id>


DO NOT MISS ANY REQUEST FIELDS and ensure that your response is a well-formed XML object!"""


MATH_USER_PROMPT_TEMPLATE = """Please solve the question step by step and create agents to delegate the task when necessary. First decide whether to solve directly or to delegate to exactly one agent. If you delegate, you must define that agent by outputting <agent> with agent_name (select one of the agents: CoTAgent, SCAgent, DebateAgent, ReflexionAgent), agent_description, required_arguments, and agent_output_id.
The final output of the delegated agent (identified by its agent_output_id) must represent the complete and final answer to the original question, not an intermediate result.
Always put either the final value or the agent_output_id in the <answer> tag, and use EXACTLY the same field names defined for the agent. If the selected agent uses roles (e.g., DebateAgent), also output debate_roles.

For example,

If you decide to solve the entire task yourself, you will output the following:

Question: What is (20+9)*(30+7)?

 <thinking>
   The problem only requires basic arithmetic.
    No specialized reasoning agent or multi-agent discussion is needed.
    (20+9)*(30+7) = 600 + 140 + 270 + 63 = 1073.
  </thinking>
  <answer>1073</answer>

If you decide to solve the entire task via delegation, you will output the following. In this case, since the agent_input is the same as the original task, you must set the agent_input as empty (\"\"), and the parser will replace it with the original question.

Question: Compute the definite integral of (2x + 5) dx from 0 to 3.

  <thinking>
    This problem involves symbolic integration and applying the Fundamental Theorem of Calculus.
    It requires structured reasoning rather than simple numeric computation.
    I will use a calculus agent that can perform step-by-step Chain-of-Thought reasoning.
    The final answer to the original question will be the output of the CoTAgent.
  </thinking>
  <agent>
    <agent_name>CoTAgent</agent_name>
    <agent_description>Definite integrals with one Chain-of-Thought call.</agent_description>
      <required_arguments>
        <agent_input></agent_input>
      </required_arguments>
      <agent_output_id>
      calc_agent_output
      </agent_output_id>
  </agent>
  <answer>calc_agent_output</answer>

Another example:

Question: What is 17 cubed?

  <thinking>
    Computing 17\u00b3 can be done directly, but arithmetic mistakes are easy to make.
    Using a Chain-of-Thought with Self-Consistency (CoT_SC) allows sampling multiple reasoning paths and combining results for accuracy.
    The final answer to the original question will be the output of the SCAgent.

  </thinking>
  <agent>
    <agent_name>SCAgent</agent_name>
    <agent_description>Performs arithmetic calculations using Chain-of-Thought with Self-Consistency (CoT_SC).</agent_description>
    <required_arguments>
      <agent_input></agent_input>
    </required_arguments>
      <agent_output_id>math_agent_output</agent_output_id>
  </agent>
  <answer>math_agent_output</answer>

More examples:

Question: Given x\u00b2 = 46.694444, which target number is closer, 45 or 46?

  <thinking>
    This question requires comparison between two close numeric choices.
    To ensure accuracy, I will let two reasoning roles debate \u2014 one focusing on mathematical precision and the other on practical rounding.
    The DebateAgent can capture both perspectives and reach a justified final answer.
    The final answer will be the output of the DebateAgent.
  </thinking>
  <agent>
    <agent_name>DebateAgent</agent_name>
    <agent_description>Near-tie numeric choice using one Debate call.</agent_description>
      <required_arguments>
        <agent_input></agent_input>
        <debate_roles>[\"Mathematics Professor\", \"Statistics Teacher\"]</debate_roles>
      </required_arguments>
      <agent_output_id>compare_agent_output</agent_output_id>
  </agent>
  <answer>compare_agent_output</answer>


Final example:

Question: A train travels 60 miles per hour. How far does it go in 2.5 hours?

  <thinking>
    This task requires reasoning with a formula and ensuring units are handled correctly.
    I will use a Reflexion agent that can reflect on and refine its reasoning if errors occur.
    The final answer to the original question will be the output of the ReflexionAgent.

  </thinking>
  <agent>
    <agent_name>ReflexionAgent</agent_name>
    <agent_description>Solves reasoning tasks with iterative self-reflection and critique using Reflexion.</agent_description>
      <required_arguments>
        <agent_input></agent_input>
      </required_arguments>
      <agent_output_id>reflexion_agent_output</agent_output_id>
  </agent>
  <answer>reflexion_agent_output</answer>

If you decide to decompose the question into smaller reasoning steps before delegation, you will output the following. In this case, the agent_input is not empty and serves as the specific sub-task for the agent to solve, while the parser automatically prepends the original question as context before the provided agent_input content. Delegation must occur only after you have completed all reasoning steps \u2014 that is, it should appear in the final step of your reasoning process.


Question: Compute the square root of the average of the first five prime numbers (2, 3, 5, 7, 11), rounded to three decimals.

  <thinking>
    Step 1: Compute the average manually:
      sum = 2 + 3 + 5 + 7 + 11 = 28
      avg = 28 / 5 = 5.6

    Step 2:
      The remaining step \u2014 computing sqrt(5.6) to three decimals \u2014 requires precision and numeric refinement.
      I will delegate that part to a ReflexionAgent using the ReflexionAgent for self-correction if rounding is wrong.
      The final answer to the original question will be the output of the ReflexionAgent.

  </thinking>
  <agent>
    <agent_name>ReflexionAgent</agent_name>
    <agent_description>Square root with a light self-refine loop (single agent call).</agent_description>
      <required_arguments>
          <agent_input>Compute sqrt(5.6) to 3 decimals</agent_input>
      </required_arguments>
      <agent_output_id>numeric_agent_output</agent_output_id>
  </agent>
  <answer>numeric_agent_output</answer>


Below is the question to solve:

{question}"""


MATH_USER_SUFFIX = """Channels:
- <thinking>: internal reasoning and planning
- <agent>: definition of agents
- <answer>: final user-facing answer

Model (the Large lanaguge model used in agent:
- gpt-4.1-nano: Fastest, most cost-efficient version of GPT-4.1. GPT-4.1 nano excels at instruction following and tool calling. It features a 1M token context window, and low latency without a reasoning step. 1,047,576 context window. 32,768 max output tokens. Jun 01, 2024 knowledge cutoff
- gpt-oss-120b: state-of-the-art language models that deliver strong real-world performance at low cost. outperform similarly sized open models on reasoning tasks, demonstrate strong tool use capabilities, and are optimized for efficient deployment on consumer hardware. It was trained using a mix of reinforcement learning and techniques informed by OpenAI's most advanced internal models, including o3 and other frontier systems. It achieves near-parity with OpenAI o4-mini on core reasoning benchmarks. It also perform strongly on tool use, few-shot function calling, CoT reasoning (as seen in results on the Tau-Bench agentic evaluation suite) and HealthBench (even outperforming proprietary models like OpenAI o1 and GPT-4o). 131,072 context window, 131,072 max output tokens, Jun 01, 2024 knowledge cutoff, Reasoning token support

MASness Levels:
- minimal: direct solve or at most one agent
- medium:  one or more agents delegation
- high: complex multi-agent delegation


Sub-agent Schema (all fields required):
<agent>
    <agent_name>...</agent_name> (select one of the agents: CoTAgent, SCAgent, DebateAgent, ReflexionAgent)
    <agent_description>...</agent_description>
    <required_arguments> (make sure all required parameters are set. Must follow XML format)
        <...>...</...>
        <...>...</...>
    </required_arguments>
    <agent_output_id>
    ...
    </agent_output_id>
</agent>

Available Agents:

CoTAgent: {\"desciption\": \"By encouraging the LLM to think step by step rather than directly outputting an answer, chain-of-thought reasoning enables complex problem-solving through intermediate steps. This practice improves the model's ability to handle tasks that require deeper reasoning and provides insight into its decision-making process.\", \"name\": \"Chain-of-Thought Agent (CoTAgent)\", \"required_arguments\": {\"agent_input\": \"The input for the CoTAgent. This is the task question for the CoT LLM to solve. If left empty (\\\"\\\") the parser will automatically replace it with the original question.\"}, \"implementation\": \"def CoTAgent(self, agent_input, model: str):\\n    from mas_r1_reasoner.agents.agent_system import LLMAgentBase, Info\\n    \\n    # Validate that agent_input is an Info object\\n    assert isinstance(agent_input, Info), f\\\"agent_input must be an Info object, got {agent_input}\\\"\\n\\n    \\n    # Basic setting\\n    temperature = 0.5\\n    \\n    # Instruction for the Chain-of-Thought (CoT) approach\\n    # It is an important practice that allows the LLM to think step by step before solving the task.\\n    cot_instruction = \\\"Please think step by step and then solve the task.\\\"\\n\\n    # Instantiate a new LLM specifically for CoT\\n    # To allow LLM thinking before answering, we need to set an additional output field 'thinking'.\\n    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought LLM', model=model, temperature=temperature)\\n\\n    # Get the response from the CoT tool\\n    thinking, answer = cot_agent([agent_input], cot_instruction)\\n    final_answer = self.make_final_answer(thinking, answer)\\n    \\n    # Return only the final answer\\n    return final_answer   \\n\"}

SCAgent: {\"desciption\": \"While an LLM can arrive at the correct answer, its reasoning may vary. By repeatedly asking the same question with high temperature settings, we can generate different reasoning paths. We then combine multiple answers from these Chain-of-Thought (CoTAgent) agents to produce a more accurate final answer through ensembling. Best for problems where you want high confidence through consensus.\", \"name\": \"Self-Consistency with Chain-of-Thought (SCAgent)\", \"required_arguments\": {\"agent_input\": \"The input for the SCAgent. This is the task question for the CoT_SC LLM to solve. If left empty (\\\"\\\"), the parser will automatically replace it with the original question.\"}, \"implementation\": \"def SCAgent(self, agent_input, model: str):\\n    from mas_r1_reasoner.agents.agent_system import LLMAgentBase, Info\\n    \\n    # Validate that agent_input is an Info object\\n    assert isinstance(agent_input, Info), f\\\"agent_input must be an Info object, got {agent_input}\\\"\\n\\n    # Basic setting\\n    temperature = 0.5\\n    num_repeated_samples = 5\\n\\n    # Instruction for step-by-step reasoning\\n    cot_instruction =  \\\"Please think step by step and then solve the task.\\\"\\n\\n    # Instruction for step-by-step reasoning\\n    # Initialize multiple CoT agents with a higher temperature for varied reasoning\\n    cot_agents = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought LLM', model=model, temperature=temperature) for _ in range(num_repeated_samples)]\\n    \\n    thinking_mapping = {}\\n    answer_mapping = {}\\n    possible_answers = []\\n    for i in range(num_repeated_samples):\\n        thinking, answer = cot_agents[i]([agent_input], cot_instruction)\\n        possible_answers.append(answer.content)\\n        thinking_mapping[answer.content] = thinking\\n        answer_mapping[answer.content] = answer\\n\\n    # Ensembling the answers from multiple CoT agents\\n    answer = self.majority_voting(possible_answers)\\n\\n    thinking = thinking_mapping[answer]\\n    answer = answer_mapping[answer]\\n\\n    final_answer = self.make_final_answer(thinking, answer)\\n\\n    return final_answer  \\n\"}

DebateAgent: {\"desciption\": \"By letting different LLMs debate with each other, we can leverage their diverse perspectives to find better solutions for tasks. Best for problems that benefit from multiple perspectives.\", \"name\": \"LLM Debate (DebateAgent)\", \"required_arguments\": {\"agent_input\": \"The input for the DebateAgent. This is the task question or context for the Debate LLM to solve. If left empty (\\\"\\\") the parser will automatically replace it with the original question.\", \"debate_roles\": \"A list of roles (musst be more than one) for the DebateAgent (e.g., 'Mathematics Professor', 'Statistician'). Each role represents a distinct perspective and viewpoint.\"}, \"implementation\": \"def DebateAgent(self, agent_input, model: str, debate_roles: List[str]):\\n    from mas_r1_reasoner.agents.agent_system import LLMAgentBase, Info\\n    \\n    # Validate that agent_input is an Info object\\n    assert isinstance(agent_input, Info), f\\\"agent_input must be an Info object, got {agent_input}\\\"\\n\\n    # Basic setting\\n    temperature = 0.5\\n    max_debate_round = 5\\n\\n    # Instruction for initial reasoning\\n    debate_initial_instruction =  \\\"Please think step by step and then solve the task.\\\"\\n\\n    # Instruction for debating and updating the solution based on other agents' solutions\\n    debate_instruction = \\\"Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer. Put your thinking process in the 'thinking' field and the updated answer in the 'answer' field. \\\"\\n\\n    # Initialize debate agents with different roles and a moderate temperature for varied reasoning\\n    debate_agents = [LLMAgentBase(['thinking', 'answer'], 'Debate LLM', model=model, role=role, temperature=temperature) for role in debate_roles]\\n\\n    # Instruction for final decision-making based on all debates and solutions\\n    final_decision_instruction = \\\"Given all the above thinking and answers, reason over them carefully and provide a final answer. Put your thinking process in the 'thinking' field and the final answer in the 'answer' field.\\\"\\n\\n\\n    final_decision_agent = LLMAgentBase(['thinking', 'answer'], 'Final Decision LLM', model=model, temperature=temperature)\\n\\n    all_thinking = [[] for _ in range(max_debate_round)]\\n    all_answer = [[] for _ in range(max_debate_round)]\\n\\n    # Perform debate rounds\\n    for r in range(max_debate_round):\\n        for i in range(len(debate_agents)):\\n            if r == 0:\\n                thinking, answer = debate_agents[i]([agent_input], debate_initial_instruction)\\n            else:\\n                input_infos = [agent_input] + [all_thinking[r-1][i]] + all_thinking[r-1][:i] + all_thinking[r-1][i+1:]\\n                thinking, answer = debate_agents[i](input_infos, debate_instruction)\\n            all_thinking[r].append(thinking)\\n            all_answer[r].append(answer)\\n    \\n    # Make the final decision based on all debate results and solutions\\n    thinking, answer = final_decision_agent([agent_input] + all_thinking[max_debate_round-1] + all_answer[max_debate_round-1], final_decision_instruction)\\n    final_answer = self.make_final_answer(thinking, answer)\\n\\n    return final_answer\\n\"}

ReflexionAgent: {\"desciption\": \"To enhance its performance, an LLM can iteratively improve its answer based on feedback. By reflecting on its previous attempts and incorporating feedback, the model can refine its reasoning and provide a more accurate solution. Best for complex problems that benefit from self-correction.\", \"name\": \"Self-Refine (Reflexion)\", \"required_arguments\": {\"agent_input\": \"The input for the ReflexionAgent. This is the task question for the CoT LLM to solve. If left empty (\\\"\\\") the parser will automatically replace it with the original question.\"}, \"implementation\": \"def ReflexionAgent(self, agent_input, model: str):\\n    from mas_r1_reasoner.agents.agent_system import LLMAgentBase, Info\\n    \\n    # Validate that agent_input is an Info object\\n    assert isinstance(agent_input, Info), f\\\"agent_input must be an Info object, got {agent_input}\\\"\\n\\n    # Basic setting\\n    temperature = 0.5\\n    max_reflection_round = 5\\n\\n    # Instruction for initial reasoning\\n    initial_instruction = \\\"Please think step by step and then solve the task.\\\"\\n\\n    # Instruction for reflecting on previous attempts and feedback to improve\\n    reflect_instruction = \\\"Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better.\\\"\\n    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought LLM', model=model, temperature=temperature)\\n\\n    # Instruction for providing feedback and correcting the answer\\n    critic_instruction = \\\"Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'.\\\"\\n\\n    critic_agent = LLMAgentBase(['feedback', 'correct'], 'Critic LLM', model=model, temperature=temperature)\\n        \\n    # Initial attempt\\n    cot_inputs = [agent_input]\\n    thinking, answer = cot_agent(cot_inputs, initial_instruction, 0)\\n\\n    for i in range(max_reflection_round):\\n        # Get feedback and correct status from the critic\\n        feedback, correct = critic_agent([agent_input, thinking, answer], critic_instruction, i)\\n        if correct.content == 'True':\\n            break\\n            \\n        # Add feedback to the inputs for the next iteration\\n        cot_inputs.extend([thinking, answer, feedback])\\n\\n        # Reflect on previous attempts and refine the answer\\n        thinking, answer = cot_agent(cot_inputs, reflect_instruction, i + 1)\\n\\n    final_answer = self.make_final_answer(thinking, answer)\\n\\n    return final_answer\\n\"}"""


def build_math_messages(question):
    return [
        {"role": "system", "content": MATH_SYSTEM_PROMPT},
        {"role": "user", "content": MATH_USER_PROMPT_TEMPLATE.replace("{question}", question)},
        {"role": "user", "content": MATH_USER_SUFFIX},
    ]

messages = build_math_messages(problem)

# print("messages: ", messages)

# =========================
# 4. Generate XML
# =========================
print("\n==============================")
print("🧠 GENERATING MAS XML PLAN")
print("==============================\n")

start = time.time()

response = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    temperature=0.7,
    max_tokens=4096,
)

end = time.time()

print(f"Model: {response.model}")
print(f"Tokens: {response.usage.prompt_tokens} prompt, {response.usage.completion_tokens} completion")
print(f"\n--- Response ---\n")
print(response.choices[0].message.content)

output = response.choices[0].message.content

end_tag = "</answer>"
end_idx = output.rfind(end_tag)

if end_idx != -1:
    xml_content = output[:end_idx + len(end_tag)]
else:
    # fallback if model is broken
    xml_content = output

with open(OUTPUT_XML, "w") as f:
    f.write(xml_content)

print("\n==============================")
print("✅ XML PLAN SAVED")
print(f"📄 File: {OUTPUT_XML}")
print(f"⏱️ Time: {end - start:.2f}s")
print("==============================")