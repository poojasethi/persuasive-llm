import os

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from algorithms.mdp import MDP
from algorithms.mcts import MonteCarloTreeSearch

from typing import List, Tuple, Dict, Union


MAX_CONVERSATION_LENGTH = 10

# Initialize OpenAI API key.
api_key = os.environ.get("OPENAI_API_KEY")

# Replace 'YOUR_OPENAI_API_KEY' with your actual API key
llm = ChatOpenAI(api_key=api_key, temperature=0.7)

GAMMA = 0.9

STATES = ["disagree", "slightly disagree", "neutral", "slightly agree", "agree"]

ACTIONS = [
    "present facts",
    "ask a question",
    "empathize",
    "confirm common ground",
    "share a personal story",
    "end conversation"
]

def reward_model(state: str, action: str) -> int:
    action_to_reward = {}

    if state == "disagree":
        action_to_reward = {
            "present facts": 5,
            "ask a question": 10,
            "empathize": 15,
            "confirm common ground": 0,
            "share a personal story": 5,
            "end conversation": -10,
        }
    elif state == "slighty disagree":
        action_to_reward = {
            "present facts": 10,
            "ask a question": 5,
            "empathize": 10,
            "confirm common ground": 0,
            "share a personal story": 10,
            "end conversation": -10,
        }
    elif state == "neutral":
        action_to_reward = {
            "present facts": 3,
            "ask a question": 1,
            "empathize": 5,
            "confirm common ground": 2,
            "share a personal story": 3,
            "end conversation": -1,
        }
    elif state == "slightly agree":
        action_to_reward = {
            "present facts": 4,
            "ask a question": 2,
            "empathize": 2,
            "confirm common ground": 6,
            "share a personal story": 3,
            "end conversation": 1,
        }
    elif state == "agree":
        action_to_reward = {
            "present facts": 0,
            "ask a question": 0,
            "empathize": 0,
            "confirm common ground": 5,
            "share a personal story": 0,
            "end conversation": 10,
        }
    else:
        print(f"Unrecognized state: {state}")
    
    return action_to_reward[action]

def transition_model(state: str, action: str, next_state: str) -> float:
    action_to_next_state_probability = {}

    if state == "disagree":
        action_to_next_state_probability = {
            "present facts": {
                "disagree": 0.25,
                "slightly disagree": 0.25,
                "neutral": 0.05,
                "slightly agree": 0.04,
                "agree": 0.01,
            },
            "ask a question": {
                "disagree": 0.20,
                "slightly disagree": 0.30,
                "neutral": 0.05,
                "slightly agree": 0.04,
                "agree": 0.01,
            },
            "empathize": {
                "disagree": 0.0,
                "slightly disagree": 0.0,
                "neutral": 0.0,
                "slightly agree": 0.0,
                "agree": 0.0,
            },
            "confirm common ground":{
                "disagree": 0.0,
                "slightly disagree": 0.0,
                "neutral": 0.0,
                "slightly agree": 0.0,
                "agree": 0.0,
            },
            "share a personal story": {
                "disagree": 0.0,
                "slightly disagree": 0.0,
                "neutral": 0.0,
                "slightly agree": 0.0,
                "agree": 0.0,
            },
            "end conversation":{
                "disagree": 1.0,
                "slightly disagree": 0.0,
                "neutral": 0.0,
                "slightly agree": 0.0,
                "agree": 0.0,
            },
        }
    elif state == "slighty disagree":
        action_to_next_state_probability = {
            "present facts": {
                "disagree": 0.0,
                "slightly disagree": 0.0,
                "neutral": 0.0,
                "slightly agree": 0.0,
                "agree": 0.0,
            },
            "ask a question": {
                "disagree": 0.0,
                "slightly disagree": 0.0,
                "neutral": 0.0,
                "slightly agree": 0.0,
                "agree": 0.0,
            },
            "empathize": {
                "disagree": 0.0,
                "slightly disagree": 0.0,
                "neutral": 0.0,
                "slightly agree": 0.0,
                "agree": 0.0,
            },
            "confirm common ground":{
                "disagree": 0.0,
                "slightly disagree": 0.0,
                "neutral": 0.0,
                "slightly agree": 0.0,
                "agree": 0.0,
            },
            "share a personal story": {
                "disagree": 0.0,
                "slightly disagree": 0.0,
                "neutral": 0.0,
                "slightly agree": 0.0,
                "agree": 0.0,
            },
            "end conversation":{
                "disagree": 0.0,
                "slightly disagree": 0.0,
                "neutral": 0.0,
                "slightly agree": 0.0,
                "agree": 0.0,
            },
        }
    elif state == "neutral":
         action_to_next_state_probability = {
            "present facts": {
                "disagree": 0.0,
                "slightly disagree": 0.0,
                "neutral": 0.0,
                "slightly agree": 0.0,
                "agree": 0.0,
            },
            "ask a question": {
                "disagree": 0.0,
                "slightly disagree": 0.0,
                "neutral": 0.0,
                "slightly agree": 0.0,
                "agree": 0.0,
            },
            "empathize": {
                "disagree": 0.0,
                "slightly disagree": 0.0,
                "neutral": 0.0,
                "slightly agree": 0.0,
                "agree": 0.0,
            },
            "confirm common ground":{
                "disagree": 0.0,
                "slightly disagree": 0.0,
                "neutral": 0.0,
                "slightly agree": 0.0,
                "agree": 0.0,
            },
            "share a personal story": {
                "disagree": 0.0,
                "slightly disagree": 0.0,
                "neutral": 0.0,
                "slightly agree": 0.0,
                "agree": 0.0,
            },
            "end conversation":{
                "disagree": 0.0,
                "slightly disagree": 0.0,
                "neutral": 0.0,
                "slightly agree": 0.0,
                "agree": 0.0,
            },
        }
    elif state == "slightly agree":
         action_to_next_state_probability = {
            "present facts": {
                "disagree": 0.0,
                "slightly disagree": 0.0,
                "neutral": 0.0,
                "slightly agree": 0.0,
                "agree": 0.0,
            },
            "ask a question": {
                "disagree": 0.0,
                "slightly disagree": 0.0,
                "neutral": 0.0,
                "slightly agree": 0.0,
                "agree": 0.0,
            },
            "empathize": {
                "disagree": 0.0,
                "slightly disagree": 0.0,
                "neutral": 0.0,
                "slightly agree": 0.0,
                "agree": 0.0,
            },
            "confirm common ground":{
                "disagree": 0.0,
                "slightly disagree": 0.0,
                "neutral": 0.0,
                "slightly agree": 0.0,
                "agree": 0.0,
            },
            "share a personal story": {
                "disagree": 0.0,
                "slightly disagree": 0.0,
                "neutral": 0.0,
                "slightly agree": 0.0,
                "agree": 0.0,
            },
            "end conversation":{
                "disagree": 0.0,
                "slightly disagree": 0.0,
                "neutral": 0.0,
                "slightly agree": 0.0,
                "agree": 0.0,
            },
        }
    elif state == "agree":
         action_to_next_state_probability = {
            "present facts": {
                "disagree": 0.0,
                "slightly disagree": 0.0,
                "neutral": 0.0,
                "slightly agree": 0.0,
                "agree": 0.0,
            },
            "ask a question": {
                "disagree": 0.0,
                "slightly disagree": 0.0,
                "neutral": 0.0,
                "slightly agree": 0.0,
                "agree": 0.0,
            },
            "empathize": {
                "disagree": 0.0,
                "slightly disagree": 0.0,
                "neutral": 0.0,
                "slightly agree": 0.0,
                "agree": 0.0,
            },
            "confirm common ground":{
                "disagree": 0.0,
                "slightly disagree": 0.0,
                "neutral": 0.0,
                "slightly agree": 0.0,
                "agree": 0.0,
            },
            "share a personal story": {
                "disagree": 0.0,
                "slightly disagree": 0.0,
                "neutral": 0.0,
                "slightly agree": 0.0,
                "agree": 0.0,
            },
            "end conversation":{
                "disagree": 0.0,
                "slightly disagree": 0.0,
                "neutral": 0.0,
                "slightly agree": 0.0,
                "agree": 0.0,
            },
        }
    else:
        print(f"Unrecognized state: {state}") 

    transition_probability = action_to_next_state_probability[action][next_state]
    
    return transition_probability

mdp = MDP(
    gamma=GAMMA,
    S=STATES,
    A=ACTIONS,
    T=transition_model,
    R=reward_model,
)

breakpoint()

prompt = """
At your turn, please respond with the following information:

1. Please predict how the user currently feels about the topic. Choose exactly one of: {states}
2. Please choose precisely one of the following actions to take: {actions}
3. Finally, please provide a response to the user consistent with your chosen action. Answer in a conversational manner and try to mirror the style and tone of the user while being respectful. Don't make your response more than 3 sentences long.

The format of your response should look like below.
User Current State: <predicted state>
My Action: <your chosen action>
My Response: <your response>

Here is the prior conversation history:
{conversation_history}

And here is the user's latest input: 
{user_input}

Given this context, please reply in the format described above.
"""

def format_conversation_history(
    history: List[List[Union[str, Tuple[str, str, str]]]]
) -> str:
    output = []
    assert len(history) % 2 == 0

    for i, row in enumerate(history):
        speaker = ""
        response = row
        if i % 2 == 0:  # Agent's turn
            speaker = "Agent"
            if type(row) != str:
                state, action, response = row
        else:  # User's turn
            speaker = "User"

        output.append(f"{speaker}: {response}")

    return "\n".join(output)


def parse_agent_response(agent_response: str) -> Tuple[str, str, str]:
    outputs = agent_response.split("\n")
    assert len(outputs) == 3, "Expecting 3 parts to LLM output"
    state = outputs[0].removeprefix("User Current State:").strip()
    action = outputs[1].removeprefix("My Action:").strip()
    response = outputs[2].removeprefix("My Response:").strip()

    return (state, action, response)


def start_persuasive_conversation(topic: str):
    opener = f'What are your thoughts on this topic: "{topic}"?'
    print(f"Agent: {opener}")
    conversation_history = []
    conversation_history.append(opener)

    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"You are a persuasive agent. Your goal is to convince the human to agree with you on this issue: {topic}.",
            ),
            ("human", prompt),
        ]
    )
    chain = prompt_template | llm

    num_turns = 0
    conversation_ended = False
    while not conversation_ended and num_turns < MAX_CONVERSATION_LENGTH:
        user_input = input("User: ")
        conversation_history.append(user_input)
        conversation_history_str = format_conversation_history(conversation_history)

        agent_response = chain.invoke(
            {
                "topic": topic,
                "states": str(STATES),
                "actions": str(ACTIONS),
                "conversation_history": conversation_history_str,
                "user_input": user_input,
            }
        )

        # TODO: Use Monte Carlo Tree Search to pick the next action:
        # https://github.com/griffinbholt/decisionmaking-code-py/blob/main/src/ch09.py
        state, action, response = parse_agent_response(agent_response.content)
        conversation_history.append((state, action, response))

        print("Agent:\n", agent_response.content)
        print()

        num_turns += 1
        if action == "end conversation":
            conversation_ended = True

def main():
    topic = "Everyone should adopt a plant-based diet for environmental reasons"
    start_persuasive_conversation(topic)


if __name__ == "__main__":
    main()


# feedback: 
# Accumulate rewards for both the baseline and improved agent
# Qualitative feedback is good too
# 