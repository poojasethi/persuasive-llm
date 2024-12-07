import os

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from algorithms.mdp import MDP
from algorithms.mcts import MonteCarloTreeSearch

from typing import List, Tuple, Dict, Union

from utils.common import reward_model


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
    "end conversation",
]


def transition_model(state: str, action: str, next_state: str) -> float:
    action_to_next_state_probability = {}

    if state == "disagree":
        action_to_next_state_probability = {
            "present facts": {
                "disagree": 0.50,
                "slightly disagree": 0.30,
                "neutral": 0.15,
                "slightly agree": 0.04,
                "agree": 0.01,
            },
            "ask a question": {
                "disagree": 0.30,
                "slightly disagree": 0.50,
                "neutral": 0.15,
                "slightly agree": 0.04,
                "agree": 0.01,
            },
            "empathize": {
                "disagree": 0.20,
                "slightly disagree": 0.45,
                "neutral": 0.30,
                "slightly agree": 0.04,
                "agree": 0.01,
            },
            "confirm common ground": {
                "disagree": 0.20,
                "slightly disagree": 0.55,
                "neutral": 0.20,
                "slightly agree": 0.04,
                "agree": 0.01,
            },
            "share a personal story": {
                "disagree": 0.20,
                "slightly disagree": 0.55,
                "neutral": 0.20,
                "slightly agree": 0.04,
                "agree": 0.01,
            },
            "end conversation": {
                "disagree": 1.0,
                "slightly disagree": 0.0,
                "neutral": 0.0,
                "slightly agree": 0.04,
                "agree": 0.01,
            },
        }
    elif state == "slightly disagree":
        action_to_next_state_probability = {
            "present facts": {
                "disagree": 0.10,
                "slightly disagree": 0.65,
                "neutral": 0.20,
                "slightly agree": 0.04,
                "agree": 0.01,
            },
            "ask a question": {
                "disagree": 0.0,
                "slightly disagree": 0.50,
                "neutral": 0.45,
                "slightly agree": 0.04,
                "agree": 0.01,
            },
            "empathize": {
                "disagree": 0.05,
                "slightly disagree": 0.40,
                "neutral": 0.50,
                "slightly agree": 0.04,
                "agree": 0.01,
            },
            "confirm common ground": {
                "disagree": 0.05,
                "slightly disagree": 0.50,
                "neutral": 0.40,
                "slightly agree": 0.04,
                "agree": 0.01,
            },
            "share a personal story": {
                "disagree": 0.05,
                "slightly disagree": 0.40,
                "neutral": 0.50,
                "slightly agree": 0.04,
                "agree": 0.01,
            },
            "end conversation": {
                "disagree": 0.0,
                "slightly disagree": 1.0,
                "neutral": 0.0,
                "slightly agree": 0.0,
                "agree": 0.0,
            },
        }
    elif state == "neutral":
        action_to_next_state_probability = {
            "present facts": {
                "disagree": 0.05,
                "slightly disagree": 0.20,
                "neutral": 0.40,
                "slightly agree": 0.30,
                "agree": 0.05,
            },
            "ask a question": {
                "disagree": 0.0,
                "slightly disagree": 0.0,
                "neutral": 0.90,
                "slightly agree": 0.10,
                "agree": 0.0,
            },
            "empathize": {
                "disagree": 0.0,
                "slightly disagree": 0.0,
                "neutral": 0.90,
                "slightly agree": 0.10,
                "agree": 0.0,
            },
            "confirm common ground": {
                "disagree": 0.0,
                "slightly disagree": 0.0,
                "neutral": 0.90,
                "slightly agree": 0.10,
                "agree": 0.0,
            },
            "share a personal story": {
                "disagree": 0.01,
                "slightly disagree": 0.14,
                "neutral": 0.50,
                "slightly agree": 0.30,
                "agree": 0.05,
            },
            "end conversation": {
                "disagree": 0.0,
                "slightly disagree": 0.0,
                "neutral": 1.0,
                "slightly agree": 0.0,
                "agree": 0.0,
            },
        }
    elif state == "slightly agree":
        action_to_next_state_probability = {
            "present facts": {
                "disagree": 0.0,
                "slightly disagree": 0.01,
                "neutral": 0.04,
                "slightly agree": 0.45,
                "agree": 0.50,
            },
            "ask a question": {
                "disagree": 0.0,
                "slightly disagree": 0.0,
                "neutral": 0.05,
                "slightly agree": 0.90,
                "agree": 0.05,
            },
            "empathize": {
                "disagree": 0.0,
                "slightly disagree": 0.0,
                "neutral": 0.0,
                "slightly agree": 0.95,
                "agree": 0.05,
            },
            "confirm common ground": {
                "disagree": 0.0,
                "slightly disagree": 0.0,
                "neutral": 0.0,
                "slightly agree": 0.60,
                "agree": 0.40,
            },
            "share a personal story": {
                "disagree": 0.0,
                "slightly disagree": 0.0,
                "neutral": 0.10,
                "slightly agree": 0.60,
                "agree": 0.30,
            },
            "end conversation": {
                "disagree": 0.0,
                "slightly disagree": 0.0,
                "neutral": 0.0,
                "slightly agree": 1.0,
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
                "agree": 1.0,
            },
            "ask a question": {
                "disagree": 0.0,
                "slightly disagree": 0.0,
                "neutral": 0.0,
                "slightly agree": 0.0,
                "agree": 1.0,
            },
            "empathize": {
                "disagree": 0.0,
                "slightly disagree": 0.0,
                "neutral": 0.0,
                "slightly agree": 0.0,
                "agree": 1.0,
            },
            "confirm common ground": {
                "disagree": 0.0,
                "slightly disagree": 0.0,
                "neutral": 0.0,
                "slightly agree": 0.0,
                "agree": 1.0,
            },
            "share a personal story": {
                "disagree": 0.0,
                "slightly disagree": 0.0,
                "neutral": 0.0,
                "slightly agree": 0.0,
                "agree": 1.0,
            },
            "end conversation": {
                "disagree": 0.0,
                "slightly disagree": 0.0,
                "neutral": 0.0,
                "slightly agree": 0.0,
                "agree": 1.0,
            },
        }
    else:
        print(f"Unrecognized state: {state}")

    if not action_to_next_state_probability: 
        return ValueError()

    transition_probability = action_to_next_state_probability[action][next_state]

    return transition_probability


mdp = MDP(
    gamma=GAMMA,
    S=STATES,
    A=ACTIONS,
    T=transition_model,
    R=reward_model,
)

N = {}
Q = {}
d = 10
m = 10
c = 10


def utility_function(state: str):
    state_to_utility = {
        "disagree": -10,
        "slightly disagree": -5,
        "neutral": -1,
        "slightly agree": 5,
        "agree": 10,
    }
    return state_to_utility[state]


mcts = MonteCarloTreeSearch(
    P=mdp,
    N=N,
    Q=Q,
    d=d,
    m=m,
    c=c,
    U=utility_function,
)

user_state_prompt = """
Please predict how the user currently feels relative to you about the topic. Choose exactly one of: {states}

The format of your response should look like below.
User Current State: <predicted state>

Here is the prior conversation history:
{conversation_history}

And here is the user's latest input: 
{user_input}

Given this context, please reply in the format described above.
"""

prompt = """
At your turn, please respond with the following information:

1. You are given the user's current state on the topic. Here is their state: {state}
2. You are given an action to follow. Here is the action: {action}
3. Please provide a response to the user consistent with the given action. Answer in a conversational manner and try to mirror the style and tone of the user while being respectful. Don't make your response more than 3 sentences long.

The format of your response should look like below.
User Current State: <given state>
My Action: <given action>
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


def parse_user_state_agent_response(agent_response: str) -> str:
    outputs = agent_response.split("\n")
    assert len(outputs) == 1, "Expecting 1 partsto LLM output"
    state = outputs[0].removeprefix("User Current State:").strip()

    return state


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

    user_state_prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"You are a persuasive agent. Your goal is to convince the human to agree with you on this issue: {topic}.",
            ),
            ("human", user_state_prompt),
        ]
    )
    user_state_chain = user_state_prompt_template | llm

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
    total_rewards = 0

    while not conversation_ended and num_turns < MAX_CONVERSATION_LENGTH:
        user_input = input("User: ")
        conversation_history.append(user_input)
        conversation_history_str = format_conversation_history(conversation_history)

        # Predict the user's current state.
        user_state_agent_response = user_state_chain.invoke(
            {
                "topic": topic,
                "states": str(STATES),
                "conversation_history": conversation_history_str,
                "user_input": user_input,
            }
        )
        state = parse_user_state_agent_response(user_state_agent_response.content)

        # Use Monte Carlo Tree Search to select the next action given the current state.
        action = mcts(state)

        # Get the agent's response, given the state and selected optimal action.
        agent_response = chain.invoke(
            {
                "topic": topic,
                "state": state,
                "action": action,
                "conversation_history": conversation_history_str,
                "user_input": user_input,
            }
        )

        parsed_state, parsed_action, response = parse_agent_response(
            agent_response.content
        )

        assert state == parsed_state, "Provided state and parsed stated don't match"
        assert action == parsed_action, "Provided action and parsed action don't match"

        conversation_history.append((state, action, response))

        reward = reward_model(state, action)
        total_rewards += reward

        print("Agent:\n", agent_response.content)
        print()

        num_turns += 1
        if action == "end conversation":
            conversation_ended = True

    print("*" * 50)
    print(f"User final state: {state}")
    print(f"Total rewards: {total_rewards}")


def main():
    topic = "Everyone should adopt a plant-based diet for environmental reasons"
    start_persuasive_conversation(topic)


if __name__ == "__main__":
    main()


# TA feedback:
# Accumulate rewards for both the baseline and improved agent
# Qualitative feedback is good too
