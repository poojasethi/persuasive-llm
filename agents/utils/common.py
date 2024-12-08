from algorithms.mdp import MDP
import os
from langchain_openai import ChatOpenAI
from typing import List, Union, Tuple

TOPICS = {
    "ethics": "Everyone should adopt a plant-based diet for environmental and health reasons.",
    "culture" :"Taylor Swift is the best female artist of all-time.",
    "politics": "The United States should offer universal health care (UHC) to improve the lives of its citizens.",
    "education": "You should take CS 238: Decision Making Under Uncertainty with Professor Mykel Kochenderfer.",
    "technology": "Artificial General Intelligence (AGI) will arrive within the next 10 years."
}
MAX_CONVERSATION_LENGTH = 10

# Initialize OpenAI API key and LLM.
API_KEY = os.environ.get("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o-mini", api_key=API_KEY, temperature=0.3)

### MDP Definition ####
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
    elif state == "slightly disagree":
        action_to_reward = {
            "present facts": 5,
            "ask a question": 7,
            "empathize": 10,
            "confirm common ground": 0,
            "share a personal story": 10,
            "end conversation": -5,
        }
    elif state == "neutral":
        action_to_reward = {
            "present facts": 4,
            "ask a question": 3,
            "empathize": 0,
            "confirm common ground": 2,
            "share a personal story": 3,
            "end conversation": -1,
        }
    elif state == "slightly agree":
        action_to_reward = {
            "present facts": 3,
            "ask a question": 2,
            "empathize": 2,
            "confirm common ground": 6,
            "share a personal story": 4,
            "end conversation": 5,
        }
    elif state == "agree":
        action_to_reward = {
            "present facts": 0,
            "ask a question": 0,
            "empathize": 0,
            "confirm common ground": 7,
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
                "slightly agree": 0.00,
                "agree": 0.00,
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
                "disagree": 0.01,
                "slightly disagree": 0.14,
                "neutral": 0.50,
                "slightly agree": 0.30,
                "agree": 0.05,
            },
            "ask a question": {
                "disagree": 0.0,
                "slightly disagree": 0.0,
                "neutral": 0.75,
                "slightly agree": 0.25,
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

# Utility at leaves for oline planing methods.
def utility_function(state: str):
    state_to_utility = {
        "disagree": -10,
        "slightly disagree": -5,
        "neutral": -1,
        "slightly agree": 1,
        "agree": 10,
    }
    return state_to_utility[state]

### Conversation Parsing ###
def format_conversation_history(
    history: List[List[Union[str, Tuple[str, str, str]]]], show_states_and_actions: bool = False,
) -> str:
    output = []
    # assert len(history) % 2 == 0

    for i, row in enumerate(history):
        speaker = ""
        response = row

        state, action = None, None
        if i % 2 == 0:  # Agent's turn
            speaker = "Agent"
            if type(row) != str:
                state, action, response = row
        else:  # User's turn
            speaker = "User"

        if show_states_and_actions:
            if state and action:
                output.append(f"User Current State: {state}")
                output.append(f"Agent Action: {action}\n")

            output.append(f"{speaker}: {response}")
        else:
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
