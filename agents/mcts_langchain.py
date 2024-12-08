import os

from langchain_core.prompts import ChatPromptTemplate

from algorithms.mcts import MonteCarloTreeSearch

from typing import List, Tuple, Dict, Union

from utils.common import (
    llm,
    reward_model,
    format_conversation_history,
    parse_agent_response,
    parse_user_state_agent_response,
    mdp,
    MAX_CONVERSATION_LENGTH,
    STATES,
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


class MCTSAgent:
    def __init__(self, max_conversation_length: int = MAX_CONVERSATION_LENGTH):
        self.max_conversation_length = max_conversation_length

    def init_mcts(self):
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
                "slightly agree": 1,
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
        return mcts

    def start_persuasive_conversation(self, topic: str):
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
        mcts = self.init_mcts()

        while not conversation_ended and num_turns < self.max_conversation_length:
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
            assert (
                action == parsed_action
            ), "Provided action and parsed action don't match"

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

        return (
            state,
            total_rewards,
            num_turns,
            conversation_history,
            conversation_history_str,
        )


def main(topic: str):
    agent = MCTSAgent()
    (
        user_state,
        total_rewards,
        num_turns,
        conversation_history,
        conversation_history_str,
    ) = agent.start_persuasive_conversation(topic)


if __name__ == "__main__":
    topic = "Everyone should adopt a plant-based diet for environmental reasons"
    main(topic)
