import os

from langchain_core.prompts import ChatPromptTemplate

from typing import Tuple, Union

from utils.common import (
    llm,
    reward_model,
    format_conversation_history,
    parse_agent_response,
    MAX_CONVERSATION_LENGTH
)

prompt = """
At your turn, please respond with the following information:

1. Please predict how the user currently feels about the topic. Choose exactly one of: [disagree, slightly disagree, neutral, slightly agree, agree]
2. Please choose precisely one of the following actions to take: [present facts, ask a question, empathize, confirm common ground, share a personal story, end conversation]
3. Finally, please provide a response to the user consistent with your chosen action. Answer in a conversational manner and try to mirror the style and tone of the user while being respectful. Keep your response from 1 to 3 sentences long.

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


class BaselineAgent:
    def __init__(self, max_conversation_length: int = MAX_CONVERSATION_LENGTH):
        self.max_conversation_length = max_conversation_length

    def start_persuasive_conversation(self, topic: str):
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
        total_rewards = 0

        while not conversation_ended and num_turns < self.max_conversation_length:
            user_input = input("User: ")
            conversation_history.append(user_input)
            conversation_history_str = format_conversation_history(
                conversation_history)

            agent_response = chain.invoke(
                {
                    "topic": topic,
                    "conversation_history": conversation_history_str,
                    "user_input": user_input,
                }
            )

            state, action, response = parse_agent_response(
                agent_response.content)
            conversation_history.append((state, action, response))

            reward = reward_model(state, action)
            total_rewards += reward

            print("Agent:\n", agent_response.content)
            print()

            num_turns += 1
            if action == "end conversation":
                conversation_ended = True

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
    agent = BaselineAgent()
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
