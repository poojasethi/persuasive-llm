import os

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from typing import List, Tuple, Dict, Union

from utils.common import reward_model


MAX_CONVERSATION_LENGTH = 10

# Initialize OpenAI API key.
api_key = os.environ.get("OPENAI_API_KEY")

# Replace 'YOUR_OPENAI_API_KEY' with your actual API key
llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0.3)

# prompt = (
#     "At your turn, please respond by choosing exactly one of the following actions: "
#     "[present facts, ask a question, empathize, confirm common ground, or share a personal story. "
#     "Before providing your response, also state which of the actions you chose. "
#     "Answer in a conversationl manner and try to mirror the style and tone of the user while being respectful. "
#     "Don't make your responses more than 3 sentences long. "
#     "User: {user_input}\n"
#     "Your Response: "
# )

prompt = """
At your turn, please respond with the following information:

1. Please predict how the user currently feels about the topic. Choose exactly one of: [disagree, slightly disagree, neutral, slightly agree, agree]
2. Please choose precisely one of the following actions to take: [present facts, ask a question, empathize, confirm common ground, share a personal story, end conversation]
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
    total_rewards = 0

    while not conversation_ended and num_turns < MAX_CONVERSATION_LENGTH:
        user_input = input("User: ")
        conversation_history.append(user_input)
        conversation_history_str = format_conversation_history(conversation_history)

        agent_response = chain.invoke(
            {
                "topic": topic,
                "conversation_history": conversation_history_str,
                "user_input": user_input,
            }
        )

        state, action, response = parse_agent_response(agent_response.content)
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


def main():
    topic = "Everyone should adopt a plant-based diet for environmental reasons"
    start_persuasive_conversation(topic)


if __name__ == "__main__":
    topic = "Everyone should adopt a plant-based diet for environmental reasons"
    # topic = "Abortion should be legalized at a federal level."
    start_persuasive_conversation(topic)
