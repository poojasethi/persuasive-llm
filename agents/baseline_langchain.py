import os

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

MAX_CONVERSATION_LENGTH = 10

# Initialize OpenAI API key.
api_key = os.environ.get("OPENAI_API_KEY")

# Replace 'YOUR_OPENAI_API_KEY' with your actual API key
llm = ChatOpenAI(api_key=api_key, temperature=0.7)

prompt = (
    "At your turn, please respond by choosing only one of the following actions: "
    "present facts, ask a question, empathize, confirm common ground, or share a personal story. "
    "Before providing your response, also state which of the actions you chose. "
    "Answer in a conversationl manner and try to mirror the style and tone of the user while being respectful. "
    "Don't make your responses more than 3 sentences long. "
    "User: {user_input}\n"
    "Your Response: "
)

def start_persuasive_conversation(topic: str):
    print(f'Agent: What are your thoughts on this topic: "{topic}"?')

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", f"You are a persuasive agent. Your goal is to convince the human to agree with you on this issue: {topic}."),
            ("human", prompt),
        ]
    )
    chain = prompt_template | llm 

    num_turns = 0
    while True and num_turns < MAX_CONVERSATION_LENGTH:
        user_input = input("User: ")
        response = chain.invoke({"topic": topic, "user_input": user_input})
        print("Agent:", response.content)
        num_turns += 1


if __name__ == "__main__":
    topic = "Everyone should adopt a plant-based diet for environmental reasons"
    start_persuasive_conversation(topic)
