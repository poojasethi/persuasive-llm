from openai import OpenAI
import argparse
import os

MAX_CONVERSATION_TURNS = 10

# Initialize OpenAI API key.
api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)


def generate_persuasive_message(topic, counterarguments=False):
    """
    Generate a persuasive message on a given topic using OpenAI's language model.

    Parameters:
    - topic (str): The topic to persuade the user about.
    - counterarguments (bool): Whether to include counterarguments and rebuttals in the message.

    Returns:
    - str: The persuasive message.
    """

    # Define the prompt that will be sent to the OpenAI model
    prompt = (
        f"Persuade someone to agree with you on this issue: {topic}. "
        "At your turn, please respond by choosing only one of the following actions: "
        "present facts, ask a question, empathize, confirm common ground, or share a personal story."
    )

    try:
        # Call the OpenAI API to generate the persuasive message
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Use the GPT-4 model for better performance
            messages=[
                {
                    "role": "system",
                    "content": f"You are a persuasive agent. Your goal is to convince the user about a specific topic: {topic}.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=200,  # Adjust this for the length of the message
            temperature=0.7,  # Adjust for creativity and persuasiveness
            top_p=1.0,  # Ensures diversity in the content
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )

        # Extract the text from the API response
        breakpoint()
        persuasive_message = response.choices[0].message.content
        return persuasive_message

    except Exception as e:
        return f"Error generating message: {e}"


def main(topic: str):
    message = generate_persuasive_message(topic, counterarguments=True)
    print("Persuasive Message:\n")
    print(message)


if __name__ == "__main__":
    topic = "Everyone should adopt a plant-based diet for environmental reasons"
    main(topic)
