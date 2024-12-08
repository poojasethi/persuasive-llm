from baseline_agent import BaselineAgent
from sparse_sampling_agent import SpareSamplingAgent
from mcts_agent import MCTSAgent
from utils.common import TOPICS
from pathlib import Path

from typing import List
import os


RESULTS_DIR = "results/"


def write_results():
    pass


def main(topics: List[str]):
    baseline_agent = BaselineAgent()
    sparse_sampling_agent = SpareSamplingAgent()
    mcts_agent = MCTSAgent()
    agents = {
        "baseline": baseline_agent,
        "sparse_sampling": sparse_sampling_agent,
        "mcts": mcts_agent,
    }

    user_name = input("What is your first name?\n").strip().lower()
    output_dir = Path(RESULTS_DIR) / user_name
    output_dir.mkdir(exist_ok=True)

    for start_state in ["disagree", "neutral"]:
        for category, topic in topics.items():
            print(f"Here's the category: {category}. For each topic, act like you initially {start_state} with the agent.")

            for agent_type, agent in agents.items():

                print(f"Starting a conversation with the {agent_type} agent.\n")
                continue_convo = input(f"Would you like to perform this conversation or skip it?\n").strip().lower()

                if continue_convo in ("yes", "y"):
                    agent_dir = output_dir / agent_type / start_state
                    agent_dir.mkdir(exist_ok=False, parents=True)

                    results_file = agent_dir / f"{category}.txt"
                    (
                        user_state,
                        total_rewards,
                        num_turns,
                        conversation_history,
                        conversation_history_str,
                    ) = agent.start_persuasive_conversation(topic)

                    with open(results_file, "w") as fh:
                        fh.write(f"Topic: {topic}\n\n")
                        fh.write(f"Total Rewards: {total_rewards}\n")
                        fh.write(f"User Final State: {user_state}\n")
                        fh.write(f"Conversation Length: {num_turns}\n\n")
                        fh.write(f"Clean Conversation History:\n{conversation_history_str}\n\n")
                        fh.write(f"Full Conversation History:\n{conversation_history}\n\n")

                    print(f"Wrote results to {results_file}\n")
                else:
                    print(f"Skipping this conversation.")


if __name__ == "__main__":
    main(TOPICS)
