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

    user_name = input("What is your name?").strip()
    output_dir = Path(RESULTS_DIR) / user_name
    output_dir.mkdir(exist_ok=True)

    for category, topic in topics.items():
        print(f"Here's the cateogry: {category}")

        for agent_type, agent in agents.items():
            agent_dir = output_dir / agent_type
            agent_dir.mkdir(exist_ok=True)

            results_file = agent_dir / f"{category}.txt"

            print(f"Starting a conversation with the {agent_type} agent.\n")
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


if __name__ == "__main__":
    main(TOPICS)
