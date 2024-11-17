# Persuasive LLM

Planning persuasive dialog with LLMs using online policy learning such as Monte Carlo Tree Search (MCTS). Final Project for Stanford AA/CS 238, Fall 2024.

## Setup

Create the `chat_pomdp` conda environment.
```
conda env create --name chat_pomdp --file=environments.yml
conda activate chat_pomdp
```

## Agents

### Baseline
This agent does not do any planning. It greedily selects the best action to take next.

```
cd agents/
python baseline_langchain.py
```