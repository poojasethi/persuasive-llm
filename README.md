# Persuasive LLM

Planning persuasive dialog with LLMs using online policy learning such as Monte Carlo Tree Search (MCTS). Final Project for Stanford AA/CS 238, Fall 2024.

## Setup

Create the `persuasive_llm` conda environment.
```
conda env create --name persuasive_llm --file=environment.yml
conda activate persuasive_llm
```

## Agents
Before running the agents, you will need to make sure to set your `OPENAI_API_KEY`. 
```
export OPENAI_API_KEY='sk-YOUR_KEY_HERE'
```

### Baseline
This agent does not do any explicit planning. It selects the action to take using the LLM's predicted best action.
```
cd agents/
python baseline_agent.py
```

### MCTS
This agent plans the conversation actions ahead of time using MCTS.
```
cd agents/
python mcts_agent.py
```
