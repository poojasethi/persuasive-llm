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
    elif state == "slighty disagree":
        action_to_reward = {
            "present facts": 5,
            "ask a question": 7,
            "empathize": 10,
            "confirm common ground": 0,
            "share a personal story": 10,
            "end conversation": -10,
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
            "end conversation": 1,
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
