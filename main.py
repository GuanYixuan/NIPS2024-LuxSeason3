import json
import sys
import numpy as np
from argparse import Namespace
from typing import Dict, Any, List

from agent import Agent
from observation import Observation

### DO NOT REMOVE THE FOLLOWING CODE ###
agent_dict: Dict[str, Agent] = dict()  # store potentially multiple dictionaries as kaggle imports code directly
agent_prev_obs: Dict[str, Any] = dict()

def agent_fn(obs: Namespace, configurations: Dict[str, Any]) -> Dict[str, List[int]]:
    """
    agent definition for kaggle submission.
    """
    global agent_dict

    observation = Observation(obs.step, obs.player, obs.remainingOverageTime, obs.obs)

    step: int = observation.step
    player: str = observation.player
    remainingOverageTime: int = observation.remainingOverageTime
    if step == 0:
        agent_dict[player] = Agent(player, configurations["env_cfg"], observation)
    agent: Agent = agent_dict[player]
    actions: np.ndarray = agent.act(step, observation, remainingOverageTime)
    return dict(action=actions.tolist())

if __name__ == "__main__":
    def read_input() -> str:
        """
        Reads input from stdin
        """
        try:
            return input()
        except EOFError as eof:
            raise SystemExit(eof)

    step: int = 0
    player_id: str
    env_cfg: Dict[str, Any]
    i: int = 0

    while True:
        inputs = read_input()
        raw_input: Dict[str, Any] = json.loads(inputs)
        observation = Namespace(**dict(step=raw_input["step"], obs=raw_input["obs"], remainingOverageTime=raw_input["remainingOverageTime"], player=raw_input["player"], info=raw_input["info"]))
        if i == 0:
            env_cfg = raw_input["info"]["env_cfg"]
            player_id = raw_input["player"]
        i += 1
        actions = agent_fn(observation, dict(env_cfg=env_cfg))
        # send actions to engine
        print(json.dumps(actions))
