import os
import sys
import numpy as np
import torch

SUMO_HOME = r"C:\Program Files (x86)\Eclipse\Sumo"
TOOLS = os.path.join(SUMO_HOME, "tools")
if TOOLS not in sys.path:
    sys.path.append(TOOLS)

from sumo_env import SumoEnv
from dqn_agent import DQNAgent
import traci

def main():
    episodes = 100
    scores = []
    agent = None

    for e in range(episodes):
        env = SumoEnv(sumo_cfg="train.sumocfg", gui=False, max_steps=3600, sensor_failure_prob=0.0)
        env.start()

        try:
            state = env.get_state()
            if agent is None:
                state_size = len(state)
                tls_logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(env.ts_id)
                action_size = len(tls_logic[0].phases)
                agent = DQNAgent(state_size=state_size, action_size=action_size)

            total_reward = 0
            while True:
                action = agent.act(state)
                next_state, reward, done = env.step(action, duration=5)
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                if done:
                    break

            agent.replay()
            if e % 10 == 0:
                agent.update_target()

            scores.append(total_reward)
            print(f"Episode {e+1}/{episodes}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

        finally:
            env.close()

    torch.save(agent.q_network.state_dict(), "dqn_sumo.pth")
    print("Модель сохранена")

if __name__ == "__main__":
    main()