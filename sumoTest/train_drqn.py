import os
import sys
import numpy as np
import torch

SUMO_HOME = r"C:\Program Files (x86)\Eclipse\Sumo"
TOOLS = os.path.join(SUMO_HOME, "tools")
if TOOLS not in sys.path:
    sys.path.append(TOOLS)

from sumo_history_env import SumoHistoryEnv
from drqn_agent import DRQNAgent
import traci

def main():
    episodes = 100
    sequence_length = 5
    scores = []
    agent = None

    for e in range(episodes):
        env = SumoHistoryEnv(
            sumo_cfg="train.sumocfg",
            gui=False,
            max_steps=3600,
            sensor_failure_prob=0.0,
            sequence_length=sequence_length
        )
        state_seq = env.reset()

        try:
            if agent is None:
                state_size = len(state_seq[0])
                tls_logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(env.ts_id)
                action_size = len(tls_logic[0].phases)
                agent = DRQNAgent(
                    state_size=state_size,
                    action_size=action_size,
                    sequence_length=sequence_length,
                    epsilon_decay=0.98 
                )
                print(f"State size: {state_size}, Action size: {action_size}")

            total_reward = 0
            step_count = 0
            while True:
                action = agent.act(state_seq)
                next_state_seq, reward, done = env.step_with_history(action, duration=10)
                agent.remember(state_seq, action, reward, next_state_seq, done)
                state_seq = next_state_seq
                total_reward += reward
                step_count += 1

                if step_count % 20 == 0:
                    agent.replay()

                if done:
                    break

            for _ in range(3):
                agent.replay()

            if e % 10 == 0:
                agent.update_target()

            scores.append(total_reward)
            avg_last5 = np.mean(scores[-5:]) if len(scores) >= 5 else total_reward
            print(f"Episode {e+1}/{episodes}, Reward: {total_reward:.2f}, "
                  f"Avg(5): {avg_last5:.2f}, Epsilon: {agent.epsilon:.3f}")

        except Exception as ex:
            print(f"Ошибка в эпизоде {e+1}: {ex}")
        finally:
            env.close()

    torch.save(agent.q_network.state_dict(), "drqn_sumo.pth")
    print("DRQN модель сохранена как 'drqn_sumo.pth'")

if __name__ == "__main__":
    main()