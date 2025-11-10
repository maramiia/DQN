# train_dqn.py
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
    # НЕ запускаем env здесь!
    # env.start() будет внутри цикла

    episodes = 100
    target_update_freq = 10
    scores = []

    # Инициализируем агента позже, когда узнаем размерности
    agent = None

    for e in range(episodes):
        # 🔥 Запускаем НОВУЮ симуляцию на каждой эпизоде
        env = SumoEnv(gui=False, max_steps=3600)
        env.start()

        try:
            state = env.get_state()
            if agent is None:  # Инициализируем агента один раз
                state_size = len(state)
                tls_logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(env.ts_id)
                action_size = len(tls_logic[0].phases)
                print(f"State size: {state_size}, Action size: {action_size}")
                agent = DQNAgent(state_size=state_size, action_size=action_size)

            total_reward = 0
            while True:
                action = agent.act(state)
                next_state, reward, done = env.step(action, duration=10)
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                if done:
                    break

            agent.replay()
            if e % target_update_freq == 0:
                agent.update_target()

            scores.append(total_reward)
            print(f"Episode {e+1}/{episodes}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

        finally:
            env.close()  # 🔥 Обязательно закрываем после каждой эпизоды

    # Сохраняем модель
    torch.save(agent.q_network.state_dict(), "dqn_sumo.pth")
    print("✅ Модель сохранена как 'dqn_sumo.pth'")

if __name__ == "__main__":
    main()