# train_dqn_simple.py
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
    episodes = 30  # Меньше эпизодов для теста
    scores = []
    best_score = -float('inf')
    agent = None
    
    print("🚦 Быстрое обучение DQN...")
    
    for e in range(episodes):
        env = SumoEnv(sumo_cfg="train.sumocfg", gui=False, max_steps=900, sensor_failure_prob=0.0)  # Укоротили симуляцию
        env.start()

        try:
            state = env.get_state()
            
            if agent is None:
                state_size = len(state)
                tls_logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(env.ts_id)
                action_size = len(tls_logic[0].phases)
                agent = DQNAgent(state_size=state_size, action_size=action_size)
                print(f"📊 Размер состояния: {state_size}, Действий: {action_size}")

            total_reward = 0
            step_count = 0
            
            while True:
                action = agent.act(state)
                next_state, reward, done = env.step(action, duration=5)  # Укоротили фазы
                
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                step_count += 1
                
                # Обучаемся реже для скорости
                if step_count % 10 == 0:
                    agent.replay()
                
                if done:
                    break

            # Обновляем целевую сеть
            if e % 5 == 0:
                agent.update_target()

            scores.append(total_reward)
            
            # Сохраняем лучшую модель
            if total_reward > best_score:
                best_score = total_reward
                filename = "dqn_simple_best.pth"
                agent.save(filename)
                print(f"💾 Новая лучшая модель! Награда: {best_score:.2f}")

            avg_score = np.mean(scores[-5:]) if len(scores) >= 5 else total_reward
            
            print(f"Эпизод {e+1}/{episodes}, Награда: {total_reward:.2f}, "
                  f"Средняя (5): {avg_score:.2f}, Epsilon: {agent.epsilon:.3f}")
                
        except Exception as ex:
            print(f"❌ Ошибка в эпизоде {e}: {ex}")
        finally:
            env.close()

    # Финальное сохранение
    final_filename = "dqn_simple_final.pth"
    agent.save(final_filename)
    print(f"✅ Обучение завершено!")
    print(f"🎯 Лучшая награда: {best_score:.2f}")

if __name__ == "__main__":
    main()