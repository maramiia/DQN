# train_dqn.py - ИСПРАВЛЕННАЯ ВЕРСИЯ
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
    episodes = 50  # Увеличим количество эпизодов
    scores = []
    best_score = -float('inf')
    
    print("🚦 Начало обучения DQN-LSTM...")
    
    for e in range(episodes):
        env = SumoEnv(sumo_cfg="train.sumocfg", gui=False, max_steps=1800, sensor_failure_prob=0.0)  # Уменьшим шаги
        env.start()

        try:
            state = env.get_state()
            if e == 0:  # Инициализация только в первом эпизоде
                state_size = len(state)
                tls_logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(env.ts_id)
                action_size = len(tls_logic[0].phases)
                agent = DQNAgent(state_size=state_size, action_size=action_size, sequence_length=5)
                print(f"🤖 DQN-LSTM: состояние {state_size}, действий {action_size}")
            else:
                agent.reset_sequence()  # Сброс последовательности

            total_reward = 0
            step_count = 0
            
            while True:
                action = agent.act(state)
                next_state, reward, done = env.step(action, duration=10)  # Увеличим duration
                
                # Сохраняем в память
                agent.remember_sequence(state, action, reward, next_state, done)
                
                state = next_state
                total_reward += reward
                step_count += 1
                
                # Обучение каждые 4 шага
                if step_count % 4 == 0:
                    agent.replay()
                
                if done:
                    break

            # Обновление целевой сети
            if e % 20 == 0:
                agent.update_target()

            scores.append(total_reward)
            
            # Сохранение лучшей модели
            if total_reward > best_score:
                best_score = total_reward
                torch.save(agent.q_network.state_dict(), "dqn_lstm_best.pth")
                print(f"💾 Сохранена лучшая модель с наградой: {best_score:.2f}")

            avg_score = np.mean(scores[-10:]) if len(scores) >= 10 else total_reward
            
            print(f"Эпизод {e+1}/{episodes}, Награда: {total_reward:.2f}, "
                  f"Средняя (10): {avg_score:.2f}, Epsilon: {agent.epsilon:.3f}")
                
        except Exception as ex:
            print(f"❌ Ошибка в эпизоде {e}: {ex}")
            import traceback
            traceback.print_exc()
        finally:
            env.close()

    # Финальное сохранение
    torch.save(agent.q_network.state_dict(), "dqn_lstm_final.pth")
    print("✅ Обучение завершено!")
    print(f"🎯 Лучшая награда: {best_score:.2f}")
    print(f"📊 Финальная средняя награда: {np.mean(scores[-20:]):.2f}")

if __name__ == "__main__":
    main()