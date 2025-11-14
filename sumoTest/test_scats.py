# train_scats.py
import os
import sys
import numpy as np
import torch

SUMO_HOME = r"C:\Program Files (x86)\Eclipse\Sumo"
TOOLS = os.path.join(SUMO_HOME, "tools")
if TOOLS not in sys.path:
    sys.path.append(TOOLS)

from scats_env import SCATSEnvironment
from scats_agent import SCATSController
import traci

def main():
    episodes = 50
    scores = []
    agent = None
    
    print("🚦 Начало обучения SCATS контроллера...")
    
    for e in range(episodes):
        env = SCATSEnvironment(sumo_cfg="train.sumocfg", gui=False, max_steps=1800, sensor_failure_prob=0.0)
        env.start()
        
        try:
            state = env.get_state()
            if agent is None:
                state_size = len(state)
                tls_logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(env.ts_id)
                action_size = len(tls_logic[0].phases)
                agent = SCATSController(state_size=state_size, action_size=action_size)
                print(f"📊 Размер состояния: {state_size}, Количество фаз: {action_size}")
            
            total_reward = 0
            step_count = 0
            
            while True:
                action = agent.act(state)
                next_state, reward, done = env.step(action, duration=5)
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                step_count += 1
                
                if done:
                    break
            
            # Обучаемся несколько раз на накопленных данных
            for _ in range(3):
                agent.replay()
            
            if e % 5 == 0:
                agent.update_target()
            
            scores.append(total_reward)
            avg_score = np.mean(scores[-5:]) if len(scores) >= 5 else total_reward
            
            print(f"Эпизод {e+1}/{episodes}, Награда: {total_reward:.2f}, "
                  f"Средняя (5): {avg_score:.2f}, Epsilon: {agent.epsilon:.3f}")
                
        except Exception as ex:
            print(f"❌ Ошибка в эпизоде {e}: {ex}")
        finally:
            env.close()
    
    # Сохранение модели
    if agent is not None:
        agent.save("scats_model.pth")
        print("✅ SCATS модель сохранена как 'scats_model.pth'")
        
        final_avg = np.mean(scores[-5:]) if len(scores) >= 5 else scores[-1]
        print(f"🎯 Финальная средняя награда: {final_avg:.2f}")

if __name__ == "__main__":
    main()