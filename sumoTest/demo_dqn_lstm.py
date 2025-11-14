# demo_dqn_lstm_simple.py
import os
import sys
import torch
from collections import deque

SUMO_HOME = r"C:\Program Files (x86)\Eclipse\Sumo"
TOOLS = os.path.join(SUMO_HOME, "tools")
if TOOLS not in sys.path:
    sys.path.append(TOOLS)

from sumo_env import SumoEnv
from dqn_agent import LSTMDQN
import traci

def main():
    env = SumoEnv(sumo_cfg="test.sumocfg", gui=True, max_steps=1800, sensor_failure_prob=0.05)
    env.start()

    try:
        state = env.get_state()
        state_size = len(state)
        tls_logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(env.ts_id)
        action_size = len(tls_logic[0].phases)

        print(f"🚦 Демо LSTM-DQN: состояние={state_size}, действий={action_size}")

        device = torch.device("cpu")
        model = LSTMDQN(state_size, action_size).to(device)
        
        # Только LSTM модели
        model_files = ["lstmdqn_best.pth", "lstmdqn_final.pth"]
        model_loaded = False
        
        for model_file in model_files:
            try:
                model.load_state_dict(torch.load(model_file, map_location=device))
                print(f"✅ Загружена LSTM-DQN модель: {model_file}")
                model_loaded = True
                break
            except Exception as e:
                print(f"❌ Не удалось загрузить {model_file}")
                continue
        
        if not model_loaded:
            print("❌ Не найдена LSTM-DQN модель! Сначала обучите:")
            print("   python train_dqn_lstm.py")
            print("📝 Используйте обычный DQN:")
            print("   python demo_dqn.py")
            return
        
        model.eval()

        # Буфер для последовательности
        sequence_buffer = deque(maxlen=5)
        total_wait = 0
        step = 0
        
        while True:
            sequence_buffer.append(state)
            
            if len(sequence_buffer) < 5:
                action = step % action_size
            else:
                sequence = list(sequence_buffer)
                state_tensor = torch.FloatTensor([sequence]).to(device)
                with torch.no_grad():
                    q_values, _ = model(state_tensor)
                    action = q_values.argmax().item()

            next_state, reward, done = env.step(action, duration=10)
            
            lanes = traci.trafficlight.getControlledLanes(env.ts_id)
            step_wait = sum(traci.lane.getWaitingTime(lane) for lane in lanes)
            total_wait += step_wait
            
            state = next_state
            step += 1
            
            if step % 50 == 0:
                print(f"⏱️ Шаг {step}: фаза={action}, ожидание={step_wait:.1f} сек")
                
            if done:
                break

        avg_wait = total_wait / step if step > 0 else 0
        print(f"\n✅ Демо LSTM-DQN завершено!")
        print(f"📊 Среднее время ожидания: {avg_wait:.2f} сек")

    finally:
        env.close()

if __name__ == "__main__":
    main()