# compare_scats_vs_dqn.py (дополненная версия)
import os
import sys
import torch

SUMO_HOME = r"C:\Program Files (x86)\Eclipse\Sumo"
TOOLS = os.path.join(SUMO_HOME, "tools")
if TOOLS not in sys.path:
    sys.path.append(TOOLS)

from scats_env import SCATSEnvironment
from scats_agent import SCATSController
from sumo_env import SumoEnv
from dqn_agent import DQN_LSTM, DQNAgent
import traci

def run_lstm_dqn():
    """Запуск LSTM-DQN контроллера"""
    env = SumoEnv(sumo_cfg="test.sumocfg", gui=False, max_steps=1800, sensor_failure_prob=0.05)
    env.start()
    
    try:
        state = env.get_state()
        state_size = len(state)
        tls_logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(env.ts_id)
        action_size = len(tls_logic[0].phases)
        
        device = torch.device("cpu")
        model = DQN_LSTM(state_size, action_size).to(device)
        model.load_state_dict(torch.load("dqn_lstm_sumo.pth", map_location=device))
        model.eval()
        
        # Инициализация буфера состояний для LSTM
        state_buffer = deque(maxlen=5)
        for _ in range(5):
            state_buffer.append([0.0] * state_size)
        
        total_waiting_time = 0
        step = 0
        
        while True:
            # Обновляем буфер и создаем последовательность
            state_buffer.append(state)
            state_sequence = torch.FloatTensor([list(state_buffer)]).to(device)
            
            with torch.no_grad():
                q_values, _ = model(state_sequence)
                action = q_values.argmax().item()
            
            next_state, reward, done = env.step(action, duration=10)
            
            lanes = traci.trafficlight.getControlledLanes(env.ts_id)
            step_waiting = sum(traci.lane.getWaitingTime(lane) for lane in lanes)
            total_waiting_time += step_waiting
            
            state = next_state
            step += 1
            if done:
                break
        
        return total_waiting_time / step if step > 0 else 0
    
    finally:
        env.close()

if __name__ == "__main__":
    print("🔍 Сравнение SCATS vs DQN vs LSTM-DQN...")
    
    scats_avg_wait = run_scats()
    dqn_avg_wait = run_dqn()
    lstm_dqn_avg_wait = run_lstm_dqn()
    
    print(f"\n📊 РЕЗУЛЬТАТЫ СРАВНЕНИЯ:")
    print(f"   SCATS контроллер: {scats_avg_wait:.2f} сек")
    print(f"   DQN контроллер: {dqn_avg_wait:.2f} сек")
    print(f"   LSTM-DQN контроллер: {lstm_dqn_avg_wait:.2f} сек")
    
    # Находим лучший результат
    results = {
        "SCATS": scats_avg_wait,
        "DQN": dqn_avg_wait,
        "LSTM-DQN": lstm_dqn_avg_wait
    }
    
    best_model = min(results, key=results.get)
    best_value = results[best_model]
    
    print(f"\n🏆 ЛУЧШАЯ МОДЕЛЬ: {best_model} ({best_value:.2f} сек)")
    
    # Сравниваем улучшения
    baseline = max(results.values())
    improvement = ((baseline - best_value) / baseline) * 100
    print(f"   📈 Улучшение относительно худшей модели: {improvement:.1f}%")