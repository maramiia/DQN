import os
import sys
import torch
import numpy as np

SUMO_HOME = r"C:\Program Files (x86)\Eclipse\Sumo"
TOOLS = os.path.join(SUMO_HOME, "tools")
if TOOLS not in sys.path:
    sys.path.append(TOOLS)

from sumo_history_env import SumoHistoryEnv
from drqn_agent import DRQN
import traci

def main():
    print("Тестирование DRQN на СЦЕНАРИИ С ВЫСОКОЙ НАГРУЗКОЙ (heavy_test.sumocfg)...")
    
    sequence_length = 5
    env = SumoHistoryEnv(
        sumo_cfg="heavy_test.sumocfg",  
        gui=False,  
        max_steps=7200,    
        sensor_failure_prob=0.05, 
        sequence_length=sequence_length
    )
    state_seq = env.reset()
    
    try:
        tls_logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(env.ts_id)
        action_size = len(tls_logic[0].phases)
        state_size = len(state_seq[0])
        print(f"Управляемый светофор: {env.ts_id}")
        print(f"Фаз: {action_size} | Состояние: {state_size} | Последовательность: {sequence_length}")

        device = torch.device("cpu")
        model = DRQN(state_size, action_size, hidden_dim=128).to(device)
        model.load_state_dict(torch.load("drqn_sumo.pth", map_location=device))
        model.eval()
        print("DRQN модель загружена (drqn_sumo.pth)")

        total_wait = 0
        total_reward = 0
        step_count = 0

        while True:
            state_tensor = torch.FloatTensor(state_seq).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values, _ = model(state_tensor)
                action = q_values.argmax().item()

            next_state_seq, reward, done = env.step_with_history(action, duration=10)

            wait = sum(traci.lane.getWaitingTime(lane) for lane in traci.trafficlight.getControlledLanes(env.ts_id))
            total_wait += wait
            total_reward += reward

            state_seq = next_state_seq
            step_count += 1

            if step_count % 20 == 0:
                avg_wait_step = total_wait / (step_count * 10)
                print(f"Шаг {step_count*10} сек | Фаза {action} | Ср. ожидание: {avg_wait_step:.2f} сек")

            if done:
                break

        sim_time_sec = step_count * 10
        avg_wait = total_wait / sim_time_sec if sim_time_sec > 0 else 0
        print("\n" + "="*60)
        print("РЕЗУЛЬТАТЫ DRQN НА HEAVY-СЦЕНАРИИ:")
        print(f"Продолжительность симуляции: {sim_time_sec} сек ({sim_time_sec/3600:.1f} ч)")
        print(f"Шагов агента: {step_count}")
        print(f"Суммарное ожидание: {total_wait:.0f} сек")
        print(f"Среднее время ожидания: {avg_wait:.2f} сек/автомобиль/секунда")
        print(f"Суммарная награда: {total_reward:.2f}")
        print("="*60)

        with open("drqn_heavy_result.txt", "w", encoding="utf-8") as f:
            f.write(f"Среднее время ожидания (DRQN, heavy_test): {avg_wait:.2f} сек\n")
            f.write(f"Суммарная награда: {total_reward:.2f}\n")
            f.write(f"Продолжительность: {sim_time_sec} сек\n")
            f.write(f"Шагов: {step_count}\n")

    finally:
        env.close()
        print("Симуляция завершена.")

if __name__ == "__main__":
    main()