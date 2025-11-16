# demo_dqn.py
import os
import sys
import torch
import numpy as np

SUMO_HOME = r"C:\Program Files (x86)\Eclipse\Sumo"
TOOLS = os.path.join(SUMO_HOME, "tools")
if TOOLS not in sys.path:
    sys.path.append(TOOLS)

from sumo_env import SumoEnv
from dqn_agent import LSTMDQN  # ← ИМПОРТИРУЕМ LSTM-версию
import traci

def main():
    env = SumoEnv(sumo_cfg="test.sumocfg", gui=True, max_steps=7200, sensor_failure_prob=0.05)
    env.start()

    try:
        state = env.get_state()
        state_size = len(state)
        tls_logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(env.ts_id)
        action_size = len(tls_logic[0].phases)

        device = torch.device("cpu")
        model = LSTMDQN(state_size, action_size).to(device)
        model.load_state_dict(torch.load("dqn_sumo.pth", map_location=device))
        model.eval()

        # Для подсчёта метрики — как в baseline
        total_wait = 0
        simulation_step = 0
        next_change = 0
        state_history = []  # ← ИСТОРИЯ СОСТОЯНИЙ

        while simulation_step < 7200:
            if simulation_step >= next_change:
                state_history.append(state)
                # Формируем последовательность длины 5
                if len(state_history) < 5:
                    padded = [np.zeros(state_size) for _ in range(5 - len(state_history))] + state_history
                else:
                    padded = state_history[-5:]
                
                state_tensor = torch.FloatTensor([padded]).to(device)
                with torch.no_grad():
                    q_values = model(state_tensor)
                    action = q_values.argmax().item()
                
                next_change = simulation_step + 5

            traci.trafficlight.setPhase(env.ts_id, action)
            traci.simulationStep()
            simulation_step += 1

            wait_now = sum(traci.lane.getWaitingTime(lane) for lane in traci.trafficlight.getControlledLanes(env.ts_id))
            total_wait += wait_now

            # Обновляем состояние каждую секунду
            state = env.get_state()

        avg_wait = total_wait / 7200
        print(f"✅ Среднее время ожидания (LSTM-DQN, реалистичный сценарий): {avg_wait:.2f} сек")

    finally:
        env.close()

if __name__ == "__main__":
    main()