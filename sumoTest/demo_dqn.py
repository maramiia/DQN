# test_dqn.py
import os
import sys
import torch
import random

SUMO_HOME = r"C:\Program Files (x86)\Eclipse\Sumo"
TOOLS = os.path.join(SUMO_HOME, "tools")
if TOOLS not in sys.path:
    sys.path.append(TOOLS)

from sumo_env import SumoEnv
from dqn_agent import DQN
import traci

def main():
    # 🔑 ТЕСТОВЫЙ СЦЕНАРИЙ + НЕОПРЕДЕЛЁННОСТЬ
    env = SumoEnv(sumo_cfg="test.sumocfg", gui=True, max_steps=7200, sensor_failure_prob=0.05)
    env.start()

    try:
        state = env.get_state()
        state_size = len(state)
        tls_logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(env.ts_id)
        action_size = len(tls_logic[0].phases)

        device = torch.device("cpu")
        model = DQN(state_size, action_size).to(device)
        model.load_state_dict(torch.load("dqn_sumo.pth", map_location=device))
        model.eval()

        total_wait = 0
        step = 0
        while True:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = model(state_tensor)
                action = q_values.argmax().item()

            next_state, reward, done = env.step(action, duration=5)
            total_wait -= reward * 100  # обратно к секундам
            state = next_state
            step += 1
            if done:
                break

        avg_wait = total_wait / step if step > 0 else 0
        print(f"✅ Среднее время ожидания (реалистичный сценарий + неопределённость): {avg_wait:.2f} сек")

    finally:
        env.close()

if __name__ == "__main__":
    main()