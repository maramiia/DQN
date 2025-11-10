# demo_dqn.py
import os
import sys
import torch

SUMO_HOME = r"C:\Program Files (x86)\Eclipse\Sumo"
TOOLS = os.path.join(SUMO_HOME, "tools")
if TOOLS not in sys.path:
    sys.path.append(TOOLS)

from sumo_env import SumoEnv
from dqn_agent import DQN
import traci

def main():
    env = SumoEnv(gui=True, max_steps=3600)
    env.start()

    try:
        state = env.get_state()
        state_size = len(state)
        tls_logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(env.ts_id)
        action_size = len(tls_logic[0].phases)

        # Загружаем обученную модель
        device = torch.device("cpu")  # SUMO обычно на CPU
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

            next_state, reward, done = env.step(action, duration=10)
            total_wait -= reward  # так как reward = -wait
            state = next_state
            step += 1

            if done:
                break

        avg_wait = total_wait / step if step > 0 else 0
        print(f"✅ Среднее время ожидания за симуляцию: {avg_wait:.2f} сек/шаг")

    finally:
        env.close()

if __name__ == "__main__":
    main()