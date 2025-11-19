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

def run():
    sequence_length = 5
    env = SumoHistoryEnv(
        sumo_cfg="test.sumocfg",
        gui=False,
        max_steps=7200,
        sensor_failure_prob=0.05,
        sequence_length=sequence_length
    )
    state_seq = env.reset()
    try:
        tls_logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(env.ts_id)
        action_size = len(tls_logic[0].phases)
        device = torch.device("cpu")
        model = DRQN(len(state_seq[0]), action_size, hidden_dim=128).to(device)
        model.load_state_dict(torch.load("drqn_sumo.pth", map_location=device))
        model.eval()

        total_wait = 0
        step = 0
        while True:
            state_tensor = torch.FloatTensor(state_seq).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values, _ = model(state_tensor)
                action = q_values.argmax().item()
            next_state_seq, reward, done = env.step_with_history(action, duration=10)
            wait = sum(traci.lane.getWaitingTime(lane) for lane in traci.trafficlight.getControlledLanes(env.ts_id))
            total_wait += wait
            state_seq = next_state_seq
            step += 1
            if done:
                break
        return total_wait / step if step > 0 else 0
    finally:
        env.close()

if __name__ == "__main__":
    avg_wait = run()
    print(f"DRQN: {avg_wait:.2f} сек")