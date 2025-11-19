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

def run():
    env = SumoEnv(sumo_cfg="test.sumocfg", gui=False, max_steps=7200, sensor_failure_prob=0.05)
    env.start()
    try:
        state = env.get_state()
        tls_logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(env.ts_id)
        action_size = len(tls_logic[0].phases)
        device = torch.device("cpu")
        model = DQN(len(state), action_size).to(device)
        model.load_state_dict(torch.load("dqn_sumo.pth", map_location=device))
        model.eval()

        total_wait = 0
        step = 0
        while True:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action = model(state_tensor).argmax().item()
            next_state, reward, done = env.step(action, duration=10)
            wait = sum(traci.lane.getWaitingTime(lane) for lane in traci.trafficlight.getControlledLanes(env.ts_id))
            total_wait += wait
            state = next_state
            step += 1
            if done:
                break
        return total_wait / step if step > 0 else 0
    finally:
        env.close()

if __name__ == "__main__":
    avg_wait = run()
    print(f"DQN: {avg_wait:.2f} сек")