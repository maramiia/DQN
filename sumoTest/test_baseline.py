# test_baseline.py
import os
import sys
SUMO_HOME = r"C:\Program Files (x86)\Eclipse\Sumo"
TOOLS = os.path.join(SUMO_HOME, "tools")
if TOOLS not in sys.path:
    sys.path.append(TOOLS)

from sumo_env import SumoEnv
import traci

def run():
    env = SumoEnv(sumo_cfg="test.sumocfg", gui=False, max_steps=7200, sensor_failure_prob=0.05)
    env.start()
    try:
        tls_logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(env.ts_id)
        phases = tls_logic[0].phases
        fixed_duration = [p.duration for p in phases]
        num_phases = len(phases)
        phase_idx = 0
        time_in_phase = 0
        total_wait = 0

        for step in range(7200):
            if time_in_phase >= fixed_duration[phase_idx]:
                phase_idx = (phase_idx + 1) % num_phases
                time_in_phase = 0
            traci.trafficlight.setPhase(env.ts_id, phase_idx)
            traci.simulationStep()
            time_in_phase += 1
            wait = sum(traci.lane.getWaitingTime(lane) for lane in traci.trafficlight.getControlledLanes(env.ts_id))
            total_wait += wait

        return total_wait / 7200
    finally:
        env.close()

if __name__ == "__main__":
    avg_wait = run()
    print(f"Baseline (фиксированные фазы): {avg_wait:.2f} сек")