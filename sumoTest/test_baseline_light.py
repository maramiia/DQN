# baseline_test.py
import os
import sys

SUMO_HOME = r"C:\Program Files (x86)\Eclipse\Sumo"
TOOLS = os.path.join(SUMO_HOME, "tools")
if TOOLS not in sys.path:
    sys.path.append(TOOLS)

from sumo_env import SumoEnv
import traci

def main():
    env = SumoEnv(sumo_cfg="test.sumocfg", gui=True, max_steps=7200, sensor_failure_prob=0.05)
    env.start()

    try:
        # Получаем фиксированные фазы
        tls_logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(env.ts_id)
        phases = tls_logic[0].phases
        num_phases = len(phases)
        fixed_duration = [p.duration for p in phases]

        total_wait = 0
        phase_idx = 0
        time_in_phase = 0

        for step in range(7200):  # 2 часа = 7200 сек
            if time_in_phase >= fixed_duration[phase_idx]:
                phase_idx = (phase_idx + 1) % num_phases
                time_in_phase = 0
            traci.trafficlight.setPhase(env.ts_id, phase_idx)
            traci.simulationStep()
            time_in_phase += 1

            wait = sum(traci.lane.getWaitingTime(lane) for lane in traci.trafficlight.getControlledLanes(env.ts_id))
            total_wait += wait

        avg_wait = total_wait / 7200
        print(f"⏱️ Среднее время ожидания (baseline, реалистичный сценарий + неопределённость): {avg_wait:.2f} сек")

    finally:
        env.close()

if __name__ == "__main__":
    main()