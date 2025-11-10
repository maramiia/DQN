# baseline_fixed.py
import os
import sys

SUMO_HOME = r"C:\Program Files (x86)\Eclipse\Sumo"
TOOLS = os.path.join(SUMO_HOME, "tools")
if TOOLS not in sys.path:
    sys.path.append(TOOLS)

from sumo_env import SumoEnv
import traci

def main():
    env = SumoEnv(gui=True, max_steps=3600)
    env.start()

    try:
        # Получаем исходные фазы из SUMO (фиксированный цикл)
        tls_logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(env.ts_id)
        phases = tls_logic[0].phases
        num_phases = len(phases)
        fixed_duration = [p.duration for p in phases]

        total_wait = 0
        phase_idx = 0
        time_in_phase = 0

        for step in range(3600):
            if time_in_phase >= fixed_duration[phase_idx]:
                phase_idx = (phase_idx + 1) % num_phases
                time_in_phase = 0
            traci.trafficlight.setPhase(env.ts_id, phase_idx)
            traci.simulationStep()
            time_in_phase += 1

            # Считаем ожидание
            wait = sum(traci.lane.getWaitingTime(lane) for lane in traci.trafficlight.getControlledLanes(env.ts_id))
            total_wait += wait

        avg_wait = total_wait / 3600
        print(f"⏱️ Среднее время ожидания (фиксированные фазы): {avg_wait:.2f} сек")

    finally:
        env.close()

if __name__ == "__main__":
    main()