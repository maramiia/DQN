# animate_dual_heavy.py
import os
import sys
import torch
import numpy as np

SUMO_HOME = r"C:\Program Files (x86)\Eclipse\Sumo"
TOOLS = os.path.join(SUMO_HOME, "tools")
if TOOLS not in sys.path:
    sys.path.append(TOOLS)

import traci
from sumo_env import SumoEnv
from drqn_agent import DRQN

class DualHeavyVisualizer:
    def __init__(self):
        self.gui = True
        self.sumo_cfg = "dual_heavy.sumocfg"
        self.max_steps = 3600  # 1 час
        self.sequence_length = 5
        self.drqn_history = None
        self.drqn_model = None

    def start_sumo(self):
        sumo_binary = "sumo-gui" if self.gui else "sumo"
        traci.start([sumo_binary, "-c", self.sumo_cfg, "--start"])
        print("✅ SUMO запущен")

    def load_drqn(self, state_size, action_size):
        device = torch.device("cpu")
        self.drqn_model = DRQN(state_size, action_size, hidden_dim=128).to(device)
        self.drqn_model.load_state_dict(torch.load("drqn_sumo.pth", map_location=device))
        self.drqn_model.eval()
        self.drqn_history = [np.zeros(state_size, dtype=np.float32) for _ in range(self.sequence_length)]
        print("✅ DRQN модель загружена")

    def get_state(self, ts_id):
        lanes = traci.trafficlight.getControlledLanes(ts_id)
        state = []
        for lane in lanes:
            halting = traci.lane.getLastStepHaltingNumber(lane)
            occupancy = traci.lane.getLastStepVehicleNumber(lane)
            mean_speed = traci.lane.getLastStepMeanSpeed(lane)
            state.extend([halting, occupancy, mean_speed])
        return np.array(state, dtype=np.float32)

    def run(self):
        self.start_sumo()
        ts_ids = traci.trafficlight.getIDList()
        if len(ts_ids) < 2:
            raise RuntimeError("Нужно минимум 2 светофора!")

        # Предполагаем: ts_ids[0] — левый (Baseline), ts_ids[1] — правый (DRQN)
        baseline_ts = ts_ids[0]
        drqn_ts = ts_ids[1]
        print(f"🚦 Baseline TS: {baseline_ts}, DRQN TS: {drqn_ts}")

        # Получаем фазы для Baseline
        tls_logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(baseline_ts)
        fixed_durations = [p.duration for p in tls_logic[0].phases]
        phase_idx = 0
        time_in_phase = 0

        # Инициализация DRQN
        state_drqn = self.get_state(drqn_ts)
        state_size = len(state_drqn)
        action_size = len(traci.trafficlight.getCompleteRedYellowGreenDefinition(drqn_ts)[0].phases)
        self.load_drqn(state_size, action_size)

        try:
            for step in range(self.max_steps):
                # === Baseline: фиксированный цикл ===
                if time_in_phase >= fixed_durations[phase_idx]:
                    phase_idx = (phase_idx + 1) % len(fixed_durations)
                    time_in_phase = 0
                traci.trafficlight.setPhase(baseline_ts, phase_idx)
                time_in_phase += 1

                # === DRQN: адаптивное управление ===
                new_state = self.get_state(drqn_ts)
                self.drqn_history = self.drqn_history[1:] + [new_state]
                seq = np.array(self.drqn_history)
                state_tensor = torch.FloatTensor(seq).unsqueeze(0)
                with torch.no_grad():
                    q_vals, _ = self.drqn_model(state_tensor)
                    action = q_vals.argmax().item()
                traci.trafficlight.setPhase(drqn_ts, action)

                # Шаг симуляции
                traci.simulationStep()

                # Лог каждые 500 шагов
                if (step + 1) % 500 == 0:
                    bl_wait = sum(traci.lane.getWaitingTime(l) for l in traci.trafficlight.getControlledLanes(baseline_ts))
                    drqn_wait = sum(traci.lane.getWaitingTime(l) for l in traci.trafficlight.getControlledLanes(drqn_ts))
                    print(f"⏱️ {step+1} сек | Baseline очередь: {bl_wait:.0f} | DRQN очередь: {drqn_wait:.0f}")

        finally:
            traci.close()
            print("🔚 Симуляция завершена")


if __name__ == "__main__":
    viz = DualHeavyVisualizer()
    viz.run()