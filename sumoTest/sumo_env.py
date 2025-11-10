# sumo_env.py
import os
import sys

SUMO_HOME = r"C:\Program Files (x86)\Eclipse\Sumo"
TOOLS = os.path.join(SUMO_HOME, "tools")
if TOOLS not in sys.path:
    sys.path.append(TOOLS)

import traci

class SumoEnv:
    def __init__(self, sumo_cfg="3x3.sumocfg", gui=False, max_steps=1000):
        self.sumo_cfg = sumo_cfg
        self.gui = gui
        self.max_steps = max_steps
        self.sumo_binary = "sumo-gui" if gui else "sumo"
        self.step_count = 0
        self.ts_id = None

    def start(self):
        if self.gui:
            traci.start([self.sumo_binary, "-c", self.sumo_cfg, "--start", "--quit-on-end"])
        else:
            traci.start([self.sumo_binary, "-c", self.sumo_cfg])
        ts_ids = traci.trafficlight.getIDList()
        if not ts_ids:
            raise RuntimeError("❌ Не найдено светофоров!")
        self.ts_id = ts_ids[0]
        print(f"✅ Подключено к SUMO. Управляем светофором: {self.ts_id}")

    def get_state(self):
        lanes = traci.trafficlight.getControlledLanes(self.ts_id)
        return [traci.lane.getLastStepHaltingNumber(lane) for lane in lanes]

    def set_phase(self, phase_index):
        traci.trafficlight.setPhase(self.ts_id, phase_index)

    # 🔥 Обязательно называется "step"
    def step(self, action, duration=10):
        self.set_phase(action)
        total_wait = 0
        for _ in range(duration):
            traci.simulationStep()
            self.step_count += 1
            lanes = traci.trafficlight.getControlledLanes(self.ts_id)
            total_wait += sum(traci.lane.getLastStepHaltingNumber(lane) for lane in lanes)
        state = self.get_state()
        reward = -total_wait
        done = self.step_count >= self.max_steps
        return state, reward, done

    def close(self):
        traci.close()