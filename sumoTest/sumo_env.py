# sumo_env.py
import os
import sys
import random
import numpy as np

SUMO_HOME = r"C:\Program Files (x86)\Eclipse\Sumo"
TOOLS = os.path.join(SUMO_HOME, "tools")
if TOOLS not in sys.path:
    sys.path.append(TOOLS)

import traci


class SumoEnv:
    def __init__(self, sumo_cfg="5x5.sumocfg", gui=False, max_steps=1000, sensor_failure_prob=0.05):
        self.sumo_cfg = sumo_cfg
        self.gui = gui
        self.max_steps = max_steps
        self.sumo_binary = "sumo-gui" if gui else "sumo"
        self.step_count = 0
        self.ts_id = None
        self.sensor_failure_prob = sensor_failure_prob

    def start(self):
        if self.gui:
            traci.start([self.sumo_binary, "-c", self.sumo_cfg, "--start", "--quit-on-end"])
        else:
            traci.start([self.sumo_binary, "-c", self.sumo_cfg])
        
        ts_ids = traci.trafficlight.getIDList()
        if not ts_ids:
            raise RuntimeError("❌ Не найдено светофоров!")
        
        # 🔑 Выбираем светофор с наибольшим числом управляемых полос (обычно центральный)
        best_ts = ts_ids[0]
        max_lanes = len(traci.trafficlight.getControlledLanes(ts_ids[0]))
        for ts_id in ts_ids[1:]:
            num_lanes = len(traci.trafficlight.getControlledLanes(ts_id))
            if num_lanes > max_lanes:
                max_lanes = num_lanes
                best_ts = ts_id

        self.ts_id = best_ts
        print(f"✅ Подключено к SUMO. Управляем светофором: {self.ts_id} (полос: {max_lanes})")


    def get_state(self):
        lanes = traci.trafficlight.getControlledLanes(self.ts_id)
        state = []
        for lane in lanes:
            halting = traci.lane.getLastStepHaltingNumber(lane)
            occupancy = traci.lane.getLastStepVehicleNumber(lane)
            mean_speed = traci.lane.getLastStepMeanSpeed(lane)
            # Убрали дублирование — 3 признака на полосу
            state.extend([halting, occupancy, mean_speed])
        
        state = np.array(state, dtype=np.float32)

        if random.random() < self.sensor_failure_prob:
            state = np.zeros_like(state)
        
        return state

    def set_phase(self, phase_index):
        traci.trafficlight.setPhase(self.ts_id, phase_index)

    def step(self, action, duration=5):  # duration=5 вместо 10
        self.set_phase(action)
        total_wait = 0
        for _ in range(duration):
            traci.simulationStep()
            self.step_count += 1
            lanes = traci.trafficlight.getControlledLanes(self.ts_id)
            for lane in lanes:
                total_wait += traci.lane.getWaitingTime(lane)

        # Упрощённая награда
        reward = -total_wait / 100.0  # нормализация
        state = self.get_state()
        done = self.step_count >= self.max_steps
        return state, reward, done

    def close(self):
        traci.close()