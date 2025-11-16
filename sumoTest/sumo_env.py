# sumo_env.py - ИСПРАВЛЕННАЯ ВЕРСИЯ
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
    def __init__(self, sumo_cfg="train.sumocfg", gui=False, max_steps=3600, sensor_failure_prob=0.0):
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
            mean_speed = max(traci.lane.getLastStepMeanSpeed(lane), 0.1)  # Избегаем нулевой скорости
            # Нормализуем значения
            norm_halting = halting / 20.0  # предполагаем макс 20 машин в очереди
            norm_occupancy = occupancy / 30.0  # предполагаем макс 30 машин
            norm_speed = mean_speed / 13.89  # нормализуем к макс скорости (~50 км/ч)
            
            state.extend([norm_halting, norm_occupancy, norm_speed])
        
        state = np.array(state, dtype=np.float32)

        if random.random() < self.sensor_failure_prob:
            state = np.zeros_like(state)
        
        return state

    def set_phase(self, phase_index):
        traci.trafficlight.setPhase(self.ts_id, phase_index)

    def step(self, action, duration=10):  # Увеличим duration для стабильности
        self.set_phase(action)
        total_wait = 0
        vehicles_passed = 0
        
        for _ in range(duration):
            traci.simulationStep()
            self.step_count += 1
            
            # Собираем более разнообразные метрики
            lanes = traci.trafficlight.getControlledLanes(self.ts_id)
            for lane in lanes:
                total_wait += traci.lane.getWaitingTime(lane)
                vehicles_passed += traci.lane.getLastStepVehicleNumber(lane)

        # УЛУЧШЕННАЯ ФУНКЦИЯ НАГРАДЫ
        reward = self.calculate_reward(total_wait, vehicles_passed, duration)
        state = self.get_state()
        done = self.step_count >= self.max_steps
        
        return state, reward, done

    def calculate_reward(self, total_wait, vehicles_passed, duration):
        """Улучшенная функция награды"""
        # Штраф за время ожидания (нормализованный)
        wait_penalty = -total_wait / (100.0 * duration)
        
        # Награда за пропускную способность
        throughput_reward = vehicles_passed / (10.0 * duration)
        
        # Комбинированная награда
        reward = wait_penalty + throughput_reward
        
        return reward

    def close(self):
        traci.close()