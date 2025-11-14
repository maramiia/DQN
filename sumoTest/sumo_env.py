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
            vehicles = traci.lane.getLastStepVehicleNumber(lane)
            mean_speed = max(traci.lane.getLastStepMeanSpeed(lane), 0.1)
            
            # Простая нормализация
            norm_halting = min(halting / 10.0, 1.0)
            norm_vehicles = min(vehicles / 15.0, 1.0)
            norm_speed = mean_speed / 13.89
            
            state.extend([norm_halting, norm_vehicles, norm_speed])
        
        state = np.array(state, dtype=np.float32)

        if random.random() < self.sensor_failure_prob:
            state = np.zeros_like(state)
        
        return state

    def set_phase(self, phase_index):
        traci.trafficlight.setPhase(self.ts_id, phase_index)

    def step(self, action, duration=10):
        self.set_phase(action)
        
        total_waiting = 0
        total_stopped = 0
        
        for _ in range(duration):
            traci.simulationStep()
            self.step_count += 1
            
            # Считаем метрики
            lanes = traci.trafficlight.getControlledLanes(self.ts_id)
            for lane in lanes:
                total_waiting += traci.lane.getWaitingTime(lane)
                total_stopped += traci.lane.getLastStepHaltingNumber(lane)

        # ИСПРАВЛЕННАЯ ФУНКЦИЯ НАГРАДЫ - ДЕЛАЕМ ЕЁ ПОНЯТНОЙ
        reward = self.calculate_reward(total_waiting, total_stopped)
        state = self.get_state()
        done = self.step_count >= self.max_steps
        
        return state, reward, done

    def calculate_reward(self, total_waiting, total_stopped):
        """Упрощенная и эффективная функция награды"""
        # Основной штраф - за время ожидания (нормализованный)
        wait_penalty = -total_waiting / 1000.0
        
        # Дополнительный штраф за стоящие автомобили
        stop_penalty = -total_stopped * 0.1
        
        # Общая награда (должна быть в разумных пределах: -10 до 10)
        reward = wait_penalty + stop_penalty
        
        # Ограничиваем награду
        reward = max(min(reward, 10.0), -10.0)
        
        return reward

    def close(self):
        traci.close()