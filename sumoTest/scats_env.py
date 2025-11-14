# scats_env.py
import os
import sys
import random
import numpy as np

SUMO_HOME = r"C:\Program Files (x86)\Eclipse\Sumo"
TOOLS = os.path.join(SUMO_HOME, "tools")
if TOOLS not in sys.path:
    sys.path.append(TOOLS)

import traci

class SCATSEnvironment:
    def __init__(self, sumo_cfg="train.sumocfg", gui=False, max_steps=3600, 
                 sensor_failure_prob=0.0):  # Убрали сбои при обучении
        self.sumo_cfg = sumo_cfg
        self.gui = gui
        self.max_steps = max_steps
        self.sumo_binary = "sumo-gui" if gui else "sumo"
        self.step_count = 0
        self.ts_id = None
        self.sensor_failure_prob = sensor_failure_prob
    
    def start(self):
        """Запуск SUMO"""
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
        print(f"✅ SCATS контроллер подключен к светофору: {self.ts_id}")

    def get_scats_detectors(self):
        """УПРОЩЕННОЕ состояние - только очереди и интенсивность"""
        lanes = traci.trafficlight.getControlledLanes(self.ts_id)
        
        # Агрегируем по направлениям (группы по 3 полосы)
        state = []
        
        for i in range(0, len(lanes), 3):
            direction_lanes = lanes[i:i+3]
            if not direction_lanes:
                continue
                
            total_queue = 0
            total_vehicles = 0
            
            for lane in direction_lanes:
                queue = traci.lane.getLastStepHaltingNumber(lane)
                vehicles = traci.lane.getLastStepVehicleNumber(lane)
                
                total_queue += queue
                total_vehicles += vehicles
            
            # Нормализованные метрики на направление
            avg_queue = total_queue / len(direction_lanes)
            avg_vehicles = total_vehicles / len(direction_lanes)
            
            state.extend([avg_queue, avg_vehicles])
        
        state = np.array(state, dtype=np.float32)
        
        # Сбои только при тестировании
        if random.random() < self.sensor_failure_prob:
            state = np.zeros_like(state)
        
        return state

    def calculate_scats_reward(self):
        """УПРОЩЕННАЯ награда - только очереди"""
        lanes = traci.trafficlight.getControlledLanes(self.ts_id)
        
        total_queue = 0
        for lane in lanes:
            queue = traci.lane.getLastStepHaltingNumber(lane)
            total_queue += queue
        
        # Мягкий штраф за очереди
        reward = -total_queue * 0.1
        return reward

    def set_phase_scats(self, phase_index, duration):
        """Установка фазы"""
        traci.trafficlight.setPhase(self.ts_id, phase_index)
        # Убрали setPhaseDuration - мешает работе

    def step(self, action, duration=5):  # Укоротили до 5 секунд!
        """Шаг симуляции"""
        self.set_phase_scats(action, duration)
        
        total_reward = 0
        for _ in range(duration):
            traci.simulationStep()
            self.step_count += 1
            total_reward += self.calculate_scats_reward()
        
        state = self.get_scats_detectors()
        done = self.step_count >= self.max_steps
        
        return state, total_reward, done

    def get_state(self):
        return self.get_scats_detectors()

    def close(self):
        traci.close()