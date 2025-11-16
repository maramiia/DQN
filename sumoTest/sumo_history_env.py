# sumo_history_env.py
import numpy as np
from sumo_env import SumoEnv

class SumoHistoryEnv(SumoEnv):
    def __init__(self, sumo_cfg="train.sumocfg", gui=False, max_steps=3600,
                 sensor_failure_prob=0.05, sequence_length=5):
        super().__init__(sumo_cfg, gui, max_steps, sensor_failure_prob)
        self.seq_len = sequence_length
        self.state_history = None

    def reset(self):
        """Сбрасывает историю при старте новой симуляции"""
        self.start()
        raw_state = self.get_state()
        self.state_history = [raw_state.copy() for _ in range(self.seq_len)]
        return np.array(self.state_history, dtype=np.float32)

    def step_with_history(self, action, duration=5):
        """Выполняет шаг и возвращает последовательность последних состояний"""
        _, reward, done = super().step(action, duration)
        new_state = self.get_state()
        # Обновляем историю: удаляем первое, добавляем новое
        self.state_history = self.state_history[1:] + [new_state]
        return np.array(self.state_history), reward, done