# traffic_analyzer.py
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

SUMO_HOME = r"C:\Program Files (x86)\Eclipse\Sumo"
TOOLS = os.path.join(SUMO_HOME, "tools")
if TOOLS not in sys.path:
    sys.path.append(TOOLS)

import traci
from sumo_env import SumoEnv

def analyze_traffic_density(sumo_cfg="test.sumocfg"):
    """Анализирует плотность трафика в симуляции"""
    env = SumoEnv(sumo_cfg=sumo_cfg, gui=False, max_steps=7200)
    env.start()
    
    try:
        # Собираем статистику
        time_steps = []
        vehicle_counts = []
        waiting_times = []
        avg_speeds = []
        
        step = 0
        while True:
            traci.simulationStep()
            step += 1
            
            # Количество транспортных средств
            vehicle_count = traci.vehicle.getIDCount()
            
            # Время ожидания
            lanes = traci.lane.getIDList()
            total_waiting = sum(traci.lane.getWaitingTime(lane) for lane in lanes)
            
            # Средняя скорость
            vehicle_ids = traci.vehicle.getIDList()
            if vehicle_ids:
                avg_speed = np.mean([traci.vehicle.getSpeed(veh) for veh in vehicle_ids])
            else:
                avg_speed = 0
            
            time_steps.append(step)
            vehicle_counts.append(vehicle_count)
            waiting_times.append(total_waiting)
            avg_speeds.append(avg_speed)
            
            if step >= 7200:
                break
        
        # Визуализация
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # График количества транспортных средств
        ax1.plot(time_steps, vehicle_counts, 'b-', alpha=0.7)
        ax1.set_title('Количество транспортных средств в сети')
        ax1.set_xlabel('Время (сек)')
        ax1.set_ylabel('Количество машин')
        ax1.grid(True)
        
        # График времени ожидания
        ax2.plot(time_steps, waiting_times, 'r-', alpha=0.7)
        ax2.set_title('Общее время ожидания')
        ax2.set_xlabel('Время (сек)')
        ax2.set_ylabel('Суммарное время ожидания (сек)')
        ax2.grid(True)
        
        # График средней скорости
        ax3.plot(time_steps, avg_speeds, 'g-', alpha=0.7)
        ax3.set_title('Средняя скорость транспортных средств')
        ax3.set_xlabel('Время (сек)')
        ax3.set_ylabel('Средняя скорость (м/с)')
        ax3.grid(True)
        
        # Гистограмма распределения времени ожидания
        ax4.hist(waiting_times, bins=50, alpha=0.7, color='orange', edgecolor='black')
        ax4.set_title('Распределение времени ожидания')
        ax4.set_xlabel('Время ожидания (сек)')
        ax4.set_ylabel('Частота')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig('traffic_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Статистика
        print("\n📊 СТАТИСТИКА ТРАФИКА:")
        print(f"Максимальное количество машин: {max(vehicle_counts)}")
        print(f"Среднее количество машин: {np.mean(vehicle_counts):.2f}")
        print(f"Максимальное время ожидания: {max(waiting_times):.2f} сек")
        print(f"Среднее время ожидания: {np.mean(waiting_times):.2f} сек")
        print(f"Средняя скорость: {np.mean(avg_speeds):.2f} м/с")
        
        # Анализ пиковых периодов
        peak_indices = np.where(np.array(vehicle_counts) > np.mean(vehicle_counts) + np.std(vehicle_counts))[0]
        if len(peak_indices) > 0:
            print(f"Пиковые периоды обнаружены: {len(peak_indices)} временных интервалов")
        
    finally:
        env.close()

def quick_traffic_check():
    """Быстрая проверка плотности трафика"""
    env = SumoEnv(sumo_cfg="test.sumocfg", gui=False, max_steps=100)
    env.start()
    
    try:
        max_vehicles = 0
        for step in range(100):
            traci.simulationStep()
            vehicle_count = traci.vehicle.getIDCount()
            max_vehicles = max(max_vehicles, vehicle_count)
            
            if step % 20 == 0:
                print(f"Шаг {step}: {vehicle_count} машин в сети")
        
        print(f"\n🔍 БЫСТРАЯ ПРОВЕРКА:")
        print(f"Максимальное количество машин за 100 шагов: {max_vehicles}")
        
        if max_vehicles < 10:
            print("⚠️  ВНИМАНИЕ: Слишком мало трафика! Увеличьте количество транспортных средств.")
        elif max_vehicles < 30:
            print("ℹ️  Трафик умеренный. Можно увеличить для более сложного сценария.")
        else:
            print("✅ Трафик достаточный для тестирования.")
            
    finally:
        env.close()

if __name__ == "__main__":
    print("1. Быстрая проверка трафика...")
    quick_traffic_check()
    
    print("\n2. Полный анализ трафика...")
    analyze_traffic_density()