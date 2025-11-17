# baseline_heavy.py
import os
import sys

SUMO_HOME = r"C:\Program Files (x86)\Eclipse\Sumo"
TOOLS = os.path.join(SUMO_HOME, "tools")
if TOOLS not in sys.path:
    sys.path.append(TOOLS)

from sumo_env import SumoEnv
import traci

def main():
    print("🚦 Запуск BASELINE (фиксированный цикл) на СЦЕНАРИИ С ВЫСОКОЙ НАГРУЗКОЙ (heavy_test.sumocfg)...")
    
    env = SumoEnv(
        sumo_cfg="heavy_test.sumocfg",     # ← главное отличие
        gui=False,
        max_steps=7200,
        sensor_failure_prob=0.05           # baseline его не использует, но среда может
    )
    env.start()
    
    try:
        # Получаем фиксированный цикл из SUMO (как задан в net.xml)
        tls_logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(env.ts_id)
        phases = tls_logic[0].phases
        fixed_duration = [p.duration for p in phases]
        num_phases = len(phases)
        
        print(f"🔄 Найдено фаз: {num_phases}")
        for i, (ph, dur) in enumerate(zip(phases, fixed_duration)):
            print(f"   Фаза {i}: длительность = {dur} сек, состояние = {ph.state}")

        phase_idx = 0
        time_in_phase = 0
        total_wait = 0
        step = 0

        # Основной цикл — 1 шаг = 1 сек (как в оригинале)
        for step in range(7200):
            # Переключение фазы при достижении длительности
            if time_in_phase >= fixed_duration[phase_idx]:
                phase_idx = (phase_idx + 1) % num_phases
                time_in_phase = 0

            traci.trafficlight.setPhase(env.ts_id, phase_idx)
            traci.simulationStep()
            time_in_phase += 1

            # Считаем суммарное ожидание на всех полосах светофора
            wait = sum(traci.lane.getWaitingTime(lane) for lane in traci.trafficlight.getControlledLanes(env.ts_id))
            total_wait += wait

            # Лог каждые 1000 секунд
            if (step + 1) % 1000 == 0:
                avg_wait_so_far = total_wait / (step + 1)
                vehicles_now = traci.vehicle.getIDCount()
                print(f"⏱️ {step + 1} сек | Фаза {phase_idx} | Ср. ожидание: {avg_wait_so_far:.2f} сек | Машин в сети: {vehicles_now}")

        # Финальный расчёт
        avg_wait = total_wait / 7200

        print("\n" + "="*60)
        print("📊 РЕЗУЛЬТАТЫ BASELINE НА HEAVY-СЦЕНАРИИ:")
        print(f"⏱️ Продолжительность: 7200 сек (2 ч)")
        print(f"⏳ Суммарное ожидание: {total_wait:.0f} сек")
        print(f"📊 Среднее время ожидания: {avg_wait:.2f} сек/автомобиль/секунда")
        print("="*60)

        # Сохраняем результат
        with open("baseline_heavy_result.txt", "w", encoding="utf-8") as f:
            f.write(f"Среднее время ожидания (baseline, heavy_test): {avg_wait:.2f} сек\n")
            f.write(f"Суммарное ожидание: {total_wait:.0f} сек\n")
            f.write(f"Продолжительность: 7200 сек\n")

    finally:
        env.close()
        print("🔚 Симуляция завершена.")

if __name__ == "__main__":
    main()