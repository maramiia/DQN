# compare_baseline_vs_drqn_heavy_simple.py
import os
import sys
import csv
import matplotlib.pyplot as plt
import torch
import numpy as np

# Настройка шрифтов для публикаций
plt.rcParams.update({
    "font.size": 12,
    "font.family": "serif",
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.figsize": (8, 5)
})

SUMO_HOME = r"C:\Program Files (x86)\Eclipse\Sumo"
TOOLS = os.path.join(SUMO_HOME, "tools")
if TOOLS not in sys.path:
    sys.path.append(TOOLS)

# --- Baseline (на heavy_test.sumocfg) ---
from sumo_env import SumoEnv
import traci

def run_baseline():
    env = SumoEnv(sumo_cfg="heavy_test.sumocfg", gui=False, max_steps=7200, sensor_failure_prob=0.05)
    env.start()
    try:
        tls_logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(env.ts_id)
        phases = tls_logic[0].phases
        fixed_duration = [p.duration for p in phases]
        num_phases = len(phases)
        phase_idx = 0
        time_in_phase = 0
        total_wait = 0

        for step in range(7200):
            if time_in_phase >= fixed_duration[phase_idx]:
                phase_idx = (phase_idx + 1) % num_phases
                time_in_phase = 0
            traci.trafficlight.setPhase(env.ts_id, phase_idx)
            traci.simulationStep()
            time_in_phase += 1
            wait = sum(traci.lane.getWaitingTime(lane) for lane in traci.trafficlight.getControlledLanes(env.ts_id))
            total_wait += wait

        return total_wait / 7200
    finally:
        env.close()

# --- DRQN (на heavy_test.sumocfg) ---
from sumo_history_env import SumoHistoryEnv
from drqn_agent import DRQN

def run_drqn():
    sequence_length = 5
    env = SumoHistoryEnv(
        sumo_cfg="heavy_test.sumocfg",
        gui=False,
        max_steps=7200,
        sensor_failure_prob=0.05,
        sequence_length=sequence_length
    )
    state_seq = env.reset()
    try:
        tls_logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(env.ts_id)
        action_size = len(tls_logic[0].phases)
        device = torch.device("cpu")
        model = DRQN(len(state_seq[0]), action_size, hidden_dim=128).to(device)
        model.load_state_dict(torch.load("drqn_sumo.pth", map_location=device))
        model.eval()

        total_wait = 0
        step = 0
        while True:
            state_tensor = torch.FloatTensor(state_seq).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values, _ = model(state_tensor)
                action = q_values.argmax().item()
            next_state_seq, reward, done = env.step_with_history(action, duration=10)
            wait = sum(traci.lane.getWaitingTime(lane) for lane in traci.trafficlight.getControlledLanes(env.ts_id))
            total_wait += wait
            state_seq = next_state_seq
            step += 1
            if done:
                break
        return total_wait / (step * 10) if step > 0 else 0  # нормируем на секунды симуляции
    finally:
        env.close()

# --- Основной запуск ---
if __name__ == "__main__":
    print("🚀 Запуск сравнения Baseline vs DRQN на heavy_test.sumocfg...\n")

    # Запуск методов
    baseline = run_baseline()
    drqn = run_drqn()

    # Вывод таблицы
    print("="*60)
    print("📊 СРЕДНЕЕ ВРЕМЯ ОЖИДАНИЯ (секунды) — heavy-сценарий")
    print("="*60)
    print(f"{'Метод':<15} {'Время ожидания':<20} {'Улучшение':<15}")
    print("-"*60)
    print(f"{'Baseline':<15} {baseline:<20.2f} {'—':<15}")
    print(f"{'DRQN':<15} {drqn:<20.2f} {((baseline - drqn) / baseline * 100):<14.1f}%")
    print("="*60)

    # Сохранение в CSV
    with open("baseline_vs_drqn_heavy_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "Avg_Waiting_Time_sec", "Improvement_vs_Baseline_%"])
        writer.writerow(["Baseline", baseline, 0])
        writer.writerow(["DRQN", drqn, (baseline - drqn) / baseline * 100])

    # Построение графика
    methods = ["Baseline", "DRQN"]
    avg_wait = [baseline, drqn]
    colors = ["#ff9999", "#99ff99"]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(methods, avg_wait, color=colors, edgecolor='black', linewidth=0.8)

    # Подписи значений на столбцах
    for bar, value in zip(bars, avg_wait):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                 f'{value:.2f}', ha='center', va='bottom', fontweight='bold')

    plt.ylabel("Среднее время ожидания (сек)")
    plt.title("Baseline vs DRQN на сценарии с высокой нагрузкой")
    plt.ylim(0, max(avg_wait) * 1.2)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Сохранение и отображение
    plt.savefig("baseline_vs_drqn_heavy_plot.png", dpi=300, bbox_inches='tight')
    print("\n✅ График сохранён как 'baseline_vs_drqn_heavy_plot.png'")
    print("✅ Результаты сохранены в 'baseline_vs_drqn_heavy_results.csv'")
    
    plt.show()