# run_dqn_demo.py
import os
import sys
import warnings

# Игнорируем ворнинг от traci (опционально)
warnings.filterwarnings("ignore", category=UserWarning, module="traci")

SUMO_HOME = r"C:\Program Files (x86)\Eclipse\Sumo"
TOOLS = os.path.join(SUMO_HOME, "tools")
if TOOLS not in sys.path:
    sys.path.append(TOOLS)

import traci
from sumo_env import SumoEnv

def main():
    env = SumoEnv(gui=True, max_steps=3600)  # 1 час симуляции
    env.start()

    try:
        state = env.get_state()
        print(f"🚦 Начальное состояние (очереди): {state}")
        
        tls_logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(env.ts_id)
        num_phases = len(tls_logic[0].phases)
        print(f"🔄 Количество фаз: {num_phases}")

        # 🔥 Заменяем цикл на бесконечный (до завершения симуляции)
        step = 0
        while True:
            action = step % num_phases
            state, reward, done = env.step(action, duration=30)
            print(f"⏱️ Шаг {step}: фаза={action}, награда={reward:.1f}, очереди={state}")
            if done:
                break
            step += 1

    finally:
        env.close()
        print("✅ Симуляция завершена.")

if __name__ == "__main__":
    main()