# compare_scats_vs_dqn.py
import os
import sys
import torch

SUMO_HOME = r"C:\Program Files (x86)\Eclipse\Sumo"
TOOLS = os.path.join(SUMO_HOME, "tools")
if TOOLS not in sys.path:
    sys.path.append(TOOLS)

from scats_env import SCATSEnvironment
from scats_agent import SCATSController
from sumo_env import SumoEnv
from dqn_agent import DQN
import traci

def run_scats():
    """Запуск SCATS контроллера"""
    env = SCATSEnvironment(sumo_cfg="test.sumocfg", gui=False, max_steps=1800, sensor_failure_prob=0.05)
    env.start()
    
    try:
        state = env.get_state()
        state_size = len(state)
        tls_logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(env.ts_id)
        action_size = len(tls_logic[0].phases)
        
        agent = SCATSController(state_size=state_size, action_size=action_size)
        agent.load("scats_model.pth")
        agent.epsilon = 0.01
        
        total_waiting_time = 0
        step = 0
        
        while True:
            action = agent.act(state)
            next_state, reward, done = env.step(action, duration=10)
            
            # Рассчитываем реальное время ожидания
            lanes = traci.trafficlight.getControlledLanes(env.ts_id)
            step_waiting = sum(traci.lane.getWaitingTime(lane) for lane in lanes)
            total_waiting_time += step_waiting
            
            state = next_state
            step += 1
            if done:
                break
        
        return total_waiting_time / step if step > 0 else 0
    
    finally:
        env.close()

def run_dqn():
    """Запуск DQN контроллера"""
    env = SumoEnv(sumo_cfg="test.sumocfg", gui=False, max_steps=1800, sensor_failure_prob=0.05)
    env.start()
    
    try:
        state = env.get_state()
        state_size = len(state)
        tls_logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(env.ts_id)
        action_size = len(tls_logic[0].phases)
        
        device = torch.device("cpu")
        model = DQN(state_size, action_size).to(device)
        model.load_state_dict(torch.load("dqn_sumo.pth", map_location=device))
        model.eval()
        
        total_waiting_time = 0
        step = 0
        
        while True:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = model(state_tensor)
                action = q_values.argmax().item()
            
            next_state, reward, done = env.step(action, duration=10)
            
            lanes = traci.trafficlight.getControlledLanes(env.ts_id)
            step_waiting = sum(traci.lane.getWaitingTime(lane) for lane in lanes)
            total_waiting_time += step_waiting
            
            state = next_state
            step += 1
            if done:
                break
        
        return total_waiting_time / step if step > 0 else 0
    
    finally:
        env.close()

if __name__ == "__main__":
    print("🔍 Сравнение SCATS vs DQN...")
    
    scats_avg_wait = run_scats()
    dqn_avg_wait = run_dqn()
    
    print(f"\n📊 РЕЗУЛЬТАТЫ СРАВНЕНИЯ:")
    print(f"   SCATS контроллер: {scats_avg_wait:.2f} сек")
    print(f"   DQN контроллер: {dqn_avg_wait:.2f} сек")
    
    if scats_avg_wait < dqn_avg_wait:
        improvement = ((dqn_avg_wait - scats_avg_wait) / dqn_avg_wait) * 100
        print(f"   ✅ SCATS лучше на {improvement:.1f}%")
    else:
        improvement = ((scats_avg_wait - dqn_avg_wait) / scats_avg_wait) * 100
        print(f"   ✅ DQN лучше на {improvement:.1f}%")