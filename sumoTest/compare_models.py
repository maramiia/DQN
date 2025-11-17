# compare_all_models.py
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec

SUMO_HOME = r"C:\Program Files (x86)\Eclipse\Sumo"
TOOLS = os.path.join(SUMO_HOME, "tools")
if TOOLS not in sys.path:
    sys.path.append(TOOLS)

from sumo_env import SumoEnv
from sumo_history_env import SumoHistoryEnv
import traci

class EnhancedModelComparator:
    def __init__(self):
        self.results = {}
        self.colors = {
            'Fixed-Time': '#FF6B6B',
            'DQN': '#4ECDC4', 
            'DRQN': '#45B7D1'
        }
        
    def run_baseline_fixed(self, sumo_cfg="test.sumocfg", sensor_failure_prob=0.05):
        """Запуск Fixed-Time baseline"""
        print("🚦 Запуск Fixed-Time Baseline...")
        env = SumoEnv(sumo_cfg=sumo_cfg, gui=False, max_steps=7200, sensor_failure_prob=sensor_failure_prob)
        env.start()

        try:
            tls_logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(env.ts_id)
            phases = tls_logic[0].phases
            num_phases = len(phases)
            fixed_duration = [p.duration for p in phases]

            total_wait = 0
            wait_times = []
            queue_lengths = []
            phase_changes = []
            phase_idx = 0
            time_in_phase = 0

            for step in range(7200):
                if time_in_phase >= fixed_duration[phase_idx]:
                    phase_idx = (phase_idx + 1) % num_phases
                    time_in_phase = 0
                    phase_changes.append(step)
                
                traci.trafficlight.setPhase(env.ts_id, phase_idx)
                traci.simulationStep()
                time_in_phase += 1

                wait = sum(traci.lane.getWaitingTime(lane) for lane in traci.trafficlight.getControlledLanes(env.ts_id))
                total_wait += wait
                wait_times.append(wait)
                
                # Собираем дополнительную статистику
                queue_length = sum(traci.lane.getLastStepHaltingNumber(lane) for lane in traci.trafficlight.getControlledLanes(env.ts_id))
                queue_lengths.append(queue_length)

            avg_wait = total_wait / 7200
            std_wait = np.std(wait_times)
            max_wait = max(wait_times)
            avg_queue = np.mean(queue_lengths)
            
            self.results['Fixed-Time'] = {
                'avg_wait': avg_wait,
                'std_wait': std_wait,
                'max_wait': max_wait,
                'avg_queue': avg_queue,
                'wait_times': wait_times,
                'queue_lengths': queue_lengths,
                'phase_changes': phase_changes
            }
            
            print(f"✅ Fixed-Time: {avg_wait:.2f} ± {std_wait:.2f} сек | Очередь: {avg_queue:.1f} машин")
            
        finally:
            env.close()
            
    def run_demo_dqn(self, sumo_cfg="test.sumocfg", sensor_failure_prob=0.05):
        """Запуск DQN модели"""
        print("🧠 Запуск DQN модели...")
        env = SumoEnv(sumo_cfg=sumo_cfg, gui=False, max_steps=7200, sensor_failure_prob=sensor_failure_prob)
        env.start()

        try:
            state = env.get_state()
            state_size = len(state)
            tls_logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(env.ts_id)
            action_size = len(tls_logic[0].phases)

            device = torch.device("cpu")
            
            # Загрузка DQN модели
            from dqn_agent import DQN
            model = DQN(state_size, action_size).to(device)
            
            model_files = ["dqn_sumo_best.pth", "dqn_sumo_final.pth", "dqn_sumo.pth"]
            model_loaded = False
            
            for model_file in model_files:
                if os.path.exists(model_file):
                    try:
                        model.load_state_dict(torch.load(model_file, map_location=device))
                        print(f"✅ Загружена {model_file}")
                        model_loaded = True
                        break
                    except Exception as e:
                        print(f"⚠️ Ошибка загрузки {model_file}: {e}")
            
            if not model_loaded:
                print("❌ Не найдена обученная DQN модель!")
                return

            model.eval()

            total_wait = 0
            wait_times = []
            queue_lengths = []
            actions_taken = []
            q_values_history = []
            step = 0
            
            while step < 7200:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_values = model(state_tensor)
                    action = q_values.argmax().item()

                next_state, reward, done = env.step(action, duration=10)
                
                # Собираем статистику
                lanes = traci.trafficlight.getControlledLanes(env.ts_id)
                wait = sum(traci.lane.getWaitingTime(lane) for lane in lanes)
                total_wait += wait
                wait_times.append(wait)
                
                queue_length = sum(traci.lane.getLastStepHaltingNumber(lane) for lane in lanes)
                queue_lengths.append(queue_length)
                actions_taken.append(action)
                q_values_history.append(q_values.cpu().numpy()[0])
                
                state = next_state
                step += 1
                if done:
                    break

            avg_wait = total_wait / step if step > 0 else 0
            std_wait = np.std(wait_times)
            max_wait = max(wait_times) if wait_times else 0
            avg_queue = np.mean(queue_lengths) if queue_lengths else 0
            
            self.results['DQN'] = {
                'avg_wait': avg_wait,
                'std_wait': std_wait,
                'max_wait': max_wait,
                'avg_queue': avg_queue,
                'wait_times': wait_times,
                'queue_lengths': queue_lengths,
                'actions_taken': actions_taken,
                'q_values': q_values_history
            }
            
            print(f"✅ DQN: {avg_wait:.2f} ± {std_wait:.2f} сек | Очередь: {avg_queue:.1f} машин")

        finally:
            env.close()
            
    def run_test_drqn(self, sumo_cfg="test.sumocfg", sensor_failure_prob=0.05):
        """Запуск DRQN модели"""
        print("🧠 Запуск DRQN модели...")
        sequence_length = 5
        env = SumoHistoryEnv(
            sumo_cfg=sumo_cfg,
            gui=False,
            max_steps=7200,
            sensor_failure_prob=sensor_failure_prob,
            sequence_length=sequence_length
        )
        state_seq = env.reset()

        try:
            state_size = len(state_seq[0])
            tls_logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(env.ts_id)
            action_size = len(tls_logic[0].phases)

            device = torch.device("cpu")
            
            # Загрузка DRQN модели
            from drqn_agent import DRQN
            model = DRQN(state_size, action_size).to(device)
            
            model_files = ["drqn_sumo_best.pth", "drqn_sumo_final.pth", "drqn_sumo.pth"]
            model_loaded = False
            
            for model_file in model_files:
                if os.path.exists(model_file):
                    try:
                        model.load_state_dict(torch.load(model_file, map_location=device))
                        print(f"✅ Загружена {model_file}")
                        model_loaded = True
                        break
                    except Exception as e:
                        print(f"⚠️ Ошибка загрузки {model_file}: {e}")
            
            if not model_loaded:
                print("❌ Не найдена обученная DRQN модель!")
                return

            model.eval()

            total_wait = 0
            wait_times = []
            queue_lengths = []
            actions_taken = []
            q_values_history = []
            step = 0
            
            hidden = model.init_hidden(device=device)
            
            while step < 7200:
                state_tensor = torch.FloatTensor(state_seq).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_values, hidden = model(state_tensor, hidden)
                    action = q_values.argmax().item()

                next_state_seq, reward, done = env.step_with_history(action, duration=10)
                
                # Собираем статистику
                lanes = traci.trafficlight.getControlledLanes(env.ts_id)
                wait = sum(traci.lane.getWaitingTime(lane) for lane in lanes)
                total_wait += wait
                wait_times.append(wait)
                
                queue_length = sum(traci.lane.getLastStepHaltingNumber(lane) for lane in lanes)
                queue_lengths.append(queue_length)
                actions_taken.append(action)
                q_values_history.append(q_values.cpu().numpy()[0])
                
                state_seq = next_state_seq
                step += 1
                if done:
                    break

            avg_wait = total_wait / step if step > 0 else 0
            std_wait = np.std(wait_times)
            max_wait = max(wait_times) if wait_times else 0
            avg_queue = np.mean(queue_lengths) if queue_lengths else 0
            
            self.results['DRQN'] = {
                'avg_wait': avg_wait,
                'std_wait': std_wait,
                'max_wait': max_wait,
                'avg_queue': avg_queue,
                'wait_times': wait_times,
                'queue_lengths': queue_lengths,
                'actions_taken': actions_taken,
                'q_values': q_values_history
            }
            
            print(f"✅ DRQN: {avg_wait:.2f} ± {std_wait:.2f} сек | Очередь: {avg_queue:.1f} машин")

        finally:
            env.close()
    
    def create_comprehensive_visualization(self):
        """Создает комплексную визуализацию сравнения моделей"""
        if len(self.results) < 2:
            print("❌ Недостаточно данных для визуализации")
            return
        
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(3, 3, figure=fig)
        
        models = list(self.results.keys())
        
        # 1. Основное сравнение времени ожидания
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_main_comparison(ax1, models)
        
        # 2. Время ожидания по шагам
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_waiting_times(ax2, models)
        
        # 3. Длины очередей
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_queue_lengths(ax3, models)
        
        # 4. Распределение действий (для RL моделей)
        ax4 = fig.add_subplot(gs[1, 2])
        self._plot_action_distribution(ax4, models)
        
        # 5. Статистика улучшений
        ax5 = fig.add_subplot(gs[2, 0])
        self._plot_improvement_stats(ax5, models)
        
        # 6. Максимальное время ожидания
        ax6 = fig.add_subplot(gs[2, 1])
        self._plot_max_waiting(ax6, models)
        
        # 7. Q-values динамика (для RL моделей)
        ax7 = fig.add_subplot(gs[2, 2])
        self._plot_q_values(ax7, models)
        
        plt.tight_layout()
        plt.savefig('comprehensive_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Создаем также упрощенную версию
        self._create_simple_comparison_chart()
    
    def _plot_main_comparison(self, ax, models):
        """Основное сравнение моделей"""
        avg_waits = [self.results[model]['avg_wait'] for model in models]
        std_waits = [self.results[model]['std_wait'] for model in models]
        
        bars = ax.bar(models, avg_waits, yerr=std_waits, capsize=8, 
                     color=[self.colors.get(model, '#999999') for model in models],
                     alpha=0.8, edgecolor='black', linewidth=1.2)
        
        ax.set_title('Сравнение среднего времени ожидания', fontsize=16, fontweight='bold', pad=20)
        ax.set_ylabel('Среднее время ожидания (сек)', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Добавляем значения на столбцы
        for bar, value in zip(bars, avg_waits):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{value:.2f} сек', ha='center', va='bottom', fontweight='bold')
        
        # Добавляем улучшения относительно Fixed-Time
        if 'Fixed-Time' in self.results:
            fixed_avg = self.results['Fixed-Time']['avg_wait']
            for i, model in enumerate(models):
                if model != 'Fixed-Time':
                    improvement = ((fixed_avg - avg_waits[i]) / fixed_avg) * 100
                    color = 'green' if improvement > 0 else 'red'
                    ax.text(i, avg_waits[i] / 2, f'{improvement:+.1f}%', 
                           ha='center', va='center', fontweight='bold', color=color, fontsize=11)
    
    def _plot_waiting_times(self, ax, models):
        """График времени ожидания по шагам"""
        for model in models:
            wait_times = self.results[model]['wait_times'][:1000]  # Первые 1000 шагов
            ax.plot(wait_times, label=model, color=self.colors.get(model, '#999999'), alpha=0.7)
        
        ax.set_title('Динамика времени ожидания', fontsize=12)
        ax.set_xlabel('Шаг симуляции')
        ax.set_ylabel('Время ожидания (сек)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_queue_lengths(self, ax, models):
        """График длин очередей"""
        for model in models:
            queues = self.results[model]['queue_lengths'][:1000]
            ax.plot(queues, label=model, color=self.colors.get(model, '#999999'), alpha=0.7)
        
        ax.set_title('Длина очереди транспортных средств', fontsize=12)
        ax.set_xlabel('Шаг симуляции')
        ax.set_ylabel('Количество машин в очереди')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_action_distribution(self, ax, models):
        """Распределение действий для RL моделей"""
        rl_models = [m for m in models if m != 'Fixed-Time']
        
        if not rl_models:
            ax.text(0.5, 0.5, 'Нет данных по RL моделям', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        action_data = []
        labels = []
        for model in rl_models:
            if 'actions_taken' in self.results[model]:
                actions = self.results[model]['actions_taken']
                unique, counts = np.unique(actions, return_counts=True)
                action_data.append(counts / len(actions) * 100)
                labels.append(model)
        
        if action_data:
            x = np.arange(len(unique))
            width = 0.35
            
            for i, (data, model) in enumerate(zip(action_data, labels)):
                ax.bar(x + i*width, data, width, label=model, 
                      color=self.colors.get(model, '#999999'), alpha=0.8)
            
            ax.set_title('Распределение действий', fontsize=12)
            ax.set_xlabel('Номер действия')
            ax.set_ylabel('Процент использования (%)')
            ax.set_xticks(x + width/2)
            ax.set_xticklabels([f'Фаза {i}' for i in unique])
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    def _plot_improvement_stats(self, ax, models):
        """Статистика улучшений"""
        if 'Fixed-Time' not in self.results:
            return
            
        fixed_avg = self.results['Fixed-Time']['avg_wait']
        improvements = []
        model_names = []
        
        for model in models:
            if model != 'Fixed-Time':
                improvement = ((fixed_avg - self.results[model]['avg_wait']) / fixed_avg) * 100
                improvements.append(improvement)
                model_names.append(model)
        
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        bars = ax.bar(model_names, improvements, color=colors, alpha=0.7)
        
        ax.set_title('Улучшение относительно Fixed-Time', fontsize=12)
        ax.set_ylabel('Улучшение (%)')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, value in zip(bars, improvements):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (1 if height >=0 else -1),
                   f'{value:+.1f}%', ha='center', va='bottom' if height >=0 else 'top',
                   fontweight='bold')
    
    def _plot_max_waiting(self, ax, models):
        """Максимальное время ожидания"""
        max_waits = [self.results[model]['max_wait'] for model in models]
        
        bars = ax.bar(models, max_waits, 
                     color=[self.colors.get(model, '#999999') for model in models],
                     alpha=0.7)
        
        ax.set_title('Максимальное время ожидания', fontsize=12)
        ax.set_ylabel('Максимальное время (сек)')
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, value in zip(bars, max_waits):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                   f'{value:.0f} сек', ha='center', va='bottom')
    
    def _plot_q_values(self, ax, models):
        """Динамика Q-values для RL моделей"""
        rl_models = [m for m in models if m != 'Fixed-Time']
        
        if not rl_models:
            ax.text(0.5, 0.5, 'Нет данных по Q-values', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        for model in rl_models:
            if 'q_values' in self.results[model]:
                q_values = self.results[model]['q_values']
                if len(q_values) > 0:
                    max_q = [np.max(q) for q in q_values[:500]]  # Первые 500 шагов
                    ax.plot(max_q, label=model, color=self.colors.get(model, '#999999'), alpha=0.7)
        
        ax.set_title('Максимальные Q-values', fontsize=12)
        ax.set_xlabel('Шаг симуляции')
        ax.set_ylabel('Q-value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _create_simple_comparison_chart(self):
        """Создает упрощенную версию графика"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        models = list(self.results.keys())
        
        # 1. Основное сравнение
        avg_waits = [self.results[model]['avg_wait'] for model in models]
        bars = ax1.bar(models, avg_waits, color=[self.colors[model] for model in models])
        ax1.set_title('Среднее время ожидания')
        ax1.set_ylabel('Секунды')
        for bar, value in zip(bars, avg_waits):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{value:.2f}', ha='center', va='bottom')
        
        # 2. Улучшение относительно Fixed-Time
        if 'Fixed-Time' in self.results:
            fixed_avg = self.results['Fixed-Time']['avg_wait']
            improvements = []
            imp_models = []
            for model in models:
                if model != 'Fixed-Time':
                    improvement = ((fixed_avg - self.results[model]['avg_wait']) / fixed_avg) * 100
                    improvements.append(improvement)
                    imp_models.append(model)
            
            colors = ['green' if imp > 0 else 'red' for imp in improvements]
            bars = ax2.bar(imp_models, improvements, color=colors)
            ax2.set_title('Улучшение относительно Fixed-Time')
            ax2.set_ylabel('Процент улучшения (%)')
            ax2.axhline(0, color='black', linewidth=0.8)
            for bar, value in zip(bars, improvements):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (1 if value >=0 else -1),
                        f'{value:+.1f}%', ha='center', va='bottom' if value >=0 else 'top')
        
        # 3. Максимальное время ожидания
        max_waits = [self.results[model]['max_wait'] for model in models]
        bars = ax3.bar(models, max_waits, color=[self.colors[model] for model in models])
        ax3.set_title('Максимальное время ожидания')
        ax3.set_ylabel('Секунды')
        for bar, value in zip(bars, max_waits):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f'{value:.0f}', ha='center', va='bottom')
        
        # 4. Средняя длина очереди
        avg_queues = [self.results[model]['avg_queue'] for model in models]
        bars = ax4.bar(models, avg_queues, color=[self.colors[model] for model in models])
        ax4.set_title('Средняя длина очереди')
        ax4.set_ylabel('Количество машин')
        for bar, value in zip(bars, avg_queues):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{value:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('simple_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def print_detailed_report(self):
        """Печатает детальный отчет"""
        print("\n" + "="*80)
        print("📊 ДЕТАЛЬНЫЙ ОТЧЕТ СРАВНЕНИЯ МОДЕЛЕЙ")
        print("="*80)
        
        data = []
        for model, results in self.results.items():
            row = {
                'Модель': model,
                'Среднее время ожидания (сек)': f"{results['avg_wait']:.2f}",
                'Стандартное отклонение': f"{results['std_wait']:.2f}",
                'Максимальное время ожидания': f"{results['max_wait']:.2f}",
                'Средняя длина очереди': f"{results['avg_queue']:.1f}"
            }
            
            if model != 'Fixed-Time' and 'Fixed-Time' in self.results:
                improvement = ((self.results['Fixed-Time']['avg_wait'] - results['avg_wait']) / 
                             self.results['Fixed-Time']['avg_wait'] * 100)
                row['Улучшение'] = f"{improvement:+.1f}%"
            else:
                row['Улучшение'] = '-'
            
            data.append(row)
        
        df = pd.DataFrame(data)
        print(df.to_string(index=False))
        print("="*80)

def main():
    """Основная функция сравнения"""
    comparator = EnhancedModelComparator()
    
    print("🎯 КОМПЛЕКСНОЕ СРАВНЕНИЕ МОДЕЛЕЙ УПРАВЛЕНИЯ СВЕТОФОРАМИ")
    print("="*60)
    
    # Запускаем все три модели
    comparator.run_baseline_fixed()
    print("-" * 50)
    
    comparator.run_demo_dqn() 
    print("-" * 50)
    
    comparator.run_test_drqn()
    print("-" * 50)
    
    # Создаем визуализации
    comparator.create_comprehensive_visualization()
    
    # Печатаем отчет
    comparator.print_detailed_report()
    
    print("\n✅ Сравнение завершено! Результаты сохранены в:")
    print("   - comprehensive_model_comparison.png")
    print("   - simple_model_comparison.png")

if __name__ == "__main__":
    main()