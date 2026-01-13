#!/usr/bin/env python3
"""
Скрипт для построения графиков производительности CPU vs CUDA
на основе результатов бенчмарков клеточного автомата
"""

import csv
import matplotlib.pyplot as plt

# Настройка стиля графиков
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (16, 6)
plt.rcParams['font.size'] = 11

# Загрузка данных из CSV
data = []
with open('benchmark_results.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append({
            'num_nodes': int(row['num_nodes']),
            'num_edges': int(row['num_edges']),
            'avg_degree': float(row['avg_degree']),
            'steps': int(row['steps']),
            'feedback_type': int(row['feedback_type']),
            'feedback_name': row['feedback_name'],
            'cpu_time_ms': float(row['cpu_time_ms']),
            'cuda_time_ms': float(row['cuda_time_ms']),
            'speedup': float(row['speedup']),
            'results_match': row['results_match'] == 'true'
        })

print(f"Загружено {len(data)} записей")

# Фильтрация данных
def filter_data(data, feedback_type, steps=100, max_nodes=1000000):
    """Фильтрует данные по типу функции и количеству шагов"""
    return sorted(
        [row for row in data 
         if row['feedback_type'] == feedback_type 
         and row['steps'] == steps 
         and row['num_nodes'] <= max_nodes],
        key=lambda x: x['num_nodes']
    )

# Фильтруем данные для двух функций
xor_data = filter_data(data, feedback_type=0)
maj_data = filter_data(data, feedback_type=1)

# Создаем два подграфика
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Сравнение производительности CPU vs CUDA\nОбобщенный клеточный автомат', 
             fontsize=16, fontweight='bold')

# ============================================================================
# График 1: Функция XOR
# ============================================================================
ax1 = axes[0]

# Извлекаем данные для графика
xor_nodes = [row['num_nodes'] for row in xor_data]
xor_cpu_times = [row['cpu_time_ms'] for row in xor_data]
xor_cuda_times = [row['cuda_time_ms'] for row in xor_data]

# Рисуем линии
ax1.plot(xor_nodes, xor_cpu_times, 
         'o-', color='#2E86AB', linewidth=1.5, markersize=6, label='CPU', 
         markeredgewidth=0.5, markeredgecolor='black')
ax1.plot(xor_nodes, xor_cuda_times, 
         's-', color='#A23B72', linewidth=1.5, markersize=6, label='CUDA', 
         markeredgewidth=0.5, markeredgecolor='black')

ax1.set_xlabel('Количество узлов графа', fontsize=13, fontweight='bold')
ax1.set_ylabel('Время выполнения (мс)', fontsize=13, fontweight='bold')
ax1.set_title('Функция обратной связи: XOR\n(100 шагов эволюции)', 
              fontsize=14, fontweight='bold')
ax1.legend(fontsize=12, loc='upper left', framealpha=0.9)
ax1.grid(True, alpha=0.3, linestyle='--')

# ============================================================================
# График 2: Функция MAJORITY
# ============================================================================
ax2 = axes[1]

# Извлекаем данные для графика
maj_nodes = [row['num_nodes'] for row in maj_data]
maj_cpu_times = [row['cpu_time_ms'] for row in maj_data]
maj_cuda_times = [row['cuda_time_ms'] for row in maj_data]

# Рисуем линии
ax2.plot(maj_nodes, maj_cpu_times, 
         'o-', color='#2E86AB', linewidth=1.5, markersize=6, label='CPU', 
         markeredgewidth=0.5, markeredgecolor='black')
ax2.plot(maj_nodes, maj_cuda_times, 
         's-', color='#A23B72', linewidth=1.5, markersize=6, label='CUDA', 
         markeredgewidth=0.5, markeredgecolor='black')

ax2.set_xlabel('Количество узлов графа', fontsize=13, fontweight='bold')
ax2.set_ylabel('Время выполнения (мс)', fontsize=13, fontweight='bold')
ax2.set_title('Функция обратной связи: MAJORITY\n(100 шагов эволюции)', 
              fontsize=14, fontweight='bold')
ax2.legend(fontsize=12, loc='upper left', framealpha=0.9)
ax2.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('benchmark_plots.png', dpi=300, bbox_inches='tight')
print("\n✓ Графики сохранены в файл: benchmark_plots.png")

# ============================================================================
# Статистика
# ============================================================================
print("\n" + "="*70)
print("СТАТИСТИКА ПРОИЗВОДИТЕЛЬНОСТИ")
print("="*70)

total_tests = len(data)
cuda_faster = len([row for row in data if row['speedup'] > 1])
cpu_faster = len([row for row in data if row['speedup'] < 1])

print(f"\nВсего тестов: {total_tests}")
print(f"CUDA быстрее: {cuda_faster} ({100*cuda_faster/total_tests:.1f}%)")
print(f"CPU быстрее:  {cpu_faster} ({100*cpu_faster/total_tests:.1f}%)")

max_speedup = max(row['speedup'] for row in data)
max_speedup_row = [row for row in data if row['speedup'] == max_speedup][0]
print(f"\nМаксимальное ускорение CUDA: {max_speedup:.2f}x")
print(f"  Граф: {max_speedup_row['num_nodes']} узлов, "
      f"{max_speedup_row['steps']} шагов, {max_speedup_row['feedback_name']}")

threshold_data = [row for row in data if row['speedup'] > 1 and row['steps'] == 100]
if threshold_data:
    threshold = min(row['num_nodes'] for row in threshold_data)
    print(f"\nПороговый размер графа (когда CUDA становится быстрее):")
    print(f"  ~{threshold} узлов (при 100 шагах)")

print("\n" + "="*70)

# Сравнение XOR и MAJORITY
print("\nСРАВНЕНИЕ ФУНКЦИЙ (при 100 шагах):")
print("-" * 70)
print(f"{'Размер':<15} {'XOR CPU':<12} {'XOR CUDA':<12} {'MAJ CPU':<12} {'MAJ CUDA':<12}")
print("-" * 70)

for size in [100, 1000, 10000, 50000, 100000, 200000, 500000]:
    xor_row = [row for row in xor_data if row['num_nodes'] == size]
    maj_row = [row for row in maj_data if row['num_nodes'] == size]
    
    if xor_row and maj_row:
        xor_cpu = xor_row[0]['cpu_time_ms']
        xor_cuda = xor_row[0]['cuda_time_ms']
        maj_cpu = maj_row[0]['cpu_time_ms']
        maj_cuda = maj_row[0]['cuda_time_ms']
        
        size_str = f"{size//1000}K узлов" if size >= 1000 else f"{size} узлов"
        print(f"{size_str:<15} {xor_cpu:>10.3f}мс {xor_cuda:>10.3f}мс {maj_cpu:>10.3f}мс {maj_cuda:>10.3f}мс")

print("\n✓ Графики успешно построены!\n")

plt.show()
