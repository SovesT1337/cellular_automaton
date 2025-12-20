/**
 * ============================================================================
 * БЕНЧМАРК: CPU vs CUDA для клеточных автоматов
 * ============================================================================
 * 
 * Программа сравнивает производительность CPU и GPU реализаций
 * обобщенного клеточного автомата на различных топологиях:
 * - Случайные графы
 * - 1D линии (классические КА)
 * - 2D сетки (Game of Life и подобные)
 * - 3D решетки
 * - N-мерные гиперрешетки
 */

#include "cellular_automaton.h"
#include <iostream>
#include <iomanip>
#include <random>
#include <cstring>

/**
 * Вывод состояния автомата (первые N ячеек)
 */
void print_state(const std::vector<uint8_t>& state, int max_display = 20) {
    int n = std::min((int)state.size(), max_display);
    for (int i = 0; i < n; i++) {
        std::cout << (int)state[i];
    }
    if ((int)state.size() > max_display) {
        std::cout << "...";
    }
    std::cout << "\n";
}

/**
 * Вывод размерности решетки в читаемом формате
 */
std::string dim_string(const std::vector<int>& dims) {
    std::string s;
    for (size_t i = 0; i < dims.size(); i++) {
        if (i > 0) s += "x";
        s += std::to_string(dims[i]);
    }
    return s;
}

/**
 * Бенчмарк на случайном графе
 */
void run_benchmark(const char* name, int num_nodes, int avg_degree, int steps, int feedback_type) {
    std::cout << "\n=== " << name << " ===\n";
    std::cout << "Nodes: " << num_nodes << ", Avg degree: " << avg_degree << ", Steps: " << steps << "\n";
    std::cout << "Feedback: " << (feedback_type == 0 ? "XOR" : "Majority") << "\n\n";
    
    Graph graph = generate_random_graph(num_nodes, avg_degree, 42);
    std::cout << "Graph: " << graph.num_nodes << " nodes, " << graph.num_edges << " edges\n";
    
    // Случайное начальное состояние
    std::vector<uint8_t> initial(num_nodes);
    std::mt19937 rng(123);
    for (int i = 0; i < num_nodes; i++) {
        initial[i] = rng() % 2;
    }
    
    std::cout << "Initial: ";
    print_state(initial);
    
    // CPU
    std::cout << "\nCPU computation...\n";
    ComputeResult cpu_result = compute_cpu(graph, initial, steps, feedback_type);
    std::cout << "CPU time: " << std::fixed << std::setprecision(3) << cpu_result.elapsed_ms << " ms\n";
    std::cout << "Final:   ";
    print_state(cpu_result.final_state);
    
    // CUDA
    std::cout << "\nCUDA computation...\n";
    ComputeResult cuda_result = compute_cuda(graph, initial, steps, feedback_type);
    std::cout << "CUDA time: " << std::fixed << std::setprecision(3) << cuda_result.elapsed_ms << " ms\n";
    std::cout << "Final:   ";
    print_state(cuda_result.final_state);
    
    // Проверка корректности
    bool match = (cpu_result.final_state == cuda_result.final_state);
    std::cout << "\nResults match: " << (match ? "YES" : "NO") << "\n";
    
    // Ускорение
    double speedup = cpu_result.elapsed_ms / cuda_result.elapsed_ms;
    std::cout << "Speedup (CPU/CUDA): " << std::fixed << std::setprecision(2) << speedup << "x\n";
}

/**
 * Бенчмарк на N-мерной решетке
 * 
 * @param dimensions  Размеры по каждому измерению
 * @param periodic    Периодические граничные условия
 * @param neighborhood 0=фон Нейман, 1=Мур
 * @param steps       Количество шагов симуляции
 * @param feedback    Тип функции обратной связи
 */
void run_nd_benchmark(const std::vector<int>& dimensions, bool periodic, 
                      int neighborhood, int steps, int feedback) {
    NDGridConfig config;
    config.dimensions = dimensions;
    config.periodic = periodic;
    config.neighborhood = neighborhood;
    
    int ndim = config.ndim();
    int total = config.total_cells();
    
    std::cout << "\n=== " << ndim << "D Grid: " << dim_string(dimensions) << " ===\n";
    std::cout << "Total cells: " << total << "\n";
    std::cout << "Periodic: " << (periodic ? "Yes" : "No") << "\n";
    std::cout << "Neighborhood: " << (neighborhood == 0 ? "von Neumann" : "Moore") << "\n";
    std::cout << "Steps: " << steps << "\n\n";
    
    Graph graph = generate_nd_grid(config);
    std::cout << "Graph: " << graph.num_nodes << " nodes, " << graph.num_edges << " edges\n";
    std::cout << "Avg degree: " << std::fixed << std::setprecision(1) 
              << (double)graph.num_edges / graph.num_nodes << "\n";
    
    // Начальное состояние: случайное
    std::vector<uint8_t> initial(total);
    std::mt19937 rng(456);
    for (int i = 0; i < total; i++) {
        initial[i] = rng() % 2;
    }
    
    // CPU
    ComputeResult cpu_result = compute_cpu(graph, initial, steps, feedback);
    std::cout << "CPU time:  " << std::fixed << std::setprecision(3) << cpu_result.elapsed_ms << " ms\n";
    
    // CUDA
    ComputeResult cuda_result = compute_cuda(graph, initial, steps, feedback);
    std::cout << "CUDA time: " << std::fixed << std::setprecision(3) << cuda_result.elapsed_ms << " ms\n";
    
    // Метрики
    bool match = (cpu_result.final_state == cuda_result.final_state);
    double speedup = cpu_result.elapsed_ms / cuda_result.elapsed_ms;
    double throughput_cpu = (double)total * steps / (cpu_result.elapsed_ms / 1000.0);
    double throughput_cuda = (double)total * steps / (cuda_result.elapsed_ms / 1000.0);
    
    std::cout << "Match: " << (match ? "YES" : "NO") << "\n";
    std::cout << "Speedup: " << std::fixed << std::setprecision(2) << speedup << "x\n";
    std::cout << "Throughput CPU:  " << std::scientific << std::setprecision(2) << throughput_cpu << " cells/s\n";
    std::cout << "Throughput CUDA: " << std::scientific << std::setprecision(2) << throughput_cuda << " cells/s\n";
}

int main(int argc, char** argv) {
    std::cout << "============================================================\n";
    std::cout << "  Generalized Cellular Automaton - CPU vs CUDA Benchmark\n";
    std::cout << "============================================================\n";
    
    // ========================================================================
    // ТЕСТЫ НА СЛУЧАЙНЫХ ГРАФАХ
    // ========================================================================
    std::cout << "\n>>> RANDOM GRAPH TESTS <<<\n";
    
    run_benchmark("Small (1K nodes)", 1000, 4, 100, 0);
    run_benchmark("Medium (10K nodes)", 10000, 6, 100, 0);
    run_benchmark("Large (100K nodes)", 100000, 8, 50, 0);
    run_benchmark("XLarge (1M nodes)", 1000000, 4, 20, 0);
    
    // ========================================================================
    // ТЕСТЫ НА N-МЕРНЫХ РЕШЕТКАХ
    // ========================================================================
    std::cout << "\n\n>>> N-DIMENSIONAL GRID TESTS <<<\n";
    
    // 1D: Линейный автомат (как Rule 110)
    run_nd_benchmark({10000}, true, 0, 1000, 0);
    
    // 2D: Классическая сетка (как Game of Life)
    run_nd_benchmark({500, 500}, false, 0, 100, 1);    // фон Нейман
    run_nd_benchmark({500, 500}, true, 1, 100, 1);     // Мур, периодический
    
    // 3D: Объемный автомат
    run_nd_benchmark({100, 100, 100}, false, 0, 50, 0);
    run_nd_benchmark({50, 50, 50}, true, 1, 50, 1);    // Мур 3D (26 соседей!)
    
    // 4D: Гиперкуб
    run_nd_benchmark({30, 30, 30, 30}, false, 0, 20, 0);
    
    // 5D: Пятимерная решетка
    run_nd_benchmark({10, 10, 10, 10, 10}, false, 0, 10, 0);
    
    // ========================================================================
    // СВОДКА
    // ========================================================================
    std::cout << "\n============================================================\n";
    std::cout << "  SUMMARY\n";
    std::cout << "============================================================\n";
    std::cout << "- CUDA shows significant speedup for large grids (>10K cells)\n";
    std::cout << "- GPU overhead affects small grids negatively\n";
    std::cout << "- Moore neighborhood has more edges (slower but richer dynamics)\n";
    std::cout << "- Higher dimensions increase neighbor count exponentially\n";
    std::cout << "  (von Neumann: 2N neighbors, Moore: 3^N - 1 neighbors)\n";
    
    return 0;
}
