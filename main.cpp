/**
 * Обобщенный клеточный автомат
 * Вычисление выходной последовательности по заданному графу и функции обратной связи
 */

#include "cellular_automaton.h"
#include <iostream>
#include <iomanip>
#include <random>
#include <vector>
#include <fstream>

// Генерация случайного начального состояния
std::vector<uint8_t> random_state(int n, unsigned int seed = 123) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, 1);
    std::vector<uint8_t> state(n);
    for (int i = 0; i < n; i++) {
        state[i] = dist(rng);
    }
    return state;
}

// Структура для хранения результатов бенчмарка
struct BenchmarkResult {
    int num_nodes;
    int num_edges;
    int steps;
    int feedback_type;
    const char* feedback_name;
    double cpu_time_ms;
    double cuda_time_ms;
    double speedup;
    bool results_match;
};

// Функция для тестирования на графе определенного размера
BenchmarkResult benchmark_graph_size(int num_nodes, int avg_degree, int steps, 
                                      int feedback_type, const char* feedback_name) {
    BenchmarkResult result;
    result.num_nodes = num_nodes;
    result.steps = steps;
    result.feedback_type = feedback_type;
    result.feedback_name = feedback_name;
    
    // Генерация графа
    Graph graph = generate_random_graph(num_nodes, avg_degree, 42);
    result.num_edges = graph.num_edges;
    
    // Начальное состояние
    auto initial = random_state(num_nodes);
    
    // Конфигурация выходной последовательности
    OutputConfig output_cfg;
    output_cfg.cells = {0, 1, 2, 3, 4};
    output_cfg.extract_every_n_steps = std::max(1, steps / 10);
    
    // Вычисление на CPU
    auto result_cpu = compute_cpu(graph, initial, steps, feedback_type, &output_cfg);
    result.cpu_time_ms = result_cpu.elapsed_ms;
    
    #ifndef NO_CUDA
    // Вычисление на CUDA
    auto result_cuda = compute_cuda(graph, initial, steps, feedback_type, &output_cfg);
    result.cuda_time_ms = result_cuda.elapsed_ms;
    
    // Проверка корректности
    result.results_match = (result_cpu.output_sequence == result_cuda.output_sequence);
    result.speedup = result.cpu_time_ms / result.cuda_time_ms;
    #else
    result.cuda_time_ms = 0.0;
    result.speedup = 0.0;
    result.results_match = false;
    #endif
    
    return result;
}

// Сохранение результатов в CSV файл для построения графиков
void save_results_csv(const std::vector<BenchmarkResult>& results, const char* filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Ошибка открытия файла " << filename << "\n";
        return;
    }
    
    // Заголовок CSV
    file << "num_nodes,num_edges,avg_degree,steps,feedback_type,feedback_name,"
         << "cpu_time_ms,cuda_time_ms,speedup,results_match\n";
    
    // Данные
    for (const auto& r : results) {
        double avg_degree = (double)r.num_edges / r.num_nodes;
        file << r.num_nodes << ","
             << r.num_edges << ","
             << std::fixed << std::setprecision(2) << avg_degree << ","
             << r.steps << ","
             << r.feedback_type << ","
             << r.feedback_name << ","
             << std::fixed << std::setprecision(6) << r.cpu_time_ms << ","
             << r.cuda_time_ms << ","
             << r.speedup << ","
             << (r.results_match ? "true" : "false") << "\n";
    }
    
    file.close();
}

int main() {
    std::vector<BenchmarkResult> all_results;
    
    try {
        // Параметры
        int avg_degree = 5;
        int steps = 100;
        
        // Очень малые графы (50 - 500 узлов)
        std::vector<int> tiny_sizes = {50, 75, 100, 150, 200, 250, 300, 400, 500};
        for (int size : tiny_sizes) {
            auto result = benchmark_graph_size(size, avg_degree, steps, 0, "XOR");
            all_results.push_back(result);
        }
        
        // Малые графы (600 - 5K узлов) - промежуточные точки
        std::vector<int> small_sizes = {600, 800, 1000, 1500, 2000, 2500, 3000, 4000, 5000};
        for (int size : small_sizes) {
            auto result = benchmark_graph_size(size, avg_degree, steps, 0, "XOR");
            all_results.push_back(result);
        }
        
        // Средние графы (6K - 20K узлов)
        std::vector<int> medium_sizes = {6000, 7000, 8000, 9000, 10000, 12000, 15000, 18000, 20000};
        for (int size : medium_sizes) {
            auto result = benchmark_graph_size(size, avg_degree, steps, 0, "XOR");
            all_results.push_back(result);
        }
        
        // Большие графы (25K - 100K узлов)
        std::vector<int> large_sizes = {25000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000};
        for (int size : large_sizes) {
            auto result = benchmark_graph_size(size, avg_degree, steps, 0, "XOR");
            all_results.push_back(result);
        }
        
        // Очень большие графы (120K - 300K узлов)
        std::vector<int> xlarge_sizes = {120000, 150000, 200000, 250000, 300000};
        for (int size : xlarge_sizes) {
            auto result = benchmark_graph_size(size, avg_degree, steps, 0, "XOR");
            all_results.push_back(result);
        }
        
        // Тестирование с Majority функцией
        std::vector<int> majority_sizes = {100, 200, 500, 750, 1000, 1500, 2000, 3000, 5000, 
                                           7000, 10000, 15000, 20000, 30000, 50000, 70000, 
                                           100000, 150000, 200000, 300000};
        for (int size : majority_sizes) {
            auto result = benchmark_graph_size(size, avg_degree, steps, 1, "MAJORITY");
            all_results.push_back(result);
        }
        
        // Сохранение результатов в CSV
        save_results_csv(all_results, "benchmark_results.csv");
        
    } catch (const std::exception& e) {
        std::cerr << "\n✗ Ошибка: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}
