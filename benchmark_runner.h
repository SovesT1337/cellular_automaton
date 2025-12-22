/**
 * ============================================================================
 * ЗАПУСК БЕНЧМАРКОВ
 * ============================================================================
 */

#ifndef BENCHMARK_RUNNER_H
#define BENCHMARK_RUNNER_H

#include "benchmark_utils.h"
#include <random>

class BenchmarkRunner {
private:
    std::mt19937 rng;
    std::vector<BenchmarkData> results;
    
public:
    BenchmarkRunner(unsigned int seed = 123) : rng(seed) {}
    
    /**
     * Запуск бенчмарка для случайного графа
     */
    void run_random_graph_test(const TestConfig& config) {
        print_test_header(config.name);
        
        // Генерация графа
        Graph graph = generate_random_graph(config.nodes, config.degree, 42);
        std::vector<uint8_t> initial = generate_random_state(config.nodes, rng);
        
        // Вывод информации
        print_graph_info(graph, config.steps, config.feedback_type);
        std::cout << "Начальное состояние: ";
        print_state(initial);
        std::cout << "\n";
        
        // Запуск бенчмарка
        auto output_cfg = create_output_config(config.nodes, config.steps);
        
        print_result_header(false);
        ComputeResult cpu_result = compute_cpu(graph, initial, config.steps, 
                                               config.feedback_type, &output_cfg);
        print_computation_result(cpu_result, false);
        
        print_result_header(true);
        ComputeResult cuda_result = compute_cuda(graph, initial, config.steps, 
                                                 config.feedback_type, &output_cfg);
        print_computation_result(cuda_result, true);
        
        // Проверка и сохранение
        bool match = (cpu_result.final_state == cuda_result.final_state) &&
                     (cpu_result.output_sequence == cuda_result.output_sequence);
        double speedup = cpu_result.elapsed_ms / cuda_result.elapsed_ms;
        
        print_comparison(match, speedup);
        
        results.push_back({config.name, cpu_result.elapsed_ms, 
                          cuda_result.elapsed_ms, speedup, match});
    }
    
    /**
     * Запуск бенчмарка для N-мерной решетки
     */
    void run_grid_test(const GridTestConfig& config) {
        print_test_header(config.name);
        
        // Генерация решетки
        NDGridConfig grid_cfg;
        grid_cfg.dimensions = config.dimensions;
        grid_cfg.periodic = config.periodic;
        grid_cfg.neighborhood = config.neighborhood;
        
        Graph graph = generate_nd_grid(grid_cfg);
        int total = grid_cfg.total_cells();
        std::vector<uint8_t> initial = generate_random_state(total, rng);
        
        // Вывод информации
        print_graph_info(graph, config.steps, config.feedback_type);
        std::cout << "Начальное состояние: ";
        print_state(initial);
        std::cout << "\n";
        
        // Запуск бенчмарка
        auto output_cfg = create_output_config(total, config.steps);
        
        print_result_header(false);
        ComputeResult cpu_result = compute_cpu(graph, initial, config.steps, 
                                               config.feedback_type, &output_cfg);
        print_computation_result(cpu_result, false);
        
        print_result_header(true);
        ComputeResult cuda_result = compute_cuda(graph, initial, config.steps, 
                                                 config.feedback_type, &output_cfg);
        print_computation_result(cuda_result, true);
        
        // Проверка и сохранение
        bool match = (cpu_result.final_state == cuda_result.final_state) &&
                     (cpu_result.output_sequence == cuda_result.output_sequence);
        double speedup = cpu_result.elapsed_ms / cuda_result.elapsed_ms;
        
        print_comparison(match, speedup);
        
        results.push_back({config.name, cpu_result.elapsed_ms, 
                          cuda_result.elapsed_ms, speedup, match});
    }
    
    /**
     * Вывод сводной таблицы результатов
     */
    void print_summary() {
        std::cout << "\n\n";
        std::cout << "╔════════════════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║                         СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ                        ║\n";
        std::cout << "╚════════════════════════════════════════════════════════════════════════════╝\n\n";
        
        std::vector<std::string> headers = {"Тест", "CPU", "CUDA", "Ускорение", "Статус"};
        std::vector<int> widths = {30, 12, 12, 12, 10};
        
        print_table_header(headers, widths);
        
        for (const auto& data : results) {
            std::vector<std::string> cells = {
                data.name,
                format_time(data.cpu_time),
                format_time(data.cuda_time),
                format_speedup(data.speedup),
                data.match ? "ДА" : "НЕТ"
            };
            print_table_row(cells, widths);
        }
        
        print_table_separator(widths);
        
        // Итоги
        double total_cpu = 0, total_cuda = 0;
        for (const auto& d : results) {
            total_cpu += d.cpu_time;
            total_cuda += d.cuda_time;
        }
        
        std::cout << "\nОбщее время CPU:  " << format_time(total_cpu) << "\n";
        std::cout << "Общее время CUDA: " << format_time(total_cuda) << "\n";
        std::cout << "Общее ускорение:  " << format_speedup(total_cpu / total_cuda) << "\n";
    }
    
    /**
     * Вывод графика производительности
     */
    void print_performance_chart() {
        std::vector<std::pair<std::string, double>> cpu_times, cuda_times;
        
        for (const auto& data : results) {
            cpu_times.push_back({data.name, data.cpu_time});
            cuda_times.push_back({data.name, data.cuda_time});
        }
        
        print_performance_graph(cpu_times, cuda_times);
    }
    
    const std::vector<BenchmarkData>& get_results() const {
        return results;
    }
};

#endif

