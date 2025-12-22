/**
 * ============================================================================
 * CPU-ONLY БЕНЧМАРК клеточных автоматов
 * ============================================================================
 * 
 * Версия для систем без CUDA. Тестирует только CPU реализацию.
 */

#include "benchmark_utils.h"
#include "test_configs.h"
#include <iostream>
#include <random>

class CPUBenchmarkRunner {
private:
    std::mt19937 rng;
    
public:
    CPUBenchmarkRunner(unsigned int seed = 123) : rng(seed) {}
    
    void run_random_graph_test(const TestConfig& config) {
        print_test_header(config.name);
        
        Graph graph = generate_random_graph(config.nodes, config.degree, 42);
        std::vector<uint8_t> initial = generate_random_state(config.nodes, rng);
        
        print_graph_info(graph, config.steps, config.feedback_type);
        std::cout << "Начальное состояние: ";
        print_state(initial);
        std::cout << "\n\n";
        
        auto output_cfg = create_output_config(config.nodes, config.steps);
        ComputeResult result = compute_cpu(graph, initial, config.steps, 
                                           config.feedback_type, &output_cfg);
        
        std::cout << "Время: " << format_time(result.elapsed_ms) << "\n";
        std::cout << "Конечное состояние: ";
        print_state(result.final_state);
        std::cout << "\n";
        std::cout << "Выходная последовательность (" << result.output_sequence.size() 
                  << " бит): ";
        print_state(result.output_sequence, 30);
        std::cout << "\n";
        
        double throughput = (double)config.nodes * config.steps / (result.elapsed_ms / 1000.0);
        std::cout << "Пропускная способность: " << format_throughput(throughput) << " ячеек/с\n";
    }
    
    void run_grid_test(const GridTestConfig& config) {
        print_test_header(config.name);
        
        NDGridConfig grid_cfg;
        grid_cfg.dimensions = config.dimensions;
        grid_cfg.periodic = config.periodic;
        grid_cfg.neighborhood = config.neighborhood;
        
        Graph graph = generate_nd_grid(grid_cfg);
        int total = grid_cfg.total_cells();
        std::vector<uint8_t> initial = generate_random_state(total, rng);
        
        print_graph_info(graph, config.steps, config.feedback_type);
        std::cout << "Начальное состояние: ";
        print_state(initial);
        std::cout << "\n\n";
        
        auto output_cfg = create_output_config(total, config.steps);
        ComputeResult result = compute_cpu(graph, initial, config.steps, 
                                           config.feedback_type, &output_cfg);
        
        std::cout << "Время: " << format_time(result.elapsed_ms) << "\n";
        std::cout << "Конечное состояние: ";
        print_state(result.final_state);
        std::cout << "\n";
        std::cout << "Выходная последовательность (" << result.output_sequence.size() 
                  << " бит): ";
        print_state(result.output_sequence, 30);
        std::cout << "\n";
        
        double throughput = (double)total * config.steps / (result.elapsed_ms / 1000.0);
        std::cout << "Пропускная способность: " << format_throughput(throughput) << " ячеек/с\n";
    }
};

int main() {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                                                                      ║\n";
    std::cout << "║         ОБОБЩЕННЫЙ КЛЕТОЧНЫЙ АВТОМАТ - CPU БЕНЧМАРК                 ║\n";
    std::cout << "║                                                                      ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════════╝\n";
    
    CPUBenchmarkRunner runner(123);
    
    print_category_header("СЛУЧАЙНЫЕ ГРАФЫ");
    for (const auto& test : get_random_graph_tests()) {
        runner.run_random_graph_test(test);
    }
    
    print_category_header("1D ЛИНЕЙНЫЕ АВТОМАТЫ");
    for (const auto& test : get_1d_tests()) {
        runner.run_grid_test(test);
    }
    
    print_category_header("2D СЕТКИ");
    for (const auto& test : get_2d_tests()) {
        runner.run_grid_test(test);
    }
    
    print_category_header("3D ОБЪЕМНЫЕ РЕШЕТКИ");
    for (const auto& test : get_3d_tests()) {
        runner.run_grid_test(test);
    }
    
    print_category_header("ВЫСОКОРАЗМЕРНЫЕ РЕШЕТКИ");
    for (const auto& test : get_high_dim_tests()) {
        runner.run_grid_test(test);
    }
    
    print_category_header("КРУПНОМАСШТАБНЫЕ ТЕСТЫ");
    for (const auto& test : get_large_scale_graph_tests()) {
        runner.run_random_graph_test(test);
    }
    for (const auto& test : get_large_scale_grid_tests()) {
        runner.run_grid_test(test);
    }
    
    return 0;
}
