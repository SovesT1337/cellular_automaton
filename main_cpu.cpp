/**
 * ============================================================================
 * CPU-ONLY БЕНЧМАРК клеточных автоматов
 * ============================================================================
 */

#include "cellular_automaton.h"
#include <iostream>
#include <iomanip>
#include <random>
#include <sstream>

void print_state(const std::vector<uint8_t>& state, int max_display = 20) {
    int n = std::min((int)state.size(), max_display);
    for (int i = 0; i < n; i++) {
        std::cout << (int)state[i];
    }
    if ((int)state.size() > max_display) {
        std::cout << "...";
    }
}

std::string format_time(double ms) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3) << ms;
    return oss.str();
}

std::string format_throughput(double throughput) {
    std::ostringstream oss;
    oss << std::scientific << std::setprecision(2) << throughput;
    return oss.str();
}

void run_cpu_test(const char* name, const Graph& graph, 
                  const std::vector<uint8_t>& initial, 
                  int steps, int feedback_type) {
    std::cout << "\n╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║ ТЕСТ: " << std::left << std::setw(56) << name << "║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n";
    
    std::cout << "Узлов: " << graph.num_nodes << ", Рёбер: " << graph.num_edges 
              << ", Шагов: " << steps << "\n";
    std::cout << "Средняя степень: " << std::fixed << std::setprecision(1) 
              << (double)graph.num_edges / graph.num_nodes << "\n";
    
    // Настройка выходной последовательности
    OutputConfig output_cfg;
    int num_output_cells = std::min(5, graph.num_nodes);
    for (int i = 0; i < num_output_cells; i++) {
        output_cfg.cells.push_back(i);
    }
    
    std::cout << "Начальное состояние: ";
    print_state(initial);
    std::cout << "\n\n";
    
    ComputeResult result = compute_cpu(graph, initial, steps, feedback_type, &output_cfg);
    
    std::cout << "Время: " << format_time(result.elapsed_ms) << " мс\n";
    std::cout << "Конечное состояние: ";
    print_state(result.final_state);
    std::cout << "\n";
    std::cout << "Выходная последовательность (" << result.output_sequence.size() 
              << " бит): ";
    print_state(result.output_sequence, 30);
    std::cout << "\n";
    
    double throughput = (double)graph.num_nodes * steps / (result.elapsed_ms / 1000.0);
    std::cout << "Пропускная способность: " << format_throughput(throughput) << " ячеек/с\n";
}

int main() {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                                                                      ║\n";
    std::cout << "║         ОБОБЩЕННЫЙ КЛЕТОЧНЫЙ АВТОМАТ - CPU БЕНЧМАРК                 ║\n";
    std::cout << "║                                                                      ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════════╝\n";
    
    std::mt19937 rng(123);
    
    std::cout << "\n\n>>> СЛУЧАЙНЫЕ ГРАФЫ <<<\n";
    
    {
        Graph g = generate_random_graph(1000, 4, 42);
        std::vector<uint8_t> init(1000);
        for (int i = 0; i < 1000; i++) init[i] = rng() % 2;
        run_cpu_test("1K узлов", g, init, 100, 0);
    }
    
    {
        Graph g = generate_random_graph(10000, 6, 42);
        std::vector<uint8_t> init(10000);
        for (int i = 0; i < 10000; i++) init[i] = rng() % 2;
        run_cpu_test("10K узлов", g, init, 100, 0);
    }
    
    {
        Graph g = generate_random_graph(100000, 8, 42);
        std::vector<uint8_t> init(100000);
        for (int i = 0; i < 100000; i++) init[i] = rng() % 2;
        run_cpu_test("100K узлов", g, init, 50, 0);
    }
    
    {
        Graph g = generate_random_graph(1000000, 4, 42);
        std::vector<uint8_t> init(1000000);
        for (int i = 0; i < 1000000; i++) init[i] = rng() % 2;
        run_cpu_test("1M узлов", g, init, 20, 0);
    }
    
    std::cout << "\n\n>>> N-МЕРНЫЕ РЕШЕТКИ <<<\n";
    
    {
        NDGridConfig cfg;
        cfg.dimensions = {10000};
        cfg.periodic = true;
        cfg.neighborhood = 0;
        Graph g = generate_nd_grid(cfg);
        std::vector<uint8_t> init(cfg.total_cells());
        for (int i = 0; i < cfg.total_cells(); i++) init[i] = rng() % 2;
        run_cpu_test("1D линия 10000", g, init, 1000, 0);
    }
    
    {
        NDGridConfig cfg;
        cfg.dimensions = {500, 500};
        cfg.periodic = false;
        cfg.neighborhood = 0;
        Graph g = generate_nd_grid(cfg);
        std::vector<uint8_t> init(cfg.total_cells());
        for (int i = 0; i < cfg.total_cells(); i++) init[i] = rng() % 2;
        run_cpu_test("2D сетка 500x500", g, init, 100, 1);
    }
    
    {
        NDGridConfig cfg;
        cfg.dimensions = {100, 100, 100};
        cfg.periodic = false;
        cfg.neighborhood = 0;
        Graph g = generate_nd_grid(cfg);
        std::vector<uint8_t> init(cfg.total_cells());
        for (int i = 0; i < cfg.total_cells(); i++) init[i] = rng() % 2;
        run_cpu_test("3D куб 100x100x100", g, init, 50, 0);
    }
    
    {
        NDGridConfig cfg;
        cfg.dimensions = {30, 30, 30, 30};
        cfg.periodic = false;
        cfg.neighborhood = 0;
        Graph g = generate_nd_grid(cfg);
        std::vector<uint8_t> init(cfg.total_cells());
        for (int i = 0; i < cfg.total_cells(); i++) init[i] = rng() % 2;
        run_cpu_test("4D гиперкуб 30^4", g, init, 20, 0);
    }
    
    return 0;
}
