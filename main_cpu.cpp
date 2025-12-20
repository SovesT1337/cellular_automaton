/**
 * ============================================================================
 * CPU-ONLY БЕНЧМАРК клеточных автоматов
 * ============================================================================
 * 
 * Версия для систем без CUDA. Тестирует только CPU реализацию.
 */

#include "cellular_automaton.h"
#include <iostream>
#include <iomanip>
#include <random>

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

std::string dim_string(const std::vector<int>& dims) {
    std::string s;
    for (size_t i = 0; i < dims.size(); i++) {
        if (i > 0) s += "x";
        s += std::to_string(dims[i]);
    }
    return s;
}

void run_cpu_benchmark(const char* name, int num_nodes, int avg_degree, int steps, int feedback_type) {
    std::cout << "\n=== " << name << " ===\n";
    std::cout << "Nodes: " << num_nodes << ", Avg degree: " << avg_degree << ", Steps: " << steps << "\n";
    std::cout << "Feedback: " << (feedback_type == 0 ? "XOR" : "Majority") << "\n\n";
    
    Graph graph = generate_random_graph(num_nodes, avg_degree, 42);
    std::cout << "Graph: " << graph.num_nodes << " nodes, " << graph.num_edges << " edges\n";
    
    std::vector<uint8_t> initial(num_nodes);
    std::mt19937 rng(123);
    for (int i = 0; i < num_nodes; i++) {
        initial[i] = rng() % 2;
    }
    
    std::cout << "Initial: ";
    print_state(initial);
    
    std::cout << "\nCPU computation...\n";
    ComputeResult result = compute_cpu(graph, initial, steps, feedback_type);
    std::cout << "CPU time: " << std::fixed << std::setprecision(3) << result.elapsed_ms << " ms\n";
    std::cout << "Final:   ";
    print_state(result.final_state);
    
    double cells_per_sec = (double)num_nodes * steps / (result.elapsed_ms / 1000.0);
    std::cout << "Throughput: " << std::scientific << std::setprecision(2) << cells_per_sec << " cells/s\n";
}

void run_nd_cpu_benchmark(const std::vector<int>& dimensions, bool periodic, 
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
    
    Graph graph = generate_nd_grid(config);
    std::cout << "Edges: " << graph.num_edges << ", Avg degree: " << std::fixed 
              << std::setprecision(1) << (double)graph.num_edges / graph.num_nodes << "\n";
    
    std::vector<uint8_t> initial(total);
    std::mt19937 rng(456);
    for (int i = 0; i < total; i++) {
        initial[i] = rng() % 2;
    }
    
    ComputeResult result = compute_cpu(graph, initial, steps, feedback);
    
    double throughput = (double)total * steps / (result.elapsed_ms / 1000.0);
    std::cout << "CPU time: " << std::fixed << std::setprecision(3) << result.elapsed_ms << " ms\n";
    std::cout << "Throughput: " << std::scientific << std::setprecision(2) << throughput << " cells/s\n";
}

int main() {
    std::cout << "============================================================\n";
    std::cout << "  Generalized Cellular Automaton - CPU Benchmark\n";
    std::cout << "============================================================\n";
    
    // Random graphs
    std::cout << "\n>>> RANDOM GRAPH TESTS <<<\n";
    run_cpu_benchmark("Small (1K nodes)", 1000, 4, 100, 0);
    run_cpu_benchmark("Medium (10K nodes)", 10000, 6, 100, 0);
    run_cpu_benchmark("Large (100K nodes)", 100000, 8, 50, 0);
    run_cpu_benchmark("XLarge (1M nodes)", 1000000, 4, 20, 0);
    
    // N-dimensional grids
    std::cout << "\n\n>>> N-DIMENSIONAL GRID TESTS <<<\n";
    
    run_nd_cpu_benchmark({10000}, true, 0, 1000, 0);           // 1D
    run_nd_cpu_benchmark({500, 500}, false, 0, 100, 1);        // 2D von Neumann
    run_nd_cpu_benchmark({500, 500}, true, 1, 100, 1);         // 2D Moore
    run_nd_cpu_benchmark({100, 100, 100}, false, 0, 50, 0);    // 3D
    run_nd_cpu_benchmark({30, 30, 30, 30}, false, 0, 20, 0);   // 4D
    run_nd_cpu_benchmark({10, 10, 10, 10, 10}, false, 0, 10, 0); // 5D
    
    return 0;
}
