/**
 * CPU реализация клеточного автомата
 */

#include "cellular_automaton.h"
#include <chrono>
#include <random>

// Генерация случайного графа
Graph generate_random_graph(int num_nodes, int avg_degree, unsigned int seed) {
    Graph g;
    g.num_nodes = num_nodes;
    g.row_ptr.resize(num_nodes + 1);
    
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> degree_dist(1, avg_degree * 2 - 1);
    std::uniform_int_distribution<int> node_dist(0, num_nodes - 1);
    
    std::vector<std::vector<int>> adj(num_nodes);
    
    for (int i = 0; i < num_nodes; i++) {
        int degree = degree_dist(rng);
        for (int j = 0; j < degree; j++) {
            int neighbor = node_dist(rng);
            if (neighbor != i) {
                adj[i].push_back(neighbor);
            }
        }
    }
    
    // Конвертация в CSR формат
    g.row_ptr[0] = 0;
    for (int i = 0; i < num_nodes; i++) {
        g.row_ptr[i + 1] = g.row_ptr[i] + adj[i].size();
        for (int n : adj[i]) {
            g.col_idx.push_back(n);
        }
    }
    g.num_edges = g.col_idx.size();
    
    return g;
}

// CPU вычисление клеточного автомата
ComputeResult compute_cpu(const Graph& graph, const std::vector<uint8_t>& initial_state, 
                          int steps, int feedback_type, const OutputConfig* output_cfg) {
    ComputeResult result;
    int n = graph.num_nodes;
    
    std::vector<uint8_t> current = initial_state;
    std::vector<uint8_t> next(n);
    std::vector<uint8_t> neighbors_buf(256);
    
    result.history.push_back(current);
    
    if (output_cfg && !output_cfg->cells.empty()) {
        for (int cell_idx : output_cfg->cells) {
            if (cell_idx >= 0 && cell_idx < n) {
                result.output_sequence.push_back(current[cell_idx]);
            }
        }
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < steps; step++) {
        for (int i = 0; i < n; i++) {
            int start_idx = graph.row_ptr[i];
            int end_idx = graph.row_ptr[i + 1];
            int degree = end_idx - start_idx;
            
            for (int j = 0; j < degree; j++) {
                int neighbor_idx = graph.col_idx[start_idx + j];
                neighbors_buf[j] = current[neighbor_idx];
            }
            
            switch (feedback_type) {
                case 0:
                    next[i] = feedback_xor(neighbors_buf.data(), degree);
                    break;
                case 1:
                    next[i] = feedback_majority(neighbors_buf.data(), degree);
                    break;
                default:
                    next[i] = feedback_xor(neighbors_buf.data(), degree);
            }
        }
        
        std::swap(current, next);
        result.history.push_back(current);
        
        if (output_cfg && !output_cfg->cells.empty()) {
            if ((step + 1) % output_cfg->extract_every_n_steps == 0) {
                for (int cell_idx : output_cfg->cells) {
                    if (cell_idx >= 0 && cell_idx < n) {
                        result.output_sequence.push_back(current[cell_idx]);
                    }
                }
            }
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    result.elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result.final_state = current;
    
    return result;
}
