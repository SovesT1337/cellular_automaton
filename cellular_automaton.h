/**
 * Обобщенный клеточный автомат
 * Вычисление выходной последовательности по заданному графу и функции обратной связи
 */

#ifndef CELLULAR_AUTOMATON_H
#define CELLULAR_AUTOMATON_H

#include <vector>
#include <cstdint>

// Граф связей в CSR (Compressed Sparse Row) формате
struct Graph {
    std::vector<int> row_ptr;    // Размер: num_nodes + 1
    std::vector<int> col_idx;    // Размер: num_edges
    int num_nodes;
    int num_edges;
};

// Результаты вычисления
struct ComputeResult {
    std::vector<uint8_t> final_state;
    std::vector<std::vector<uint8_t>> history;
    std::vector<uint8_t> output_sequence;
    double elapsed_ms;
};

// Конфигурация выходной последовательности
struct OutputConfig {
    std::vector<int> cells;           // Индексы ячеек для выходной последовательности
    int extract_every_n_steps;        // Извлекать каждые N шагов
    
    OutputConfig() : extract_every_n_steps(1) {}
};

// Функции обратной связи
inline uint8_t feedback_xor(const uint8_t* neighbors, int count) {
    uint8_t result = 0;
    for (int i = 0; i < count; i++) {
        result ^= neighbors[i];
    }
    return result;
}

inline uint8_t feedback_majority(const uint8_t* neighbors, int count) {
    int sum = 0;
    for (int i = 0; i < count; i++) {
        sum += neighbors[i];
    }
    return (sum > count / 2) ? 1 : 0;
}

// Генератор графа
Graph generate_random_graph(int num_nodes, int avg_degree, unsigned int seed);

// Вычисления
ComputeResult compute_cpu(const Graph& graph, const std::vector<uint8_t>& initial_state, 
                          int steps, int feedback_type, const OutputConfig* output_cfg = nullptr);

ComputeResult compute_cuda(const Graph& graph, const std::vector<uint8_t>& initial_state, 
                           int steps, int feedback_type, const OutputConfig* output_cfg = nullptr);

#endif
