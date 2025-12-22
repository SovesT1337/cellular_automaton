/**
 * ============================================================================
 * УТИЛИТЫ ДЛЯ БЕНЧМАРКОВ
 * ============================================================================
 */

#ifndef BENCHMARK_UTILS_H
#define BENCHMARK_UTILS_H

#include "cellular_automaton.h"
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <random>

// ============================================================================
// СТРУКТУРЫ ДАННЫХ
// ============================================================================

struct BenchmarkData {
    std::string name;
    double cpu_time;
    double cuda_time;
    double speedup;
    bool match;
};

struct TestConfig {
    std::string name;
    int nodes;
    int degree;
    int steps;
    int feedback_type;
    
    TestConfig(const std::string& n, int nodes, int deg, int s, int fb = 0)
        : name(n), nodes(nodes), degree(deg), steps(s), feedback_type(fb) {}
};

struct GridTestConfig {
    std::string name;
    std::vector<int> dimensions;
    bool periodic;
    int neighborhood;
    int steps;
    int feedback_type;
    
    GridTestConfig(const std::string& n, std::vector<int> dims, bool per, int nb, int s, int fb = 0)
        : name(n), dimensions(dims), periodic(per), neighborhood(nb), steps(s), feedback_type(fb) {}
};

// ============================================================================
// ФОРМАТИРОВАНИЕ
// ============================================================================

inline void print_state(const std::vector<uint8_t>& state, int max_display = 20) {
    int n = std::min((int)state.size(), max_display);
    for (int i = 0; i < n; i++) {
        std::cout << (int)state[i];
    }
    if ((int)state.size() > max_display) {
        std::cout << "...";
    }
}

inline std::string format_time(double ms) {
    std::ostringstream oss;
    if (ms >= 1000) {
        oss << std::fixed << std::setprecision(2) << (ms / 1000.0) << " с";
    } else {
        oss << std::fixed << std::setprecision(3) << ms << " мс";
    }
    return oss.str();
}

inline std::string format_speedup(double speedup) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << speedup << "x";
    return oss.str();
}

inline std::string format_throughput(double throughput) {
    std::ostringstream oss;
    oss << std::scientific << std::setprecision(2) << throughput;
    return oss.str();
}

inline std::string feedback_name(int type) {
    switch (type) {
        case 0: return "XOR";
        case 1: return "Majority";
        case 2: return "Rule110";
        default: return "Unknown";
    }
}

// ============================================================================
// ВЫВОД
// ============================================================================

inline void print_test_header(const std::string& name) {
    std::cout << "\n╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║ ТЕСТ: " << std::left << std::setw(56) << name << "║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n";
}

inline void print_graph_info(const Graph& graph, int steps, int feedback_type) {
    std::cout << "Узлов: " << graph.num_nodes 
              << ", Рёбер: " << graph.num_edges 
              << ", Шагов: " << steps << "\n";
    std::cout << "Средняя степень: " << std::fixed << std::setprecision(1) 
              << (double)graph.num_edges / graph.num_nodes << "\n";
    std::cout << "Функция обратной связи: " << feedback_name(feedback_type) << "\n";
}

inline void print_result_header(bool is_cuda) {
    std::cout << "\n[" << (is_cuda ? "CUDA" : "CPU") << "] Вычисление...\n";
}

inline void print_computation_result(const ComputeResult& result, bool is_cuda) {
    std::cout << "[" << (is_cuda ? "CUDA" : "CPU") << "] Время: " 
              << format_time(result.elapsed_ms) << "\n";
    std::cout << "[" << (is_cuda ? "CUDA" : "CPU") << "] Конечное состояние: ";
    print_state(result.final_state);
    std::cout << "\n";
    std::cout << "[" << (is_cuda ? "CUDA" : "CPU") << "] Выходная последовательность (" 
              << result.output_sequence.size() << " бит): ";
    print_state(result.output_sequence, 30);
    std::cout << "\n";
}

inline void print_comparison(bool match, double speedup) {
    std::cout << "\n┌─────────────────────────────────────────────────────────┐\n";
    std::cout << "│ РЕЗУЛЬТАТ: " << (match ? "✓ Совпадение" : "✗ Несовпадение") 
              << std::string(match ? 37 : 35, ' ') << "│\n";
    std::cout << "│ Ускорение (CPU/CUDA): " << std::setw(33) << std::right 
              << format_speedup(speedup) << " │\n";
    std::cout << "└─────────────────────────────────────────────────────────┘\n";
}

inline void print_category_header(const std::string& category) {
    std::cout << "\n\n>>> " << category << " <<<\n";
}

// ============================================================================
// СОЗДАНИЕ ВЫХОДНОЙ КОНФИГУРАЦИИ
// ============================================================================

inline OutputConfig create_output_config(int num_nodes, int steps) {
    OutputConfig config;
    int num_output_cells = std::min(5, num_nodes);
    for (int i = 0; i < num_output_cells; i++) {
        config.cells.push_back(i);
    }
    config.extract_every_n_steps = std::max(1, steps / 10);
    return config;
}

// ============================================================================
// ГЕНЕРАЦИЯ НАЧАЛЬНОГО СОСТОЯНИЯ
// ============================================================================

inline std::vector<uint8_t> generate_random_state(int size, std::mt19937& rng) {
    std::vector<uint8_t> state(size);
    for (int i = 0; i < size; i++) {
        state[i] = rng() % 2;
    }
    return state;
}

#endif

