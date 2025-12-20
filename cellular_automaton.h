/**
 * ============================================================================
 * ОБОБЩЕННЫЙ КЛЕТОЧНЫЙ АВТОМАТ - Заголовочный файл
 * ============================================================================
 * 
 * Клеточный автомат (КА) - это дискретная вычислительная модель, состоящая из:
 * 1. Сетки ячеек, каждая из которых имеет состояние (обычно 0 или 1)
 * 2. Правила перехода, определяющего новое состояние на основе соседей
 * 
 * ОБОБЩЕННЫЙ КА отличается тем, что:
 * - Топология связей задается произвольным графом (не только регулярная сетка)
 * - Функция обратной связи может быть произвольной
 * 
 * Данная реализация поддерживает:
 * - Произвольные графы в CSR формате (эффективно для GPU)
 * - N-мерные регулярные решетки (1D, 2D, 3D, ... ND)
 * - Различные функции обратной связи (XOR, Majority, и др.)
 */

#ifndef CELLULAR_AUTOMATON_H
#define CELLULAR_AUTOMATON_H

#include <vector>
#include <cstdint>
#include <numeric>

/**
 * Структура графа связей в CSR (Compressed Sparse Row) формате
 * 
 * CSR - эффективный формат хранения разреженных графов:
 * - row_ptr[i] указывает на начало списка соседей вершины i в массиве col_idx
 * - col_idx содержит индексы всех соседей подряд
 * 
 * Пример для графа: 0->1, 0->2, 1->2, 2->0
 *   row_ptr = [0, 2, 3, 4]  (вершина 0 имеет 2 соседа, 1 имеет 1, 2 имеет 1)
 *   col_idx = [1, 2, 2, 0]  (соседи: 0->{1,2}, 1->{2}, 2->{0})
 * 
 * Преимущества CSR для GPU:
 * - Последовательный доступ к памяти
 * - Фиксированный размер структуры
 * - Эффективная коалесценция памяти
 */
struct Graph {
    std::vector<int> row_ptr;    // Размер: num_nodes + 1
    std::vector<int> col_idx;    // Размер: num_edges
    int num_nodes;               // Количество вершин (ячеек)
    int num_edges;               // Количество ребер (связей)
};

/**
 * Результаты вычисления автомата
 */
struct ComputeResult {
    std::vector<uint8_t> final_state;              // Конечное состояние
    std::vector<std::vector<uint8_t>> history;     // История всех состояний
    double elapsed_ms;                              // Время вычисления в мс
};

/**
 * Конфигурация N-мерной решетки
 * 
 * Позволяет задать решетку произвольной размерности:
 * - 1D: dimensions = {100}         -> линия из 100 ячеек
 * - 2D: dimensions = {100, 100}    -> сетка 100x100
 * - 3D: dimensions = {50, 50, 50}  -> куб 50x50x50
 * - ND: dimensions = {d1, d2, ..., dN}
 */
struct NDGridConfig {
    std::vector<int> dimensions;  // Размеры по каждому измерению
    bool periodic;                // Периодические граничные условия (тор)
    int neighborhood;             // 0 = фон Нейман (ортогональные соседи)
                                  // 1 = Мур (все соседи включая диагональные)
    
    // Общее количество ячеек
    int total_cells() const {
        if (dimensions.empty()) return 0;
        int total = 1;
        for (int d : dimensions) total *= d;
        return total;
    }
    
    // Количество измерений
    int ndim() const { return dimensions.size(); }
};

// ============================================================================
// ФУНКЦИИ ОБРАТНОЙ СВЯЗИ
// ============================================================================

/**
 * XOR (исключающее ИЛИ) всех соседей
 * 
 * Применение: линейные регистры сдвига, криптография
 * Свойство: обратима, сохраняет четность
 */
inline uint8_t feedback_xor(const uint8_t* neighbors, int count) {
    uint8_t result = 0;
    for (int i = 0; i < count; i++) {
        result ^= neighbors[i];
    }
    return result;
}

/**
 * Majority (большинство голосов)
 * 
 * Возвращает 1, если больше половины соседей = 1
 * Применение: моделирование консенсуса, сглаживание
 */
inline uint8_t feedback_majority(const uint8_t* neighbors, int count) {
    int sum = 0;
    for (int i = 0; i < count; i++) {
        sum += neighbors[i];
    }
    return (sum > count / 2) ? 1 : 0;
}

/**
 * Rule 110 - универсальный КА (Тьюринг-полный)
 * Работает только для 3 соседей (1D автомат)
 */
inline uint8_t feedback_rule110(const uint8_t* neighbors, int count) {
    if (count != 3) return feedback_xor(neighbors, count);
    // Правило 110: 01101110 в двоичном = 110
    int pattern = (neighbors[0] << 2) | (neighbors[1] << 1) | neighbors[2];
    return (110 >> pattern) & 1;
}

/**
 * Game of Life правило (для 8 соседей + центр)
 * B3/S23: рождение при 3, выживание при 2-3
 */
inline uint8_t feedback_life(const uint8_t* neighbors, int count, uint8_t current) {
    int alive = 0;
    for (int i = 0; i < count; i++) alive += neighbors[i];
    if (current) return (alive == 2 || alive == 3) ? 1 : 0;
    return (alive == 3) ? 1 : 0;
}

// ============================================================================
// ГЕНЕРАТОРЫ ГРАФОВ
// ============================================================================

// Случайный граф с заданной средней степенью
Graph generate_random_graph(int num_nodes, int avg_degree, unsigned int seed);

// 2D решетка (для совместимости)
Graph generate_grid_graph(int width, int height);

// N-мерная решетка (универсальная функция)
Graph generate_nd_grid(const NDGridConfig& config);

// ============================================================================
// ВЫЧИСЛЕНИЯ
// ============================================================================

// CPU реализация
ComputeResult compute_cpu(const Graph& graph, const std::vector<uint8_t>& initial_state, 
                          int steps, int feedback_type);

// CUDA реализация (GPU)
ComputeResult compute_cuda(const Graph& graph, const std::vector<uint8_t>& initial_state, 
                           int steps, int feedback_type);

#endif
