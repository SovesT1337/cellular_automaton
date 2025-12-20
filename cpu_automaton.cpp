/**
 * ============================================================================
 * CPU РЕАЛИЗАЦИЯ КЛЕТОЧНОГО АВТОМАТА
 * ============================================================================
 * 
 * Последовательная реализация для выполнения на центральном процессоре.
 * Используется как эталон для проверки корректности GPU версии.
 */

#include "cellular_automaton.h"
#include <chrono>
#include <random>
#include <cstring>
#include <algorithm>

/**
 * Генерация случайного графа (Эрдёша-Реньи подобный)
 * 
 * @param num_nodes   Количество вершин
 * @param avg_degree  Средняя степень вершины (количество соседей)
 * @param seed        Seed для воспроизводимости
 * 
 * Алгоритм:
 * 1. Для каждой вершины случайно выбираем степень около avg_degree
 * 2. Случайно выбираем соседей из всех вершин
 * 3. Конвертируем в CSR формат
 */
Graph generate_random_graph(int num_nodes, int avg_degree, unsigned int seed) {
    Graph g;
    g.num_nodes = num_nodes;
    g.row_ptr.resize(num_nodes + 1);
    
    // Mersenne Twister - качественный ГПСЧ
    std::mt19937 rng(seed);
    
    // Степень варьируется от 1 до 2*avg_degree-1 (среднее = avg_degree)
    std::uniform_int_distribution<int> degree_dist(1, avg_degree * 2 - 1);
    std::uniform_int_distribution<int> node_dist(0, num_nodes - 1);
    
    // Временное хранение списков смежности
    std::vector<std::vector<int>> adj(num_nodes);
    
    for (int i = 0; i < num_nodes; i++) {
        int degree = degree_dist(rng);
        for (int j = 0; j < degree; j++) {
            int neighbor = node_dist(rng);
            // Исключаем петли (связь вершины с самой собой)
            if (neighbor != i) {
                adj[i].push_back(neighbor);
            }
        }
    }
    
    // Конвертация в CSR формат
    // row_ptr[i] = сумма степеней вершин 0..i-1
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

/**
 * Генерация 2D решетки (частный случай N-мерной)
 * 
 * Окрестность фон Неймана: 4 соседа (вверх, вниз, влево, вправо)
 * Граничные условия: не периодические (края не связаны)
 */
Graph generate_grid_graph(int width, int height) {
    NDGridConfig config;
    config.dimensions = {width, height};
    config.periodic = false;
    config.neighborhood = 0;  // фон Нейман
    return generate_nd_grid(config);
}

/**
 * Генерация N-мерной регулярной решетки
 * 
 * Это ключевая функция для многомерных автоматов.
 * 
 * Идея: каждая ячейка имеет координаты (x1, x2, ..., xN).
 * Линейный индекс вычисляется как: idx = x1 + d1*(x2 + d2*(x3 + ...))
 * 
 * Соседи ячейки:
 * - Фон Нейман: ячейки, отличающиеся на ±1 по одной координате (2N соседей)
 * - Мур: ячейки, отличающиеся на ±1 по любым координатам (3^N - 1 соседей)
 * 
 * @param config Конфигурация решетки (размеры, периодичность, тип окрестности)
 */
Graph generate_nd_grid(const NDGridConfig& config) {
    Graph g;
    int ndim = config.ndim();
    int total = config.total_cells();
    
    g.num_nodes = total;
    g.row_ptr.resize(total + 1);
    
    if (ndim == 0 || total == 0) {
        g.row_ptr[0] = 0;
        g.num_edges = 0;
        return g;
    }
    
    // Вычисляем шаги для перехода между координатами
    // strides[i] = произведение dimensions[0..i-1]
    // Это позволяет быстро конвертировать между линейным индексом и координатами
    std::vector<int> strides(ndim);
    strides[0] = 1;
    for (int d = 1; d < ndim; d++) {
        strides[d] = strides[d-1] * config.dimensions[d-1];
    }
    
    // Функция: линейный индекс -> координаты
    auto idx_to_coords = [&](int idx, std::vector<int>& coords) {
        coords.resize(ndim);
        for (int d = ndim - 1; d >= 0; d--) {
            coords[d] = idx / strides[d];
            idx %= strides[d];
        }
    };
    
    // Функция: координаты -> линейный индекс
    auto coords_to_idx = [&](const std::vector<int>& coords) {
        int idx = 0;
        for (int d = 0; d < ndim; d++) {
            idx += coords[d] * strides[d];
        }
        return idx;
    };
    
    std::vector<int> coords(ndim);
    std::vector<int> neighbor_coords(ndim);
    
    g.row_ptr[0] = 0;
    
    for (int node = 0; node < total; node++) {
        idx_to_coords(node, coords);
        
        if (config.neighborhood == 0) {
            // Окрестность фон Неймана: только ортогональные соседи
            // В N измерениях: 2N соседей (±1 по каждой оси)
            for (int d = 0; d < ndim; d++) {
                for (int delta : {-1, 1}) {
                    neighbor_coords = coords;
                    int new_val = coords[d] + delta;
                    
                    if (config.periodic) {
                        // Периодические условия: тор
                        new_val = (new_val + config.dimensions[d]) % config.dimensions[d];
                        neighbor_coords[d] = new_val;
                        g.col_idx.push_back(coords_to_idx(neighbor_coords));
                    } else {
                        // Открытые границы
                        if (new_val >= 0 && new_val < config.dimensions[d]) {
                            neighbor_coords[d] = new_val;
                            g.col_idx.push_back(coords_to_idx(neighbor_coords));
                        }
                    }
                }
            }
        } else {
            // Окрестность Мура: все соседи включая диагональные
            // В N измерениях: 3^N - 1 соседей
            // Перебираем все комбинации смещений {-1, 0, 1}^N кроме (0,0,...,0)
            
            int num_combinations = 1;
            for (int d = 0; d < ndim; d++) num_combinations *= 3;
            
            for (int combo = 0; combo < num_combinations; combo++) {
                if (combo == num_combinations / 2) continue;  // Пропускаем (0,0,...,0)
                
                neighbor_coords = coords;
                bool valid = true;
                int temp = combo;
                
                for (int d = 0; d < ndim; d++) {
                    int delta = (temp % 3) - 1;  // -1, 0, или 1
                    temp /= 3;
                    
                    int new_val = coords[d] + delta;
                    
                    if (config.periodic) {
                        new_val = (new_val + config.dimensions[d]) % config.dimensions[d];
                    } else if (new_val < 0 || new_val >= config.dimensions[d]) {
                        valid = false;
                        break;
                    }
                    neighbor_coords[d] = new_val;
                }
                
                if (valid) {
                    g.col_idx.push_back(coords_to_idx(neighbor_coords));
                }
            }
        }
        
        g.row_ptr[node + 1] = g.col_idx.size();
    }
    
    g.num_edges = g.col_idx.size();
    return g;
}

/**
 * CPU вычисление клеточного автомата
 * 
 * Алгоритм:
 * 1. Для каждого шага времени:
 *    a. Для каждой ячейки:
 *       - Собрать состояния всех соседей
 *       - Применить функцию обратной связи
 *       - Записать новое состояние
 *    b. Поменять местами текущее и новое состояние
 * 
 * Сложность: O(steps * num_edges)
 * 
 * @param graph         Граф связей в CSR формате
 * @param initial_state Начальное состояние ячеек
 * @param steps         Количество шагов симуляции
 * @param feedback_type 0 = XOR, 1 = Majority
 */
ComputeResult compute_cpu(const Graph& graph, const std::vector<uint8_t>& initial_state, 
                          int steps, int feedback_type) {
    ComputeResult result;
    int n = graph.num_nodes;
    
    // Два буфера для текущего и следующего состояния (double buffering)
    std::vector<uint8_t> current = initial_state;
    std::vector<uint8_t> next(n);
    
    // Буфер для хранения состояний соседей одной ячейки
    std::vector<uint8_t> neighbors_buf(256);
    
    // Сохраняем начальное состояние
    result.history.push_back(current);
    
    // Замер времени с высоким разрешением
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < steps; step++) {
        // Обработка каждой ячейки
        for (int i = 0; i < n; i++) {
            // CSR: соседи ячейки i находятся в col_idx[row_ptr[i]..row_ptr[i+1])
            int start_idx = graph.row_ptr[i];
            int end_idx = graph.row_ptr[i + 1];
            int degree = end_idx - start_idx;
            
            // Собираем состояния соседей
            for (int j = 0; j < degree; j++) {
                int neighbor_idx = graph.col_idx[start_idx + j];
                neighbors_buf[j] = current[neighbor_idx];
            }
            
            // Применяем функцию обратной связи
            switch (feedback_type) {
                case 0:
                    next[i] = feedback_xor(neighbors_buf.data(), degree);
                    break;
                case 1:
                    next[i] = feedback_majority(neighbors_buf.data(), degree);
                    break;
                case 2:
                    next[i] = feedback_rule110(neighbors_buf.data(), degree);
                    break;
                default:
                    next[i] = feedback_xor(neighbors_buf.data(), degree);
            }
        }
        
        // Эффективная замена буферов (O(1) вместо O(n) копирования)
        std::swap(current, next);
        
        // Сохраняем историю
        result.history.push_back(current);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    result.elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result.final_state = current;
    
    return result;
}
