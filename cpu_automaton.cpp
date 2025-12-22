/**
 * ============================================================================
 * CPU РЕАЛИЗАЦИЯ КЛЕТОЧНОГО АВТОМАТА
 * ============================================================================
 * 
 * Последовательная реализация для выполнения на центральном процессоре.
 * Используется как эталон для проверки корректности GPU версии.
 * 
 * ОСНОВНЫЕ ФУНКЦИИ:
 * - generate_random_graph()  - генерация случайного графа Эрдёша-Реньи
 * - generate_nd_grid()       - генерация N-мерной регулярной решетки
 * - compute_cpu()            - вычисление эволюции автомата на CPU
 * - print_*()                - утилиты вывода таблиц и графиков
 */

#include "cellular_automaton.h"
#include <chrono>
#include <random>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <sstream>

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
                          int steps, int feedback_type, const OutputConfig* output_cfg) {
    ComputeResult result;
    int n = graph.num_nodes;
    
    // Два буфера для текущего и следующего состояния (double buffering)
    std::vector<uint8_t> current = initial_state;
    std::vector<uint8_t> next(n);
    
    // Буфер для хранения состояний соседей одной ячейки
    std::vector<uint8_t> neighbors_buf(256);
    
    // Сохраняем начальное состояние
    result.history.push_back(current);
    
    // Формирование выходной последовательности (если задана конфигурация)
    if (output_cfg && !output_cfg->cells.empty()) {
        for (int cell_idx : output_cfg->cells) {
            if (cell_idx >= 0 && cell_idx < n) {
                result.output_sequence.push_back(current[cell_idx]);
            }
        }
    }
    
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
        
        // Извлекаем значения для выходной последовательности
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

// ============================================================================
// УТИЛИТЫ ДЛЯ КРАСИВОГО ВЫВОДА
// ============================================================================

#include <iostream>
#include <sstream>
#include <iomanip>

void print_table_separator(const std::vector<int>& widths) {
    std::cout << "+";
    for (int w : widths) {
        std::cout << std::string(w + 2, '-') << "+";
    }
    std::cout << "\n";
}

void print_table_header(const std::vector<std::string>& headers, const std::vector<int>& widths) {
    print_table_separator(widths);
    std::cout << "|";
    for (size_t i = 0; i < headers.size(); i++) {
        std::cout << " " << std::left << std::setw(widths[i]) << headers[i] << " |";
    }
    std::cout << "\n";
    print_table_separator(widths);
}

void print_table_row(const std::vector<std::string>& cells, const std::vector<int>& widths) {
    std::cout << "|";
    for (size_t i = 0; i < cells.size(); i++) {
        std::cout << " " << std::left << std::setw(widths[i]) << cells[i] << " |";
    }
    std::cout << "\n";
}

void print_performance_graph(const std::vector<std::pair<std::string, double>>& cpu_times,
                             const std::vector<std::pair<std::string, double>>& cuda_times) {
    if (cpu_times.size() != cuda_times.size() || cpu_times.empty()) return;
    
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║         ГРАФИК СРАВНЕНИЯ ПРОИЗВОДИТЕЛЬНОСТИ CPU vs CUDA           ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════╝\n\n";
    
    // Находим максимальное время для масштабирования
    double max_time = 0;
    for (const auto& p : cpu_times) {
        if (p.second > max_time) max_time = p.second;
    }
    
    const int bar_width = 50;
    
    for (size_t i = 0; i < cpu_times.size(); i++) {
        std::cout << std::setw(25) << std::left << cpu_times[i].first << " |\n";
        
        // CPU bar
        int cpu_len = (int)((cpu_times[i].second / max_time) * bar_width);
        std::cout << "  CPU:  " << std::string(cpu_len, '#') << " " 
                  << std::fixed << std::setprecision(2) << cpu_times[i].second << " мс\n";
        
        // CUDA bar
        int cuda_len = (int)((cuda_times[i].second / max_time) * bar_width);
        double speedup = cpu_times[i].second / cuda_times[i].second;
        std::cout << "  CUDA: " << std::string(cuda_len, '#') << " " 
                  << std::fixed << std::setprecision(2) << cuda_times[i].second 
                  << " мс (ускорение: " << std::setprecision(1) << speedup << "x)\n\n";
    }
}
