/**
 * ============================================================================
 * CUDA РЕАЛИЗАЦИЯ КЛЕТОЧНОГО АВТОМАТА
 * ============================================================================
 * 
 * Параллельная реализация для выполнения на графическом процессоре NVIDIA.
 * 
 * АРХИТЕКТУРА CUDA:
 * - GPU состоит из множества SM (Streaming Multiprocessors)
 * - Каждый SM выполняет блоки потоков
 * - Потоки в блоке могут синхронизироваться и делить shared memory
 * 
 * МОДЕЛЬ ПАРАЛЛЕЛИЗМА:
 * - Каждая ячейка автомата обрабатывается отдельным потоком (thread)
 * - Потоки группируются в блоки по 256 (оптимально для большинства GPU)
 * - Все ячейки обрабатываются параллельно за один проход kernel
 * 
 * ПРЕИМУЩЕСТВА GPU:
 * - Тысячи ячеек обрабатываются одновременно
 * - Высокая пропускная способность памяти
 * - Эффективно для регулярных вычислений (SIMD)
 */

#include "cellular_automaton.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>

/**
 * Макрос для проверки ошибок CUDA
 * 
 * Каждый вызов CUDA API может вернуть ошибку.
 * Этот макрос проверяет результат и завершает программу при ошибке.
 */
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

/**
 * CUDA Kernel: XOR функция обратной связи
 * 
 * __global__ означает, что функция выполняется на GPU и вызывается с хоста.
 * 
 * Каждый поток обрабатывает одну ячейку:
 * 1. Вычисляет свой глобальный индекс (i)
 * 2. Находит соседей через CSR структуру
 * 3. Вычисляет XOR состояний всех соседей
 * 4. Записывает результат в выходной массив
 * 
 * @param current  Текущее состояние всех ячеек (device memory)
 * @param next     Выходной массив для нового состояния
 * @param row_ptr  CSR: указатели на начало списка соседей
 * @param col_idx  CSR: индексы соседей
 * @param n        Общее количество ячеек
 */
__global__ void step_kernel_xor(const uint8_t* current, uint8_t* next,
                                 const int* row_ptr, const int* col_idx, int n) {
    // Глобальный индекс потока
    // blockIdx.x - номер блока, blockDim.x - размер блока, threadIdx.x - номер в блоке
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Защита от выхода за границы (последний блок может быть неполным)
    if (i >= n) return;
    
    // Диапазон соседей в CSR формате
    int start = row_ptr[i];
    int end = row_ptr[i + 1];
    
    // Вычисление XOR всех соседей
    uint8_t result = 0;
    for (int j = start; j < end; j++) {
        result ^= current[col_idx[j]];  // Чтение состояния соседа
    }
    next[i] = result;
}

/**
 * CUDA Kernel: Majority функция обратной связи
 * 
 * Аналогично XOR, но считает сумму и сравнивает с половиной.
 */
__global__ void step_kernel_majority(const uint8_t* current, uint8_t* next,
                                      const int* row_ptr, const int* col_idx, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    int start = row_ptr[i];
    int end = row_ptr[i + 1];
    int degree = end - start;
    
    // Подсчет активных соседей
    int sum = 0;
    for (int j = start; j < end; j++) {
        sum += current[col_idx[j]];
    }
    
    // Большинство голосов
    next[i] = (sum > degree / 2) ? 1 : 0;
}

/**
 * CUDA Kernel: Rule 110 (универсальный 1D автомат)
 */
__global__ void step_kernel_rule110(const uint8_t* current, uint8_t* next,
                                     const int* row_ptr, const int* col_idx, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    int start = row_ptr[i];
    int end = row_ptr[i + 1];
    int degree = end - start;
    
    if (degree == 3) {
        // Rule 110 для 3 соседей
        int pattern = (current[col_idx[start]] << 2) | 
                      (current[col_idx[start+1]] << 1) | 
                      current[col_idx[start+2]];
        next[i] = (110 >> pattern) & 1;
    } else {
        // Fallback к XOR для других степеней
        uint8_t result = 0;
        for (int j = start; j < end; j++) {
            result ^= current[col_idx[j]];
        }
        next[i] = result;
    }
}

/**
 * Основная функция вычисления на CUDA
 * 
 * ЭТАПЫ:
 * 1. Выделение памяти на GPU (cudaMalloc)
 * 2. Копирование данных на GPU (cudaMemcpy Host -> Device)
 * 3. Запуск kernel для каждого шага
 * 4. Копирование результатов обратно (cudaMemcpy Device -> Host)
 * 5. Освобождение памяти GPU (cudaFree)
 * 
 * ЗАМЕР ВРЕМЕНИ:
 * Используются CUDA Events - аппаратные таймеры GPU.
 * Это точнее, чем CPU таймеры, т.к. учитывает реальное время GPU.
 */
ComputeResult compute_cuda(const Graph& graph, const std::vector<uint8_t>& initial_state, 
                           int steps, int feedback_type, const OutputConfig* output_cfg) {
    ComputeResult result;
    int n = graph.num_nodes;
    int num_edges = graph.num_edges;
    
    // ========================================================================
    // ШАГ 1: Выделение памяти на GPU
    // ========================================================================
    // d_ prefix означает device (GPU) память
    uint8_t *d_current, *d_next;
    int *d_row_ptr, *d_col_idx;
    
    CUDA_CHECK(cudaMalloc(&d_current, n * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_next, n * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_row_ptr, (n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_idx, num_edges * sizeof(int)));
    
    // ========================================================================
    // ШАГ 2: Копирование данных на GPU
    // ========================================================================
    // cudaMemcpyHostToDevice: CPU память -> GPU память
    CUDA_CHECK(cudaMemcpy(d_current, initial_state.data(), 
                          n * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_row_ptr, graph.row_ptr.data(), 
                          (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, graph.col_idx.data(), 
                          num_edges * sizeof(int), cudaMemcpyHostToDevice));
    
    // ========================================================================
    // ШАГ 3: Конфигурация запуска
    // ========================================================================
    // block_size: количество потоков в блоке (256 - стандартное значение)
    // grid_size: количество блоков (округление вверх)
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    
    std::vector<uint8_t> host_state(n);
    result.history.push_back(initial_state);
    
    // Формирование выходной последовательности (если задана конфигурация)
    if (output_cfg && !output_cfg->cells.empty()) {
        for (int cell_idx : output_cfg->cells) {
            if (cell_idx >= 0 && cell_idx < n) {
                result.output_sequence.push_back(initial_state[cell_idx]);
            }
        }
    }
    
    // ========================================================================
    // ШАГ 4: Создание CUDA Events для замера времени
    // ========================================================================
    cudaEvent_t start_event, stop_event;
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));
    
    CUDA_CHECK(cudaEventRecord(start_event));  // Начало замера
    
    // ========================================================================
    // ШАГ 5: Основной цикл симуляции
    // ========================================================================
    for (int step = 0; step < steps; step++) {
        // Запуск kernel с конфигурацией <<<grid_size, block_size>>>
        // grid_size блоков по block_size потоков = n потоков всего
        switch (feedback_type) {
            case 0:
                step_kernel_xor<<<grid_size, block_size>>>(
                    d_current, d_next, d_row_ptr, d_col_idx, n);
                break;
            case 1:
                step_kernel_majority<<<grid_size, block_size>>>(
                    d_current, d_next, d_row_ptr, d_col_idx, n);
                break;
            case 2:
                step_kernel_rule110<<<grid_size, block_size>>>(
                    d_current, d_next, d_row_ptr, d_col_idx, n);
                break;
            default:
                step_kernel_xor<<<grid_size, block_size>>>(
                    d_current, d_next, d_row_ptr, d_col_idx, n);
        }
        
        // КРИТИЧЕСКИ ВАЖНО: проверка ошибок запуска kernel
        CUDA_CHECK(cudaGetLastError());
        // КРИТИЧЕСКИ ВАЖНО: синхронизация для корректности результатов
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Обмен указателей (эффективнее чем копирование)
        uint8_t* tmp = d_current;
        d_current = d_next;
        d_next = tmp;
        
        // Копирование для сохранения истории
        CUDA_CHECK(cudaMemcpy(host_state.data(), d_current, 
                              n * sizeof(uint8_t), cudaMemcpyDeviceToHost));
        result.history.push_back(host_state);
        
        // Извлекаем значения для выходной последовательности
        if (output_cfg && !output_cfg->cells.empty()) {
            if ((step + 1) % output_cfg->extract_every_n_steps == 0) {
                for (int cell_idx : output_cfg->cells) {
                    if (cell_idx >= 0 && cell_idx < n) {
                        result.output_sequence.push_back(host_state[cell_idx]);
                    }
                }
            }
        }
    }
    
    // ========================================================================
    // ШАГ 6: Завершение замера времени
    // ========================================================================
    CUDA_CHECK(cudaEventRecord(stop_event));
    CUDA_CHECK(cudaEventSynchronize(stop_event));  // Ожидание завершения GPU
    
    float elapsed_ms;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event));
    result.elapsed_ms = elapsed_ms;
    
    // ========================================================================
    // ШАГ 7: Копирование финального результата
    // ========================================================================
    CUDA_CHECK(cudaMemcpy(host_state.data(), d_current, 
                          n * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    result.final_state = host_state;
    
    // ========================================================================
    // ШАГ 8: Освобождение ресурсов
    // ========================================================================
    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));
    CUDA_CHECK(cudaFree(d_current));
    CUDA_CHECK(cudaFree(d_next));
    CUDA_CHECK(cudaFree(d_row_ptr));
    CUDA_CHECK(cudaFree(d_col_idx));
    
    return result;
}
