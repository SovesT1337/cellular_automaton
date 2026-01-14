/**
 * ====================================================================
 * CUDA РЕАЛИЗАЦИЯ КЛЕТОЧНОГО АВТОМАТА
 * ====================================================================
 * 
 * CUDA (Compute Unified Device Architecture) - это технология NVIDIA
 * для параллельных вычислений на графических процессорах (GPU).
 * 
 * Основные преимущества:
 * - Массивный параллелизм: тысячи потоков выполняются одновременно
 * - Высокая пропускная способность памяти
 * - Ускорение до 100x на больших задачах
 * 
 * Архитектура:
 * - Kernel (ядро) - функция, выполняемая параллельно на GPU
 * - Thread (нить) - минимальная единица выполнения
 * - Block (блок) - группа нитей, выполняемых на одном мультипроцессоре
 * - Grid (сетка) - набор блоков для решения всей задачи
 * 
 * В нашей задаче:
 * - Каждая нить обрабатывает один узел графа
 * - Все узлы обновляются параллельно
 */

#include "cellular_automaton.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>

// ====================================================================
// МАКРОС ДЛЯ ПРОВЕРКИ ОШИБОК CUDA
// ====================================================================
/**
 * CUDA функции возвращают код ошибки cudaError_t.
 * Этот макрос автоматически проверяет код ошибки и выводит
 * информативное сообщение в случае ошибки.
 * 
 * Использование:
 *   CUDA_CHECK(cudaMalloc(&ptr, size));
 *   CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
 * 
 * Если операция завершилась с ошибкой, программа выведет:
 * "CUDA error at file.cu:123: out of memory"
 * и завершится с exit(1).
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

// ====================================================================
// CUDA KERNEL: XOR ФУНКЦИЯ ОБРАТНОЙ СВЯЗИ
// ====================================================================
/**
 * __global__ - это спецификатор CUDA, означающий:
 * - Функция выполняется на GPU (device)
 * - Вызывается с CPU (host)
 * - Может запускаться параллельно тысячами нитей
 * 
 * Принцип работы:
 * 1. Каждая нить вычисляет свой глобальный индекс i
 * 2. Нить i обрабатывает узел i графа
 * 3. Извлекает состояния соседей из current
 * 4. Вычисляет XOR всех соседей
 * 5. Записывает результат в next[i]
 * 
 * Организация нитей:
 * - blockIdx.x - индекс блока в сетке
 * - blockDim.x - количество нитей в блоке (обычно 256 или 512)
 * - threadIdx.x - индекс нити внутри блока
 * - Глобальный индекс: i = blockIdx.x * blockDim.x + threadIdx.x
 * 
 * Параметры:
 * - current: текущее состояние всех узлов (на GPU)
 * - next: буфер для нового состояния (на GPU)
 * - row_ptr, col_idx: граф в CSR формате (на GPU)
 * - n: количество узлов
 */
__global__ void step_kernel_xor(const uint8_t* current, uint8_t* next,
                                 const int* row_ptr, const int* col_idx, int n) {
    // Вычисляем глобальный индекс нити
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Проверка границ: если индекс >= n, эта нить лишняя
    // (количество нитей часто не кратно количеству узлов)
    if (i >= n) return;
    
    // Извлекаем границы списка соседей из CSR представления
    int start = row_ptr[i];      // Начало списка соседей узла i
    int end = row_ptr[i + 1];    // Конец списка соседей узла i
    
    // Вычисляем XOR всех соседей
    uint8_t result = 0;
    for (int j = start; j < end; j++) {
        result ^= current[col_idx[j]];  // XOR с состоянием соседа
    }
    
    // Записываем результат в выходной буфер
    next[i] = result;
}

// ====================================================================
// CUDA KERNEL: MAJORITY ФУНКЦИЯ ОБРАТНОЙ СВЯЗИ
// ====================================================================
/**
 * Реализует правило большинства на GPU.
 * 
 * Алгоритм:
 * 1. Подсчитываем количество единиц среди соседей (sum)
 * 2. Если sum > degree/2, результат = 1, иначе 0
 * 
 * Отличия от XOR kernel:
 * - Используется сложение вместо XOR
 * - Требуется дополнительное сравнение
 * - Нелинейная операция (сложнее для анализа)
 * 
 * Параметры: те же, что и у step_kernel_xor
 */
__global__ void step_kernel_majority(const uint8_t* current, uint8_t* next,
                                      const int* row_ptr, const int* col_idx, int n) {
    // Вычисляем глобальный индекс нити
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Проверка границ
    if (i >= n) return;
    
    // Извлекаем границы списка соседей
    int start = row_ptr[i];
    int end = row_ptr[i + 1];
    int degree = end - start;  // Количество соседей
    
    // Подсчитываем количество единиц среди соседей
    int sum = 0;
    for (int j = start; j < end; j++) {
        sum += current[col_idx[j]];
    }
    
    // Применяем правило большинства
    // Если больше половины соседей имеют значение 1, результат = 1
    next[i] = (sum > degree / 2) ? 1 : 0;
}

// ====================================================================
// ОСНОВНАЯ ФУНКЦИЯ ВЫЧИСЛЕНИЯ НА CUDA
// ====================================================================
/**
 * Выполняет симуляцию клеточного автомата на GPU.
 * 
 * Этапы работы:
 * 1. Выделение памяти на GPU
 * 2. Копирование данных CPU → GPU
 * 3. Запуск CUDA kernels (параллельные вычисления)
 * 4. Копирование результатов GPU → CPU
 * 5. Освобождение памяти GPU
 * 
 * Архитектура памяти GPU:
 * - Global Memory: основная память GPU (большая, но медленная)
 * - Shared Memory: быстрая память, разделяемая внутри блока
 * - Registers: самая быстрая память, доступна только одной нити
 * 
 * В данной реализации мы используем Global Memory для простоты.
 * Для больших графов можно оптимизировать, используя Shared Memory.
 */
ComputeResult compute_cuda(const Graph& graph, const std::vector<uint8_t>& initial_state, 
                           int steps, int feedback_type, const OutputConfig* output_cfg) {
    ComputeResult result;
    int n = graph.num_nodes;
    int num_edges = graph.num_edges;
    
    // ====================================================================
    // ВЫДЕЛЕНИЕ ПАМЯТИ НА GPU
    // ====================================================================
    /**
     * Префикс d_ означает "device" (GPU), в отличие от host (CPU).
     * 
     * Выделяемая память:
     * - d_current, d_next: буферы для состояний узлов (двойная буферизация)
     * - d_row_ptr, d_col_idx: граф в CSR формате
     * 
     * cudaMalloc похож на malloc, но выделяет память на GPU.
     * Эта память недоступна напрямую с CPU!
     */
    uint8_t *d_current, *d_next;  // Буферы состояний на GPU
    int *d_row_ptr, *d_col_idx;   // Граф на GPU
    
    CUDA_CHECK(cudaMalloc(&d_current, n * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_next, n * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_row_ptr, (n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_idx, num_edges * sizeof(int)));
    
    // ====================================================================
    // СОЗДАНИЕ CUDA EVENTS ДЛЯ ЗАМЕРА ВРЕМЕНИ
    // ====================================================================
    /**
     * CUDA Events - это механизм для точного измерения времени
     * выполнения операций на GPU.
     * 
     * Преимущества перед std::chrono:
     * - Измеряет время непосредственно на GPU
     * - Учитывает только GPU операции (без учёта CPU overhead)
     * - Высокая точность
     */
    cudaEvent_t start_event, stop_event;
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));
    
    // Записываем событие начала
    CUDA_CHECK(cudaEventRecord(start_event));
    
    // ====================================================================
    // КОПИРОВАНИЕ ДАННЫХ CPU → GPU
    // ====================================================================
    /**
     * cudaMemcpy передаёт данные между CPU и GPU.
     * 
     * Направления передачи:
     * - cudaMemcpyHostToDevice: CPU → GPU (используется здесь)
     * - cudaMemcpyDeviceToHost: GPU → CPU
     * - cudaMemcpyDeviceToDevice: GPU → GPU
     * 
     * Это медленная операция (ограничена пропускной способностью PCIe шины).
     * Для оптимизации можно использовать:
     * - cudaMemcpyAsync (асинхронное копирование)
     * - Pinned Memory (ускоряет передачу)
     * - Уменьшение количества копирований
     */
    CUDA_CHECK(cudaMemcpy(d_current, initial_state.data(), 
                          n * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_row_ptr, graph.row_ptr.data(), 
                          (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, graph.col_idx.data(), 
                          num_edges * sizeof(int), cudaMemcpyHostToDevice));
    
    // ====================================================================
    // КОНФИГУРАЦИЯ ЗАПУСКА KERNEL
    // ====================================================================
    /**
     * Определяем организацию нитей для выполнения на GPU:
     * 
     * - block_size (размер блока): количество нитей в одном блоке
     *   Обычно 128, 256 или 512. Оптимальное значение зависит от GPU.
     *   
     * - grid_size (размер сетки): количество блоков
     *   Вычисляется так, чтобы покрыть все n узлов.
     *   Формула: grid_size = ceil(n / block_size)
     * 
     * Пример: для n=1000 узлов и block_size=256:
     *   grid_size = (1000 + 255) / 256 = 4 блока
     *   Всего нитей: 4 × 256 = 1024 (некоторые будут лишние)
     */
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    
    // Буфер на CPU для получения результатов с GPU
    std::vector<uint8_t> host_state(n);
    
    // Сохраняем начальное состояние
    result.history.push_back(initial_state);
    
    // Извлекаем начальные значения для выходной последовательности
    if (output_cfg && !output_cfg->cells.empty()) {
        for (int cell_idx : output_cfg->cells) {
            if (cell_idx >= 0 && cell_idx < n) {
                result.output_sequence.push_back(initial_state[cell_idx]);
            }
        }
    }
    
    // ====================================================================
    // ОСНОВНОЙ ЦИКЛ СИМУЛЯЦИИ НА GPU
    // ====================================================================
    /**
     * На каждом шаге:
     * 1. Запускаем kernel на GPU
     * 2. Ждём завершения
     * 3. Меняем буферы местами (swap)
     * 4. Копируем результат обратно на CPU (для истории)
     * 5. Извлекаем выходную последовательность
     * 
     * Узкое место: копирование GPU → CPU на каждом шаге.
     * Для оптимизации можно копировать только финальное состояние.
     */
    for (int step = 0; step < steps; step++) {
        
        // ====================================================================
        // ЗАПУСК CUDA KERNEL
        // ====================================================================
        /**
         * Синтаксис запуска kernel:
         *   kernel_name<<<grid_size, block_size>>>(параметры);
         * 
         * <<<grid_size, block_size>>> - это execution configuration:
         * - grid_size блоков
         * - block_size нитей в каждом блоке
         * 
         * Запуск kernel асинхронный: CPU продолжает выполнение,
         * пока GPU работает. Нужен cudaDeviceSynchronize() для ожидания.
         */
        if (feedback_type == 1) {
            step_kernel_majority<<<grid_size, block_size>>>(d_current, d_next, d_row_ptr, d_col_idx, n);
        } else {
            step_kernel_xor<<<grid_size, block_size>>>(d_current, d_next, d_row_ptr, d_col_idx, n);
        }
        
        // Проверка на ошибки во время запуска kernel
        CUDA_CHECK(cudaGetLastError());
        
        // ====================================================================
        // СИНХРОНИЗАЦИЯ
        // ====================================================================
        /**
         * cudaDeviceSynchronize() блокирует CPU до завершения всех
         * операций GPU. Это необходимо, потому что:
         * 1. Запуск kernel асинхронный
         * 2. Нам нужен результат для следующего шага
         * 3. Мы хотим скопировать результат обратно на CPU
         */
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // ====================================================================
        // SWAP БУФЕРОВ (обмен указателями)
        // ====================================================================
        /**
         * Меняем роли буферов:
         * - d_current (старое) становится d_next (будет перезаписано)
         * - d_next (новое) становится d_current (текущее состояние)
         * 
         * Это быстрая операция O(1), не требует копирования данных.
         */
        uint8_t* tmp = d_current;
        d_current = d_next;
        d_next = tmp;
        
        // ====================================================================
        // КОПИРОВАНИЕ РЕЗУЛЬТАТА ОБРАТНО НА CPU
        // ====================================================================
        /**
         * Копируем текущее состояние GPU → CPU для:
         * 1. Сохранения в историю
         * 2. Извлечения выходной последовательности
         * 
         * Это медленная операция (узкое место производительности).
         * Если не нужна полная история, можно копировать только
         * финальное состояние после всех шагов.
         */
        CUDA_CHECK(cudaMemcpy(host_state.data(), d_current, 
                              n * sizeof(uint8_t), cudaMemcpyDeviceToHost));
        result.history.push_back(host_state);
        
        // Извлечение выходной последовательности (если настроено)
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
    
    // ====================================================================
    // ФИНАЛИЗАЦИЯ И ОЧИСТКА
    // ====================================================================
    
    // Копируем финальное состояние обратно на CPU
    CUDA_CHECK(cudaMemcpy(host_state.data(), d_current, 
                          n * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    result.final_state = host_state;
    
    // ====================================================================
    // ЗАВЕРШЕНИЕ ЗАМЕРА ВРЕМЕНИ
    // ====================================================================
    /**
     * Записываем событие окончания и вычисляем прошедшее время.
     * cudaEventElapsedTime возвращает время в миллисекундах
     * между двумя событиями.
     */
    CUDA_CHECK(cudaEventRecord(stop_event));
    CUDA_CHECK(cudaEventSynchronize(stop_event));
    
    float elapsed_ms;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event));
    result.elapsed_ms = elapsed_ms;
    
    // ====================================================================
    // ОСВОБОЖДЕНИЕ РЕСУРСОВ GPU
    // ====================================================================
    /**
     * Важно освободить всю выделенную память на GPU!
     * Утечки памяти на GPU могут привести к:
     * - Исчерпанию памяти GPU
     * - Снижению производительности
     * - Невозможности запуска других CUDA программ
     * 
     * cudaFree - аналог free() для памяти GPU
     */
    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));
    CUDA_CHECK(cudaFree(d_current));
    CUDA_CHECK(cudaFree(d_next));
    CUDA_CHECK(cudaFree(d_row_ptr));
    CUDA_CHECK(cudaFree(d_col_idx));
    
    return result;
}
