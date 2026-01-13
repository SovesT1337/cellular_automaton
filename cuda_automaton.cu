/**
 * CUDA реализация клеточного автомата
 */

#include "cellular_automaton.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// CUDA Kernel: XOR функция обратной связи
__global__ void step_kernel_xor(const uint8_t* current, uint8_t* next,
                                 const int* row_ptr, const int* col_idx, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    int start = row_ptr[i];
    int end = row_ptr[i + 1];
    
    uint8_t result = 0;
    for (int j = start; j < end; j++) {
        result ^= current[col_idx[j]];
    }
    next[i] = result;
}

// CUDA Kernel: Majority функция обратной связи
__global__ void step_kernel_majority(const uint8_t* current, uint8_t* next,
                                      const int* row_ptr, const int* col_idx, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    int start = row_ptr[i];
    int end = row_ptr[i + 1];
    int degree = end - start;
    
    int sum = 0;
    for (int j = start; j < end; j++) {
        sum += current[col_idx[j]];
    }
    
    next[i] = (sum > degree / 2) ? 1 : 0;
}

// Основная функция вычисления на CUDA
ComputeResult compute_cuda(const Graph& graph, const std::vector<uint8_t>& initial_state, 
                           int steps, int feedback_type, const OutputConfig* output_cfg) {
    ComputeResult result;
    int n = graph.num_nodes;
    int num_edges = graph.num_edges;
    
    // Выделение памяти на GPU
    uint8_t *d_current, *d_next;
    int *d_row_ptr, *d_col_idx;
    
    CUDA_CHECK(cudaMalloc(&d_current, n * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_next, n * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_row_ptr, (n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_idx, num_edges * sizeof(int)));
    
    // Создание CUDA Events для замера времени
    cudaEvent_t start_event, stop_event;
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));
    
    CUDA_CHECK(cudaEventRecord(start_event));
    
    // Копирование данных на GPU
    CUDA_CHECK(cudaMemcpy(d_current, initial_state.data(), 
                          n * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_row_ptr, graph.row_ptr.data(), 
                          (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, graph.col_idx.data(), 
                          num_edges * sizeof(int), cudaMemcpyHostToDevice));
    
    // Конфигурация запуска
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    
    std::vector<uint8_t> host_state(n);
    result.history.push_back(initial_state);
    
    if (output_cfg && !output_cfg->cells.empty()) {
        for (int cell_idx : output_cfg->cells) {
            if (cell_idx >= 0 && cell_idx < n) {
                result.output_sequence.push_back(initial_state[cell_idx]);
            }
        }
    }
    
    // Основной цикл симуляции
    for (int step = 0; step < steps; step++) {
        switch (feedback_type) {
            case 0:
                step_kernel_xor<<<grid_size, block_size>>>(
                    d_current, d_next, d_row_ptr, d_col_idx, n);
                break;
            case 1:
                step_kernel_majority<<<grid_size, block_size>>>(
                    d_current, d_next, d_row_ptr, d_col_idx, n);
                break;
            default:
                step_kernel_xor<<<grid_size, block_size>>>(
                    d_current, d_next, d_row_ptr, d_col_idx, n);
        }
        
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        uint8_t* tmp = d_current;
        d_current = d_next;
        d_next = tmp;
        
        CUDA_CHECK(cudaMemcpy(host_state.data(), d_current, 
                              n * sizeof(uint8_t), cudaMemcpyDeviceToHost));
        result.history.push_back(host_state);
        
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
    
    // Копирование финального результата
    CUDA_CHECK(cudaMemcpy(host_state.data(), d_current, 
                          n * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    result.final_state = host_state;
    
    // Завершение замера времени
    CUDA_CHECK(cudaEventRecord(stop_event));
    CUDA_CHECK(cudaEventSynchronize(stop_event));
    
    float elapsed_ms;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event));
    result.elapsed_ms = elapsed_ms;
    
    // Освобождение ресурсов
    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));
    CUDA_CHECK(cudaFree(d_current));
    CUDA_CHECK(cudaFree(d_next));
    CUDA_CHECK(cudaFree(d_row_ptr));
    CUDA_CHECK(cudaFree(d_col_idx));
    
    return result;
}
