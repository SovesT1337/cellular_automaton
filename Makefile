# Makefile for Cellular Automaton CPU/CUDA benchmark

NVCC = /usr/local/cuda-13.1/bin/nvcc
CXX = g++
# RTX 3060 compute capability 8.6 (Ampere)
NVCC_FLAGS = -O3 -std=c++17 -arch=sm_86 --ptxas-options=-v
CXX_FLAGS = -O3 -std=c++17

TARGET = cellular_automaton

all: $(TARGET)

$(TARGET): main.cpp cpu_automaton.cpp cuda_automaton.cu cellular_automaton.h
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) main.cpp cpu_automaton.cpp cuda_automaton.cu

# CPU-only версия (без CUDA)
cpu_only: main_cpu.cpp cpu_automaton.cpp cellular_automaton.h
	$(CXX) $(CXX_FLAGS) -o cellular_automaton_cpu main_cpu.cpp cpu_automaton.cpp -DNO_CUDA

clean:
	rm -f $(TARGET) cellular_automaton_cpu

.PHONY: all clean cpu_only

