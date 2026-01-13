# Makefile для обобщенного клеточного автомата

NVCC = /usr/local/cuda/bin/nvcc
CXX = g++
NVCC_FLAGS = -O3 -std=c++17 -arch=sm_86
CXX_FLAGS = -O3 -std=c++17

TARGET = cellular_automaton
CPU_TARGET = cellular_automaton_cpu

# Сборка с CUDA
all: $(TARGET)

$(TARGET): main.cpp cpu_automaton.cpp cuda_automaton.cu cellular_automaton.h
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) main.cpp cpu_automaton.cpp cuda_automaton.cu

# Сборка только CPU версии
cpu: main.cpp cpu_automaton.cpp cellular_automaton.h
	$(CXX) $(CXX_FLAGS) -DNO_CUDA -o $(CPU_TARGET) main.cpp cpu_automaton.cpp

# Очистка
clean:
	rm -f $(TARGET) $(CPU_TARGET)

.PHONY: all cpu clean
