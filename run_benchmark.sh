#!/bin/bash

echo "Запуск полного бенчмарка..."
echo "Начало: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

START_TIME=$(date +%s)

./cellular_automaton > output_extended.txt 2>&1

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "Завершено: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Общее время выполнения: ${MINUTES} мин ${SECONDS} сек"
echo ""
echo "Результаты сохранены в output_extended.txt"

