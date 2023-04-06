PP=g++
CFLAGS=-I /usr/include/eigen3

all: experiment

run: experiment run_cpp plot

run_cpp: bin/experiment
	@echo "============================================================================================="
	@echo "\tStarting Experiment 1..."
	@echo "============================================================================================="
	./bin/experiment data/dataset_A.csv
	@echo "============================================================================================="
	@echo "\tStarting Experiment 2..."
	@echo "============================================================================================="
	./bin/experiment data/dataset_B.csv

experiment: src/main.cpp
	$(PP) $(CFLAGS) -o bin/experiment src/main.cpp

plot: data/dataset_A.csv results/experiment_1_0_results.csv
	@echo "============================================================================================="
	@echo "\tPlotting Data..."
	@echo "============================================================================================="
	python3 scripts/plot.py

clean:
	rm -rf ./bin/*
	rm -rf ./results/*
	rm -rf ./images/*