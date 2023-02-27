PP=g++
CFLAGS=-I /usr/lib/eigen-3.4.0

all: experiment_2

run: experiment_2
	@echo "===================================================="
	@echo "\tRunning the compiled file..."
	@echo "===================================================="
	./bin/experiment_2 data/data.csv
	@echo "===================================================="
	@echo "\tPlotting Data..."
	@echo "===================================================="
	python3 scripts/plot.py

experiment_2: src/main.cpp
	$(PP) $(CFLAGS) -o bin/experiment_2 src/main.cpp

plot: data/data.csv data/decision_boundary.csv
	@echo "===================================================="
	@echo "\tPlotting Data..."
	@echo "===================================================="
	python3 scripts/plot.py

clean:
	rm -rf ./bin/*