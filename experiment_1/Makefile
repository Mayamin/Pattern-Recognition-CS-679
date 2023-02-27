PP=g++
CFLAGS=-I /usr/lib/eigen-3.4.0

all: experiment_1

run: experiment_1
	@echo "===================================================="
	@echo "\tRunning the compiled file..."
	@echo "===================================================="
	./bin/experiment_1 data/data.csv

experiment_1: src/main.cpp
	$(PP) $(CFLAGS) -o bin/experiment_1 src/main.cpp

clean:
	rm -rf ./bin/*