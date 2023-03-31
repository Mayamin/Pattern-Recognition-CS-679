PP=g++

all: data

run: data
	@echo "===================================================="
	@echo "\tRunning the compiled file..."
	@echo "===================================================="
	./bin/data 

data: src/box_muller.cc
	$(PP) -o bin/data src/box_muller.cc
