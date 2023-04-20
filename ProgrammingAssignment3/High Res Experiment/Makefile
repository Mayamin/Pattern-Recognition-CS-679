PP=g++
CFLAGS=-I /usr/include/eigen3
openCVCflags=`pkg-config --cflags opencv4`
openCVLflags=`pkg-config --libs opencv4`

all: experiment

run: cmp_train train plot

train: cmp_train
	./bin/train "./data/fa_H/"

test: scripts/main.py
	echo "Hello world!"

cmp_train: src/main.cpp
	$(PP) $(CFLAGS) -o bin/train src/main.cpp $(openCVCflags) $(openCVLflags)

plot: data/dataset_A.csv results/experiment_1_0_results.csv
	@echo "============================================================================================="
	@echo "\tPlotting Data..."
	@echo "============================================================================================="
# 	python3 scripts/plot.py

clean:
	rm -rf ./bin/*
	rm -rf ./results/*
	rm -rf ./images/*