PP=g++
CFLAGS=-I /usr/include/eigen3 -fopenmp
openCVCflags=`pkg-config --cflags opencv4`
openCVLflags=`pkg-config --libs opencv4`

all: cmp_train

run: cmp_train train test

train: cmp_train
	./bin/train "./data/fa_H/"

test: scripts/experiment.py
	python3 scripts/experiment.py

cmp_train: src/main.cpp
	$(PP) $(CFLAGS) -o bin/train src/main.cpp $(openCVCflags) $(openCVLflags)

clean:
	rm -rf ./bin/*
	rm -rf ./results/*
	rm -rf ./images/*