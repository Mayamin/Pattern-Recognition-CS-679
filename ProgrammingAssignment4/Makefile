PP=g++
CFLAGS=-I /usr/include/eigen3 -fopenmp

all: bayes_classifier train_test

run: train_test

train_test: bin/bayes
	./bin/bayes "./GenderDataRowOrder/16_20/trPCA_0" "./GenderDataRowOrder/16_20/TtrPCA_0" "./GenderDataRowOrder/16_20/tsPCA_0" "./GenderDataRowOrder/16_20/TtsPCA_0" "./results/" 30

	./bin/bayes "./GenderDataRowOrder/48_60/trPCA_0" "./GenderDataRowOrder/48_60/TtrPCA_0" "./GenderDataRowOrder/48_60/tsPCA_0" "./GenderDataRowOrder/48_60/TtsPCA_0" "./results/" 30

bayes_classifier: src/main.cpp
	$(PP) $(CFLAGS) -o bin/bayes src/main.cpp

clean:
	rm -rf ./bin/*
	rm -rf ./results/*