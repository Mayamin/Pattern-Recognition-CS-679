PP=g++
CFLAGS=-I /usr/include/eigen3 -fopenmp

all: bayes_classifier test_bayes test_svm

run: test_bayes test_svm

test_bayes: bin/bayes
	rm -rf results/16_20_bayes_output.txt
	touch results/16_20_bayes_output.txt
	./bin/bayes "./GenderDataRowOrder/16_20/trPCA_0" "./GenderDataRowOrder/16_20/TtrPCA_0" "./GenderDataRowOrder/16_20/tsPCA_0" "./GenderDataRowOrder/16_20/TtsPCA_0" "./results/" 30 > results/16_20_bayes_output.txt
	
	rm -rf results/48_60_bayes_output.txt
	touch results/48_60_bayes_output.txt
	./bin/bayes "./GenderDataRowOrder/48_60/trPCA_0" "./GenderDataRowOrder/48_60/TtrPCA_0" "./GenderDataRowOrder/48_60/tsPCA_0" "./GenderDataRowOrder/48_60/TtsPCA_0" "./results/" 30 > results/48_60_bayes_output.txt

test_svm: scripts/16_20_svm.py scripts/48_60_svm.py
	python scripts/16_20_svm.py
	rm -rf ./*.csv

	python scripts/48_60_svm.py
	rm -rf ./*.csv
	rm -rf ./*.csv#

bayes_classifier: src/main.cpp
	$(PP) $(CFLAGS) -o bin/bayes src/main.cpp

clean:
	rm -rf ./bin/*
	rm -rf ./results/*