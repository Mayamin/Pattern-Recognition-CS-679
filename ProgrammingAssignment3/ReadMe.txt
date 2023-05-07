Authors: Tyler Becker, Mayamin Hamid Raha
Date: 04/25/23

General Layout:
- (H|L)_Res_(Class|Intruder): contains all of the necessary files to perform experiments for High Res(H) on Classification and Intruder Detection and the same for Low Res(L).


Execution Instructions:
- All experiments have a makefile which can be run with the following commands:
"make run" - Trains and Tests the Models on the data.
"make train" - Just trains
"make test" - Just tests

File Structure (homogenous):
- bin: just the output folder for the compiled binary.
- data: directory containing the necessary data for each experiment
- images: output directory for the eigen_faces, graphs, etc.
- model: output directory for the parameters of the models.
- scripts: directory containing the python implementation for testing.
- src: directory with the C++ mcode used for training.
