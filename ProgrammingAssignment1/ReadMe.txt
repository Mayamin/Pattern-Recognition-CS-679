Authors: Tyler Becker, Mayamin Hamid Raha
Date: 03/01/23

General Layout:
- Dataset Generation: contains all of the necesary files to generate the data used in the experiments.
- Experiment #: contains all of the necessary files to perform experiment #.

NOTE: Experiments 3 and 4 do not require different processes to run and were therefore grouped together
in a single folder. The files contained within automatically perform both experiments.

Execution Instructions:
- Dataset Generation: The file must first be compiled, and then can be executed in the normal way. Instructions for linux are shown below:
"g++ box_muller.cc -o box_muller"
"./box_muller"

	This should generate both datasets and save them in the folder.

- Experiment 1 and 2: These utilize makefiles so the following command will perform the experiment and plot the results automatically:
"make run"
	Similarly the following instruction will plot already created results without running:
"make plot"

- Experiment 3 and 4: As described above these are run together, they must be run from the folder Experiments 3 and 4 with the following
	command:
"python euclidean_classifier.py"

File Structure:
- DatasetGeneration: All files are in the root directory and are expected to be run from there. The data files that Experiment 3 and 4
	require are in this folder as well.

- Experiment 1 and 2: the subfolder src contains the main cpp file that is compiled, the include subfolder is empty, the data is kept in
	data folder and the bin contains the binary after it is compiled. The makefile is the intended way to compile and run the experiments.

- Experiments 3 and 4: the root directory contains only the python file which is expected to be run from the same root directory, it may
	not give the expected results or not run at all if run from a different directory.