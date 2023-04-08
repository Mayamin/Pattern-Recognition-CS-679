Authors: Tyler Becker, Mayamin Hamid Raha
Date: 04/07/23

General Layout:
- Experiment #: contains all of the necessary files to perform experiment #.

NOTE: Experiments 1 and 2 do not require different processes to run and were therefore grouped together
in a single folder. The files contained within automatically perform both experiments.

Execution Instructions:
- Experiment 1 and 2: These utilize a makefile so the following command will perform the experiment and plot the results automatically:
"make run"
	Similarly the following instruction will plot already created results without running:
"make plot"

- Experiment 3: the python executable is in the root directory of the experiment 3 folder and can be run with the following command:
"python Skin_color_detector.py"

File Structure:
- Experiment 1 and 2: the subfolder src contains the main cpp file that is compiled, the include subfolder is empty, the data is kept in
	data folder and the bin contains the binary after it is compiled. The makefile is the intended way to compile and run the experiments.
	The results and images folder will be where the results are output as well as the graphs saved. 

- Experiments 3 and 4: the root directory contains the python file which is expected to be run from the same root directory as well as the
	outputted ROC curves that are shown in the submitted report. The images are output into the folder structure that starts in the root 
	directory.