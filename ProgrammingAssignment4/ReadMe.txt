Authors: Tyler Becker, Mayamin Hamid Raha
Date: 05/10/23

General Layout:
- All experiments are done in the main directory. The svm models are trained using the python scripts in the scripts folder, and the bayesian classifiers are made using the c++ files in the src folder.

Execution Instructions:
- Both experiments can be run by using the Makefile included:
"make" 					- compiles and runs the bayes classifier on both data sets, and then runs the svm classifier on both sets as well
"make run" 				- runs the bayes classifier on both data sets, and then runs the svm classifier on both sets as well
"make test_bayes" 		- runs the bayes classifier on both data sets
"make test_svm" 		- then runs the svm classifier on both data sets

File Structure (homogenous):
- bin 					: just the output folder for the compiled binary.
- GenderDataRowOrder	: directory containing the necessary data for each experiment. Has a separate subfolder for the 16-20 and 48-60 images.
- results				: output directory for the classification results of each model.
- scripts				: directory containing the python implementation for testing. Specifically the code for training and testing the SVMs are here.
- src 					: directory with the C++ mcode used for training. Specifically the code for training and testing the Bayes classifiers are here.
