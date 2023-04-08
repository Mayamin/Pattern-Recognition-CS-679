#!/usr/bin/python3

import matplotlib.pyplot as plt
import csv
from py_include import util

# only run this file through the make file.
# run 'make plot', or 'make run' 

if __name__ == '__main__':
	data_A = [ [[], []], [[], []]] 
	data_B = [ [[], []], [[], []]] 

	c_data_all = [ [], [] ]
	labels = [ [], [] ]

	guess = [ [-16, 16], [-6, 6]]

	util.load_dataset('data/dataset_A.csv', data_A)
	util.load_dataset('data/dataset_B.csv', data_B)

	for i in range(1, 3):
		for j in range(5):
			n_classifier = []

			util.load_classifier('results/experiment_' + str(i) + '_' + str(j) + '_results.csv', n_classifier)

			labels[i - 1].append('experiment_' + str(i) + '_' + str(j))

			c_data_all[i - 1].append(n_classifier)

	fig1, ax1 = plt.subplots()

	# plot the data and all 5 classifiers on 1 graph
	util.plot_classifiers_and_data(ax1, data_A, c_data_all[0], labels[0], guess[0])

	fig2, ax2 = plt.subplots()
	util.plot_classifiers_and_data(ax2, data_B, c_data_all[1], labels[1], guess[1])

	ax1.legend()
	ax2.legend()

	plt.show()

	fig1.savefig('images/dataset_A_plot.pdf', format='pdf')
	fig2.savefig('images/dataset_B_plot.pdf', format='pdf')