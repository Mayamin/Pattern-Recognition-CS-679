#!/usr/bin/python3

import matplotlib.pyplot as plt
import csv

# only run this file through the make file.
# run 'make plot', or 'make run' 

line_start = -3
line_end = 8
increments = .1

data = [ [[], []], [[], []]] 

def decision_boundary_class_1 (W: list, x0: list, x1: list):
	x2 = []

	for i in x1:
		x2.append( -1 * (W[0] / W[1]) * (i - x0[0]) + x0[1])

	return x2

if __name__ == '__main__':

	with open('./data/data.csv', 'r') as file:
		csvreader = csv.reader(file)
		for row in csvreader:
			data[int(row[2])][0].append(float(row[0]))
			data[int(row[2])][1].append(float(row[1]))

	W = []
	x0 = []

	with open('./data/decision_boundary.csv', 'r') as file:
		csvreader = csv.reader(file)
		W.append(float(next(csvreader)[0]))
		W.append(float(next(csvreader)[0]))
		x0.append(float(next(csvreader)[0]))
		x0.append(float(next(csvreader)[0]))

	x1 = [(increments * x + line_start) for x in range( (line_end - line_start) * 10 )]
	x2 = decision_boundary_class_1(W, x0, x1)

	fig1, ax1 = plt.subplots()

	ax1.scatter(data[0][0], data[0][1], c='red', s=.2)
	ax1.scatter(data[1][0], data[1][1], c='blue', s=.2)

	ax1.plot(x1, x2, c='black')

	plt.xlim(-10, 15)
	plt.ylim(-10, 15)

	plt.show()

	fig1.savefig('data/plot.pdf')