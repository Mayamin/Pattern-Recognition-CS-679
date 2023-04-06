#!/usr/bin/python3
import matplotlib.pyplot as plt
import csv

# globals for plotting the data, should be more general
line_start = -4
line_end = 8
increments = .1

colors = ['g', 'c', 'm', 'y', 'k', 'w']

def find_guess(a: float, b: float, c: float, chosen_range: list):
	x_list = [(increments * x + chosen_range[0]) for x in range( (chosen_range[1] - chosen_range[0]) * 10 )]

	mini = 10 ** 6
	mini_guess_1 = mini

	for i in x_list:
		checked_val = a * (i ** 2) + b * i + c

		if abs(checked_val) < mini:
			mini = abs(checked_val)
			mini_guess_1 = i

	mini = 10 ** 6
	mini_guess_2 = mini

	for i in x_list:
		if i == mini_guess_1:
			continue

		checked_val = a * (i ** 2) + b * i + c
		if abs(checked_val) < mini:
			mini = abs(checked_val)
			mini_guess_2 = i

	if mini_guess_2 < mini_guess_1:
		tmp = mini_guess_1
		mini_guess_1 = mini_guess_2
		mini_guess_2 = tmp

	dist1 = a * (mini_guess_1 ** 2) + b * mini_guess_1 + c
	dist2 = a * (mini_guess_2 ** 2) + b * mini_guess_2 + c

	if dist1 > 1.0:
		mini_guess_1 = 0
	if dist2 > 1.0:
		mini_guess_2 = 0

	return mini_guess_1, mini_guess_2

# Newton's method for assumed: ax^2 + bx + c = f(x)
def newtons_method (a: float, b: float, c: float, guess: float):
	xn = guess

	# shouldn't need to iterate more than 5 times bc this method is awesome ( probably can do less )
	for i in range(5):
		f_x_n = (a * (xn * xn)) + b * xn + c
		f_prime_x_n = 2 * a * xn + b

		xn -= f_x_n / f_prime_x_n

	f_x_n_final = (a * (xn * xn)) + b * xn + c

	if abs(f_x_n_final) > .2:
		xn = 0.0

	return xn

def decision_boundary_class_1 (c_data: list, x1: list):
	x2 = []

	W = c_data[1]
	x0 = c_data[2]

	for i in x1:
		x2.append( -1 * (W[0] / W[1]) * (i - x0[0]) + x0[1])

	return x2

def decision_boundary_class_2 (c_data: list, x1: list):
	x2 = []

	w_0 = c_data[1]
	w_0_0 = c_data[2]
	w_1 = c_data[3]
	w_1_0 = c_data[4]

	b = ( w_1[0] - w_0[0] ) / ( w_0[1] - w_1[1] )
	c = ( w_1_0 - w_0_0 ) / ( w_0[1] - w_1[1] )

	for i in x1:
		x2.append(b * i + c)

	return x2

def decision_boundary_class_3 (c_data: list, x1: list, guess_min: float, guess_max: float):
	# this is an ellipsoid so there should always be 2 solutions
	x2 = []
	x2_prime = []

	W_0 = c_data[1]
	w_0 = c_data[2]
	w_0_0 = c_data[3]
	W_1 = c_data[4]
	w_1 = c_data[5]
	w_1_0 = c_data[6]

	# these are all the constants from xT*(W_0 - W_1)*x + (w_0T - w_1T)*x + (w_0_0 - w_1_0) = 0
	# this is then simplified
	a = W_0[1][1] - W_1[1][1]
	b = W_0[1][0] - W_1[1][0] + W_0[0][1] - W_1[0][1]
	c = w_0[1] - w_1[1]
	e = w_0_0 - w_1_0
	g = W_0[0][0] - W_1[0][0]
	h = w_0[0] - w_1[0]

	for i in x1:
		# this represents all of the terms that are collapsed once we have a value for x1
		# then all we do is solve for the roots of the remaining function with newton's method (couldn't do the algebra)
		d = b * i + c
		f = g * (i ** 2) + h * i + e

		# Newton's Method only works if we have a close enough guess
		guess_1, guess_2 = find_guess( a, d, f, [guess_min, guess_max] )

		# then the method iterates the guess closer and closer
		x2.append( newtons_method( a, d, f, guess_1 ) )

		# get the other solution as well
		x2_prime.append( newtons_method( a, d, f, guess_2 ) )


	return x2, x2_prime

def load_dataset(file_name, data):
	with open(file_name, 'r') as file:
		csvreader = csv.reader(file)
		for row in csvreader:

			data[int(row[2])][0].append(float(row[0]))
			data[int(row[2])][1].append(float(row[1]))

def load_classifier_1(reader, c_data):
	w = []
	x0 = []

	w.append(float(next(reader)[0]))
	w.append(float(next(reader)[0]))
	x0.append(float(next(reader)[0]))
	x0.append(float(next(reader)[0]))

	c_data.append(w)
	c_data.append(x0)

def load_classifier_2(reader, c_data):
	w_0 = []
	w_0_0 = 0.0
	w_1 = []
	w_1_0 = 0.0

	w_0.append(float(next(reader)[0]))
	w_0.append(float(next(reader)[0]))
	w_0_0.append(float(next(reader)[0]))
	w_1.append(float(next(reader)[0]))
	w_1.append(float(next(reader)[0]))
	w_1_0.append(float(next(reader)[0]))

	c_data.append(w_0)
	c_data.append(w_0_0)
	c_data.append(w_1)
	c_data.append(w_1_0)

def load_classifier_3(reader, c_data):
	W_0 = []
	w_0 = []
	w_0_0 = 0.0
	W_1 = []
	w_1 = []
	w_1_0 = 0.0

	tmp = next(reader)[0].split(' ')

	while '' in tmp:
		tmp.remove('')

	W_0.append( [ float(x) for x in tmp ] )

	tmp = next(reader)[0].split(' ')

	while '' in tmp:
		tmp.remove('')

	W_0.append( [ float(x) for x in tmp ] )

	w_0.append(float(next(reader)[0]))
	w_0.append(float(next(reader)[0]))
	w_0_0 = float(next(reader)[0])

	tmp = next(reader)[0].split(' ')

	while '' in tmp:
		tmp.remove('')

	W_1.append( [ float(x) for x in tmp ] )

	tmp = next(reader)[0].split(' ')

	while '' in tmp:
		tmp.remove('')

	W_1.append( [ float(x) for x in tmp ] )

	w_1.append(float(next(reader)[0]))
	w_1.append(float(next(reader)[0]))
	w_1_0 = float(next(reader)[0])

	c_data.append(W_0)
	c_data.append(w_0)
	c_data.append(w_0_0)
	c_data.append(W_1)
	c_data.append(w_1)
	c_data.append(w_1_0)

import_classes = [load_classifier_1, load_classifier_2, load_classifier_3]

def load_classifier(file_name, c_data):
	with open(file_name, 'r') as file:
		csvreader = csv.reader(file)

		case_num = int(next(csvreader)[0])

		c_data.append(case_num)

		import_classes[case_num - 1](csvreader, c_data)

def plot_data(ax, data):
	ax.scatter(data[0][0], data[0][1], c='red', s=.2)
	ax.scatter(data[1][0], data[1][1], c='blue', s=.2)

def plot_decision_boundary(ax, c_data, color, line_label, guess):

	x1 = [(increments * x + line_start) for x in range( (line_end - line_start) * 10 )]

	case_num = c_data[0]

	x2 = []
	x3 = []

	if case_num == 3:
		x2, x3 = decision_boundary_class_3(c_data, x1, guess[0], guess[1])
	elif case_num == 2:
		x2 = decision_boundary_class_2(c_data, x1)
	else:
		x2 = decision_boundary_class_1(c_data, x1)

	ax.plot(x1, x2, c=color, label=line_label)
	
	if case_num == 3:
		ax.plot(x1, x3, c=color)

def plot_classifiers_and_data(ax, data, c_data_all, labels, guess):

	plot_data(ax, data)

	# if len(c_data_all) <= len(labels) or len(c_data_all) <= len(colors):
	# 	print(f"Hi{len(labels)}")

	for i in range(len(c_data_all)):
		plot_decision_boundary(ax, c_data_all[i], colors[i], labels[i], guess)