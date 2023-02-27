#!/usr/bin/python3

import matplotlib.pyplot as plt
import csv

# only run this file through the make file.
# run 'make plot', or 'make run' 

line_start = -3
line_end = 3
increments = .1

guess_min = -8
guess_max = 16

data = [ [[], []], [[], []] ]



def find_guess(a: float, b: float, c: float, chosen_range: list):
	x_list = [(increments * x + chosen_range[0]) for x in range( (chosen_range[1] - chosen_range[0]) * 10 )]

	mini = 10 ** 6
	mini_guess_1 = mini

	for i in x_list:
		checked_val = a * (i ** 2) + b * i + c

		if (i > 0):
			break

		if abs(checked_val) < mini:
			mini = abs(checked_val)
			mini_guess_1 = i

	mini = 10 ** 6
	mini_guess_2 = mini

	for i in x_list:
		if i <= 0:
			continue
		checked_val = a * (i ** 2) + b * i + c
		if abs(checked_val) < mini:
			mini = abs(checked_val)
			mini_guess_2 = i

	return mini_guess_1, mini_guess_2

# Newton's method for assumed: ax^2 + bx + c = f(x)
def newtons_method (a: float, b: float, c: float, guess: float):
	xn = guess

	# shouldn't need to iterate more than 10 times bc this method is awesome ( probably can do less )
	for i in range(10):
		f_x_n = (a * (xn * xn)) + b * xn + c
		f_prime_x_n = 2 * a * xn + b

		xn -= f_x_n / f_prime_x_n

	return xn

def decision_boundary_class_3 (W_0: list, w_0: list, w_0_0:float, W_1: list, w_1: list, w_1_0:float, x1: list):
	x2 = []
	# this is a sideways parabola so there should always be 2 solutions
	x2_prime = []

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

if __name__ == '__main__':

	with open('./data/data.csv', 'r') as file:
		csvreader = csv.reader(file)
		for row in csvreader:
			data[int(row[2])][0].append(float(row[0]))
			data[int(row[2])][1].append(float(row[1]))

	W_0 = []
	w_0 = []
	w_0_0 = 0.0
	W_1 = []
	w_1 = []
	w_1_0 = 0.0

	with open('./data/decision_boundary.csv', 'r') as file:
		csvreader = csv.reader(file)
		
		tmp = next(csvreader)[0].split(' ')
		tmp.remove('')
		tmp.remove('')

		W_0.append( [ float(x) for x in tmp ] )

		tmp = next(csvreader)[0].split(' ')
		tmp.remove('')
		tmp.remove('')

		W_0.append( [ float(x) for x in tmp ] )

		w_0.append(float(next(csvreader)[0]))
		w_0.append(float(next(csvreader)[0]))
		w_0_0 = float(next(csvreader)[0])

		tmp = next(csvreader)[0].split(' ')
		tmp.remove('')
		tmp.remove('')
		tmp.remove('')

		W_1.append( [ float(x) for x in tmp ] )

		tmp = next(csvreader)[0].split(' ')
		tmp.remove('')

		W_1.append( [ float(x) for x in tmp ] )

		w_1.append(float(next(csvreader)[0]))
		w_1.append(float(next(csvreader)[0]))
		w_1_0 = float(next(csvreader)[0])

	x1 = [(increments * x + line_start) for x in range( (line_end - line_start) * 10 )]
	x2, x3 = decision_boundary_class_3(W_0, w_0, w_0_0, W_1, w_1, w_1_0, x1[2:-1])
	fig1, ax1 = plt.subplots()

	ax1.scatter(data[0][0], data[0][1], c='red', s=.2)
	ax1.scatter(data[1][0], data[1][1], c='blue', s=.2)

	ax1.plot(x1[2:-1], x2, c='black')
	ax1.plot(x1[2:-1], x3, c='black')

	plt.show()