#include <iostream>
#include <math.h>
#include <vector>
#include <bits/stdc++.h>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

MatrixXf W;
VectorXf x_0;

// algo taken from here: https://www.heikohoffmann.de/htmlthesis/node134.html
// using this because we are taking the average over a large set of floats and therefore have a high change of overflow
// this way there is no overflow essentially as the value simply iterates toward the average instead of the average
// being calculated after a very long summation
VectorXf estimate_mean(MatrixXf inp)
{
	VectorXf average = inp.col(0);

	for (int i = 1; i < inp.cols(); i++)
	{
		VectorXf x = inp.col(i);
		average += (x - average) / i;
	}

	return average;
}

// estimate taken from here: https://en.wikipedia.org/wiki/Estimation_of_covariance_matrices
MatrixXf estimate_cov_matrix(MatrixXf points, VectorXf mean)
{
	MatrixXf Q = MatrixXf::Zero(mean.size() , mean.size());

	for (int i = 0; i < points.cols(); i++)
	{
		VectorXf x = points.col(i);
		Q += ( (x - mean) * (x - mean).transpose() ) ;
	}

	return Q / (points.cols() - 1);
}

MatrixXf matricize (vector<Vector2f> inp)
{
	MatrixXf to_ret(2, inp.size());

	for (int i = 0; i < inp.size(); i++)
	{
		to_ret(0, i) = inp[i][0];
		to_ret(1, i) = inp[i][1];
	}

	return to_ret;	
}

vector<MatrixXf> load_from_file (string file_name)
{
	vector<MatrixXf> data;

	ifstream in_file;
	string temp;

	in_file.open("./" + file_name, ios::in);

	vector<Vector2f> data_set_i;
	int cur_data_set = 0;

	while(in_file >> temp)
	{
		string data_val = "";

		Vector2f data_line;
		int j = 0;

		//cout << temp << endl;

		for (int i = 0 ; i < temp.length(); i++)
		{
			//cout << temp[i];
			if (('0' <= temp[i] && temp[i] <= '9') || temp[i] == '.' || temp[i] == '-')
			{
				data_val += temp[i];
			}
			else if (',' == temp[i])
			{
				//cout << "== " << data_val << endl;
				data_line[j++] = stof(data_val);
				
				//if (j == 1)
				//	cout << data_line << endl;

				data_val = "";
			}
		}

		if (stoi(data_val) == cur_data_set)
		{
			data_set_i.push_back(data_line);
		}
		else
		{
			data.push_back(matricize(data_set_i));

			cur_data_set = stoi(data_val);

			data_set_i.clear();
			data_set_i.push_back(data_line);

		}

	}
	
	data.push_back(matricize(data_set_i));

	in_file.close();

	//cout << data.size() << endl;

	return data;
};

// leveraging my knowledge that there are 2 classes
VectorXf classify_case_1 (MatrixXf to_classify, vector<VectorXf> mean_vector, float variance, VectorXf prior_probabilities)
{
	VectorXf classifications(to_classify.cols());

	for ( int i = 0 ; i < to_classify.cols() ; i++ )
	{
		VectorXf x = to_classify.col(i);

		// this already returns a scalar so I just take the 0th element to make cpp happy
		float class_0_score = -1 * ( pow( ( x - mean_vector[0] ).norm(), 2 ) ) / (2 * variance ) + log(prior_probabilities[0]);
		float class_1_score = -1 * ( pow( ( x - mean_vector[1] ).norm(), 2 ) ) / (2 * variance ) + log(prior_probabilities[1]);

		if (class_0_score > class_1_score)
			classifications[i] = 0;
		else
			classifications[i] = 1;
	}

	return classifications;
}

void compute_decision_boundary (vector<VectorXf> mean_vector, float standard_deviation, VectorXf prior_probabilities)
{
	MatrixXf W = (mean_vector[1] - mean_vector[0]);
	VectorXf x0 = (.5) * (mean_vector[1] + mean_vector[0]) 
					+ ( pow(standard_deviation, 2) / ( (mean_vector[1] - mean_vector[0]).transpose() * (mean_vector[1] - mean_vector[0]) ) )
					  * log(prior_probabilities[1] / prior_probabilities[0]) 
					  * (mean_vector[1] - mean_vector[0]); 

	ofstream o_file;

	o_file.open("data/decision_boundary.csv");

	o_file << W << endl;
	o_file << x0 << endl;

	o_file.close();
}

double get_classification_error(int ground_truth, VectorXf classifications)
{
	int num_incorrect = 0;

	for (int i = 0; i < classifications.size(); i++)
		if (classifications[i] != ground_truth)
			num_incorrect++;

	return ((float) num_incorrect) / classifications.size();
}

double compute_bhattacharyya ( vector<VectorXf> mean_vector, vector<MatrixXf> covariance_matrix, VectorXf prior_probabilities)
{
	// assumption for Bhattacharrya
	double beta = .5;

	double interm_0 = ( beta * ( 1 - beta ) ) / 2;
	MatrixXf interm_1 = (mean_vector[0] - mean_vector[1]).transpose();
	MatrixXf interm_2 = ( ( 1 - beta ) * covariance_matrix[0] + beta * covariance_matrix[1]).inverse();

	//again making cpp happy
	double first_term = ( interm_0 * ( interm_1 * ( interm_2 * interm_1.transpose() ) )(0, 0) );

	double top = ( (1 - beta) * covariance_matrix[0] + beta * covariance_matrix[1] ).determinant();
	double bottom = pow( covariance_matrix[0].determinant(), 1 - beta) * pow(covariance_matrix[1].determinant(), beta);

	double second_term = top / bottom;

	double k_beta = first_term + second_term;

	return pow(prior_probabilities[0], beta) * pow(prior_probabilities[1], 1 - beta) * exp(-1 * k_beta);
}

int main (int argc, char** argv)
{
	srand(time(0));

	if (argc < 2)
		return -1;

	cout << "Loading data from: \"" << argv[1] << "\"..." << endl;

	vector<MatrixXf> test_data = load_from_file(argv[1]);

	vector<VectorXf> mean_vector;
	vector<MatrixXf> covariance_matrix;
	VectorXf prior_probabilities(test_data.size());

	cout << "Computing estimates... " << endl;

	int total_n = 0;

	for (int i = 0; i < test_data.size() ; i++)
		total_n += test_data[i].cols();


	for (int i = 0; i < test_data.size(); i++)
	{
		VectorXf mean_i = estimate_mean(test_data[i]);
		MatrixXf covar_matrix_i = estimate_cov_matrix(test_data[i], mean_i);
		float probability_i = ( (float) test_data[i].cols() ) / (total_n);

		mean_vector.push_back(mean_i);
		covariance_matrix.push_back(covar_matrix_i);
		prior_probabilities[i] = probability_i;

		cout << "Class " << i << " Mean: " << endl;
		cout << mean_i << endl;
		cout << "Class " << i << " Covariance Matrix: " << endl;
		cout << covar_matrix_i << endl;
	}

	cout << "Class prior probability vector: " << endl;
	cout << prior_probabilities << endl;

	double total_error = 0;

	for (int i = 0; i < mean_vector.size(); i++)
	{
		VectorXf classifications = classify_case_1(test_data[i], mean_vector, (float) covariance_matrix[0](0, 0), prior_probabilities);

		double error = get_classification_error(i, classifications);
		total_error += error;

		cout << "Misclassification error for class " << i << ": " << error << endl;
	}

	cout << "Total classification error: " << total_error << endl;
	cout << "Bhattacharrya error upper bound: " << compute_bhattacharyya(mean_vector, covariance_matrix, prior_probabilities) << endl;

	compute_decision_boundary(mean_vector, (float) covariance_matrix[0](0, 0), prior_probabilities);

	return 0;
}