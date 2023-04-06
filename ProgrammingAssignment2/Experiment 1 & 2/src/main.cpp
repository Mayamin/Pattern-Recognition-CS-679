#include <iostream>
#include <math.h>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <bits/stdc++.h>
#include <Eigen/Dense>

#define OUTPUT_PATH "results/"
#define EPSILON .1

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

bool isApprox(MatrixXf inp1, MatrixXf inp2)
{
	if ( ! ( inp1.cols() == inp2.cols() && inp1.rows() == inp2.rows() ) )
		return false;

	MatrixXf full_comp = inp1 - inp2;

	for (int i = 0; i < full_comp.rows(); i++)
		for (int j = 0; j < full_comp.cols(); j++)
			if (abs(full_comp(i, j)) >= EPSILON)
				return false;

	return true;
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

		float class_0_score = -1 * ( pow( ( x - mean_vector[0] ).norm(), 2 ) / ( 2 * variance ) ) + log(prior_probabilities[0]);
		float class_1_score = -1 * ( pow( ( x - mean_vector[1] ).norm(), 2 ) / ( 2 * variance ) ) + log(prior_probabilities[1]);

		if (class_0_score > class_1_score)
			classifications[i] = 0;
		else
			classifications[i] = 1;
	}

	return classifications;
}

VectorXf classify_euclid(MatrixXf to_classify, vector<VectorXf> mean_vector)
{
	VectorXf classifications(to_classify.cols());

	for ( int i = 0 ; i < to_classify.cols() ; i++ )
	{
		VectorXf x = to_classify.col(i);

		float class_0_score = ( x - mean_vector[0] ).norm();
		float class_1_score = ( x - mean_vector[1] ).norm();

		if (class_0_score > class_1_score)
			classifications[i] = 0;
		else
			classifications[i] = 1;
	}

	return classifications;
}

VectorXf classify_case_2 (MatrixXf to_classify, vector<VectorXf> mean_vector, MatrixXf covariance_matrix, VectorXf prior_probabilities)
{
	VectorXf classifications(to_classify.cols());

	float prior_0 = log(prior_probabilities[0]);
	float prior_1 = log(prior_probabilities[1]);

	for ( int i = 0 ; i < to_classify.cols() ; i++ )
	{
		VectorXf x = to_classify.col(i);

		float class_0_score = (-.5) * ( x - mean_vector[0] ).transpose() * covariance_matrix.inverse() * (x - mean_vector[0]) + prior_0;
		float class_1_score = (-.5) * ( x - mean_vector[1] ).transpose() * covariance_matrix.inverse() * (x - mean_vector[1]) + prior_1;

		if (class_0_score > class_1_score)
			classifications[i] = 0;
		else
			classifications[i] = 1;
	}

	return classifications;
}

// leveraging my knowledge that there are 2 classes
VectorXf classify_case_3 (MatrixXf to_classify, vector<VectorXf> mean_vector, vector<MatrixXf> covariance_matrix, VectorXf prior_probabilities)
{
	VectorXf classifications(to_classify.cols());

	MatrixXf W_0 = -.5 * covariance_matrix[0].inverse();
	MatrixXf w_0 = covariance_matrix[0].inverse() * mean_vector[0];
	float w_0_0 = -.5 * mean_vector[0].transpose() * covariance_matrix[0].inverse() * mean_vector[0] 
					- .5 * log(covariance_matrix[0].determinant()) + log(prior_probabilities[0]);


	MatrixXf W_1 = -.5 * covariance_matrix[1].inverse();
	MatrixXf w_1 = covariance_matrix[1].inverse() * mean_vector[1];
	float w_1_0 = -.5 * mean_vector[1].transpose() * covariance_matrix[1].inverse() * mean_vector[1] 
					- .5 * log(covariance_matrix[1].determinant()) + log(prior_probabilities[1]);

	for ( int i = 0 ; i < to_classify.cols() ; i++ )
	{
		VectorXf x = to_classify.col(i);

		float class_0_score = ( x.transpose() * W_0 * x + w_0.transpose() * x)(0, 0) + w_0_0;
		float class_1_score = ( x.transpose() * W_1 * x + w_1.transpose() * x)(0, 0) + w_1_0;

		if (class_0_score > class_1_score)
			classifications[i] = 0;
		else
			classifications[i] = 1;
	}

	return classifications;
}

bool in(int check, vector<int> list)
{
	for (int i = 0; i < list.size(); i++)
		if (list[i] == check)
			return true;

	return false;
}

MatrixXf random_select_n(MatrixXf full_data, int n)
{
	vector<Vector2f> tmp;
	vector<int> selected_indices;

	srand(time(NULL));

	for ( int i = 0; i < n ; i++ )
	{
		int r_ind = rand() % full_data.cols();

		if ( !in( r_ind, selected_indices ) )
		{
			selected_indices.push_back(r_ind);
			tmp.push_back(full_data.col(r_ind));
		}
		else
		{
			i--;
			continue;
		}
	}

	return matricize(tmp);
}

void compute_decision_boundary_1 (vector<VectorXf> mean_vector, float standard_deviation, VectorXf prior_probabilities, string file_name)
{
	MatrixXf W = (mean_vector[1] - mean_vector[0]);
	VectorXf x0 = (.5) * (mean_vector[1] + mean_vector[0]) 
					- ( standard_deviation / ( pow( (mean_vector[1] - mean_vector[0]).norm(),2 ) ) )
					  * log(prior_probabilities[1] / prior_probabilities[0]) 
					  * (mean_vector[1] - mean_vector[0]); 

	ofstream o_file;
	ostringstream os;

	os << OUTPUT_PATH << file_name;

	o_file.open(os.str());

	o_file << 1 << endl;
	o_file << W << endl;
	o_file << x0 << endl;

	o_file.close();
}

void compute_decision_boundary_2 (vector<VectorXf> mean_vector, MatrixXf covariance_matrix, VectorXf prior_probabilities, string file_name)
{
	VectorXf w_0 = covariance_matrix.inverse() * mean_vector[0];
	float w_0_0 = (-.5) * (mean_vector[0].transpose() * covariance_matrix.inverse() * mean_vector[0] + log(prior_probabilities[0]));

	VectorXf w_1 = covariance_matrix.inverse() * mean_vector[1];
	float w_1_0 = (-.5) * (mean_vector[1].transpose() * covariance_matrix.inverse() * mean_vector[1] + log(prior_probabilities[1]));

	ofstream o_file;
	ostringstream os;

	os << OUTPUT_PATH << file_name;

	o_file.open(os.str());

	o_file << 2 << endl;
	o_file << w_0 << endl;
	o_file << w_0_0 << endl;
	o_file << w_1 << endl;
	o_file << w_1_0 << endl;

	o_file.close();
}

void compute_decision_boundary_3 (vector<VectorXf> mean_vector, vector<MatrixXf> covariance_matrix, VectorXf prior_probabilities, string file_name)
{
	MatrixXf W_0 = -.5 * covariance_matrix[0].inverse();
	MatrixXf w_0 = covariance_matrix[0].inverse() * mean_vector[0];
	float w_0_0 = -.5 * mean_vector[0].transpose() * covariance_matrix[0].inverse() * mean_vector[0] 
					- .5 * log(covariance_matrix[0].determinant()) + log(prior_probabilities[0]);


	MatrixXf W_1 = -.5 * covariance_matrix[1].inverse();
	MatrixXf w_1 = covariance_matrix[1].inverse() * mean_vector[1];
	float w_1_0 = -.5 * mean_vector[1].transpose() * covariance_matrix[1].inverse() * mean_vector[1] 
					- .5 * log(covariance_matrix[1].determinant()) + log(prior_probabilities[1]);

	ofstream o_file;
	ostringstream os;

	os << OUTPUT_PATH << file_name;

	o_file.open(os.str());

	o_file << 3 << endl;
	o_file << W_0 << endl;
	o_file << w_0 << endl;
	o_file << w_0_0 << endl;
	o_file << W_1 << endl;
	o_file << w_1 << endl;
	o_file << w_1_0 << endl;

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

	double second_term = .5 * log( top / bottom );

	double k_beta = first_term + second_term;

	return pow(prior_probabilities[0], beta) * pow(prior_probabilities[1], 1 - beta) * exp(-1 * k_beta);
}

int get_correct_case(vector<MatrixXf> covariance_matrix)
{
	MatrixXf z_matrix = MatrixXf::Zero(covariance_matrix[0].rows(), covariance_matrix[0].cols());

	bool they_are_equal = true;

	for (int i = 0; i < covariance_matrix.size(); i++)
		for (int j = i + 1; j < covariance_matrix.size(); j++)
			they_are_equal = they_are_equal && isApprox(covariance_matrix[i], covariance_matrix[j]);

	if (they_are_equal)
	{
		// if they are just the identity matrix multiplied by some constant
		MatrixXf i_matrix = MatrixXf::Identity(covariance_matrix[0].rows(), covariance_matrix[0].cols());

		MatrixXf comp_matrix = covariance_matrix[0] / covariance_matrix[0](0, 0);

		if (isApprox(comp_matrix, i_matrix))
			return 1;

		// otherwise it must be case 2
		return 2;
	}

	// if they are not equal then it must be the general case
	return 3;
}

int main (int argc, char** argv)
{
	srand(time(0));

	if (argc < 2)
		return -1;

	cout << "Loading data from: \"" << argv[1] << "\"..." << endl;

	int experiment_num = (!strcmp(argv[1], "data/dataset_A.csv"))? 1: 2;


	vector<vector<MatrixXf>> training_data;
	vector<MatrixXf> test_data = load_from_file(argv[1]);
	vector<int> total_size;
	int data_set_size = 200000;

	vector<VectorXf> mean_vector;
	vector<MatrixXf> covariance_matrix;
	VectorXf prior_probabilities(test_data.size());

	cout << "Computing estimates... " << endl;


	vector<float> percent_to_use = { 1, .0001, .001, .01, .1 };

	for (int j = 0; j < percent_to_use.size(); j++)
	{
		//cout << "Using " << percent_to_use[j] << " of each dataset..." << endl;
		int total_n = 0;

		vector<MatrixXf> training_data_i;

		for (int i = 0; i < test_data.size() ; i++)
		{
			int amount = (int) (test_data[i].cols() * percent_to_use[j] ) ;

			total_n += amount;
			
			//cout << amount << " from class " << i + 1 << endl;
			MatrixXf training_data_i_class_i;

			if ( amount < test_data[i].cols() )
				training_data_i_class_i = random_select_n(test_data[i], amount);
			else
				training_data_i_class_i = test_data[i];
			//cout << "Actual vector size: " << training_data_i_class_i.cols() << endl;

			training_data_i.push_back(training_data_i_class_i);
		}

		total_size.push_back(total_n);
		training_data.push_back(training_data_i);
	}


	for (int j = 0; j < training_data.size(); j++)
	{

		cout << "=============================================================================================" << endl;
		cout << "Performing experiment " << (char)( 'a' + j ) << " with " << training_data[j][0].cols() << " points from class 1 and " << training_data[j][1].cols() << " points from class 2..." << endl;
		cout << "=============================================================================================" << endl;

		for (int i = 0; i < training_data[j].size(); i++)
		{
			VectorXf mean_i = estimate_mean(training_data[j][i]);
			MatrixXf covar_matrix_i = estimate_cov_matrix(training_data[j][i], mean_i);
			float probability_i = ( (float) training_data[j][i].cols() ) / (total_size[j]);

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

		int case_num = get_correct_case(covariance_matrix);
		cout << "Using Case " << case_num << " for classification!" << endl;

		for (int i = 0; i < mean_vector.size(); i++)
		{
			VectorXf classifications;

			if (case_num == 3)
				classifications = classify_case_3(test_data[i], mean_vector, covariance_matrix, prior_probabilities);
			else if (case_num == 2)
				classifications = classify_case_2(test_data[i], mean_vector, covariance_matrix[0], prior_probabilities);
			else
				classifications = classify_case_1(test_data[i], mean_vector, covariance_matrix[0](0, 0), prior_probabilities);

			double error = get_classification_error(i, classifications);
			total_error += error * test_data[i].cols();

			cout << "Misclassification error for class " << i << ": " << error << endl;
		}

		total_error /= data_set_size;

		cout << "Total classification error: " << total_error << endl;
		cout << "Bhattacharrya error upper bound: " << compute_bhattacharyya(mean_vector, covariance_matrix, prior_probabilities) << endl;


		ostringstream os;
		os << "experiment_" << experiment_num << "_" << j << "_results.csv";

		if (case_num == 3)
			compute_decision_boundary_3(mean_vector, covariance_matrix, prior_probabilities, os.str());
		else if (case_num == 2)
			compute_decision_boundary_2(mean_vector, covariance_matrix[0], prior_probabilities, os.str());
		else	
			compute_decision_boundary_1(mean_vector, (float) covariance_matrix[0](0, 0), prior_probabilities, os.str());

		mean_vector.clear();
		covariance_matrix.clear();
	}

	return 0;
}