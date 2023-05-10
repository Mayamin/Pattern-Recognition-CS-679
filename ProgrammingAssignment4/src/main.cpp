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

VectorXf get_mean(MatrixXf inp)
{
	return inp.rowwise().mean();
}

MatrixXf get_covariance_matrix(MatrixXf inp_data, VectorXf mean)
{
	MatrixXf temp = inp_data.colwise() - mean;

	temp = (temp *  temp.transpose()) / temp.cols();

	MatrixXf to_ret = MatrixXf::Identity(temp.rows(), temp.cols());

	for (int i = 0; i < to_ret.cols(); i++)
		to_ret(i, i) = temp (i, i);

	return to_ret;
}

MatrixXf matricize (vector<VectorXf> inp)
{
	MatrixXf to_ret(inp[0].rows(), inp.size());

	for (int i = 0; i < inp.size(); i++)
	{
		to_ret.col(i) = inp[i];
	}

	return to_ret;	
}

MatrixXf load_data_from_file (string file_name, int num_principle_components)
{
	ifstream in_file;
	string temp;

	in_file.open("./" + file_name, ios::in);

	vector<VectorXf> data_vec;

	int cur_data_set = 0;

	while(getline(in_file, temp))
	{
		// cout << temp << endl;

		int data_num = 0;
		string data_val = "";
		VectorXf picture_i(num_principle_components);

		for (int i = 0 ; i < temp.length(); i++)
		{

			if (('0' <= temp[i] && temp[i] <= '9') || temp[i] == '.' || temp[i] == '-')
			{
				data_val += temp[i];
			}
			else if (' ' == temp[i])
			{
				picture_i[data_num++] = stof(data_val);

				data_val = "";
			}

			if (data_num >= num_principle_components)
				break;
		}

		data_vec.push_back(picture_i);

	}

	in_file.close();

	return matricize(data_vec);
}

VectorXi load_labels_from_file(string file_name)
{
	vector<int> labels_std;

	ifstream in_file;
	string temp;

	in_file.open("./" + file_name, ios::in);

	getline(in_file, temp);

	string data_val = "";

	for (int i = 0 ; i < temp.length(); i++)
	{
		// cout << temp[i];

		if (('0' <= temp[i] && temp[i] <= '9') || temp[i] == '.' || temp[i] == '-')
		{
			data_val += temp[i];
		}
		else if (' ' == temp[i])
		{
			labels_std.push_back(stoi(data_val));

			data_val = "";
		}
	}
	
	labels_std.push_back(stoi(data_val));

	VectorXi labels(labels_std.size());

	for (int i = 0; i < labels_std.size(); i++)
		labels[i] = labels_std[i]; 

	return labels;
}

vector<MatrixXf> separate_data_by_label(MatrixXf data, VectorXi labels)
{
	vector<MatrixXf> class_separated_data;

	vector<vector<VectorXf>> temp_vector_classes;

	int num_sorted = 0;
	int total_num = data.cols();
	
	// labels start at 1
	int iter = 1;

	while (num_sorted < total_num)
	{
		vector<VectorXf> class_i;

		for (int i = 0; i < data.cols(); i++)
			if (labels[i] == iter)
				class_i.push_back(data.col(i));

		temp_vector_classes.push_back(class_i);

		num_sorted += class_i.size();

		iter++;
	}

	for (int i = 0; i < temp_vector_classes.size(); i++)
	{
		MatrixXf class_i(temp_vector_classes[i][0].rows(), temp_vector_classes[i].size());

		for (int j = 0; j < temp_vector_classes[i].size(); j++)
			class_i.col(j) = temp_vector_classes[i][j];

		class_separated_data.push_back(class_i);
	}

	return class_separated_data;
}

// leveraging my knowledge that there are 2 classes
VectorXi classify_case_3 (MatrixXf to_classify, vector<VectorXf> mean_vector, vector<MatrixXf> covariance_matrix, VectorXf prior_probabilities)
{
	VectorXi classifications(to_classify.cols());

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
			classifications[i] = 1;
		else
			classifications[i] = 2;
	}

	return classifications;
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

double get_classification_error(VectorXi classifications, VectorXi labels)
{
	int num_incorrect = 0;
	double to_ret = 0;

	VectorXi comparisons = classifications - labels;

	comparisons = comparisons.array().abs();

	num_incorrect = comparisons.sum();

	to_ret += num_incorrect;
	to_ret /= classifications.rows();

	return to_ret;
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

double calc_rmse(VectorXf estimate, VectorXf gt)
{
	return (estimate - gt).norm() / sqrt((estimate.size()));
}

double calc_rmse(MatrixXf estimate, MatrixXf gt)
{
	return (estimate - gt).norm() / sqrt((estimate.cols() * estimate.rows()));
}

int main (int argc, char** argv)
{
	srand(time(0));

	if (argc < 7)
		return -1;

	int npc = stoi(argv[6]);
		
	cout << "Classifying using top " << npc << " Principle Components..." << endl;

	for (int i = 0; i < 3; i++)
	{
		cout << "======================================================================" << endl;

		int fold_num = i + 1;
	
		cout << "Performing training and classification with data from fold " << fold_num << "..." << endl;

		ostringstream train_data_file_name;
		train_data_file_name << argv[1] << fold_num << ".txt";
		cout << "Loading train data from: \"" << train_data_file_name.str() << "\"..." << endl;
		MatrixXf train_data = load_data_from_file(train_data_file_name.str(), npc);

		ostringstream train_labels_file_name;
		train_labels_file_name << argv[2] << fold_num << ".txt";
		cout << "Loading train labels from: \"" << train_labels_file_name.str() << "\"..." << endl;
		VectorXi train_labels = load_labels_from_file(train_labels_file_name.str());

		cout << "Separating data by label..." << endl;
		vector<MatrixXf> data_by_label = separate_data_by_label(train_data, train_labels);

		cout << "Estimating mean and covariance for each class..." << endl;

		vector<VectorXf> class_means;
		vector<MatrixXf> class_covariances;

		for (int i = 0; i < data_by_label.size(); i++)
		{
			class_means.push_back(get_mean(data_by_label[i]));
			class_covariances.push_back(get_covariance_matrix(data_by_label[i], class_means[i]));
		}

		ostringstream test_data_file_name;
		test_data_file_name << argv[3] << fold_num << ".txt";
		cout << "Loading test data from: \"" << test_data_file_name.str() << "\"..." << endl;
		MatrixXf test_data = load_data_from_file(test_data_file_name.str(), npc);


		ostringstream test_labels_file_name;
		test_labels_file_name << argv[4] << fold_num << ".txt";
		cout << "Loading test labels from: \"" << test_labels_file_name.str() << "\"..." << endl;
		VectorXi test_labels = load_labels_from_file(test_labels_file_name.str());

		VectorXf prior_probabilities(2);
		prior_probabilities[0] = .5;
		prior_probabilities[1] = .5;

		cout << "Classifying test data set with trained values..." << endl;

		VectorXi classifications = classify_case_3(test_data, class_means, class_covariances,  prior_probabilities);

		cout << "Computing error in classification..." << endl;

		double error = get_classification_error(classifications, test_labels);

		cout << "Total error in classification for fold " << fold_num << ": " << error << endl;

		cout << "======================================================================" << endl;
	}

	return 0;
}