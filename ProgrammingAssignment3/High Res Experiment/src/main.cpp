// cpp libraries
#include <iostream>
#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>
#include <limits>

// c libraries
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <dirent.h>
#include <sys/types.h>
#include <bits/stdc++.h>

// Eigen libraries
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

// opencv2 libraries
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

// overall constants
#define EPSILON .001
#define RECONSTRUCTION_ERROR_THRESHOLD 2

// use these namespaces to keep the code legible
using namespace Eigen;
using namespace std;
using namespace cv;


// dataset filenames are of the form nnnnn_yymmdd_xx_qq.pgm so this function returns the nnnnn
string get_id_from_file_name(string file_name)
{
	return file_name.substr(0, 5);
}

// example taken from here: https://www.tutorialspoint.com/how-can-i-get-the-list-of-files-in-a-directory-using-c-or-cplusplus
vector<string> get_files_from_directory(string path)
{
	DIR *directory;
	struct dirent *entry;
	vector<string> to_ret;

	directory = opendir(path.c_str());

	if (directory)
	{
		entry = readdir(directory);

		while (entry != nullptr)
		{
			if (strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0)
			{
				to_ret.push_back(entry->d_name);
			}

			entry = readdir(directory);
		}
	}

	return to_ret;
}

vector<MatrixXd> load_images_from_path(string directory_path, int& image_width, int& image_height)
{
	cout << "Loading files... " << endl;
	vector<string> file_paths = get_files_from_directory(directory_path);
	vector<MatrixXd> to_ret;

	// first get the labels as integers in to_ret[0] as a column vector
	MatrixXd labels(file_paths.size(), 1);

	for (int i = 0; i < file_paths.size(); i++)
	{
		labels(i, 0) = stoi(get_id_from_file_name(file_paths[i]));
	}

	int image_num = 0;

	Mat img = imread(directory_path + file_paths[image_num], IMREAD_COLOR);

	if (img.empty())
	{
		cout << "Could not read image..." << endl;

		return to_ret;
	}

	// create a D x M matrix to contain all of the data
	MatrixXd data(img.rows * img.cols, file_paths.size());

	image_width = img.cols;
	image_height = img.rows;


	for (;image_num < file_paths.size(); image_num++)
	{
		if (image_num != 0)
		{
			// cout << "Reading image: " << file_paths[image_num] << endl;
			img = imread(directory_path + file_paths[image_num], IMREAD_COLOR);

			if (img.empty())
			{
				cout << "Could not read image..." << endl;

				return to_ret;
			}
		}

		for (int i = 0; i < img.rows; i++)
		{
			for (int j = 0; j < img.cols; j++)
			{
				data(i * img.cols + j, image_num) = img.at<Vec3b>(i, j)[0];
			}
		}
	}

	// cout << data << endl;

	to_ret.push_back(labels);
	to_ret.push_back(data);

	return to_ret;
}

// to check is assumed to be square as only square matrices can be symmetric
bool check_symmetric (MatrixXd to_check)
{
	// we can start at 1 because the middle row is irrelevant
	for (int i = 1; i < to_check.rows(); i++)
		// then we can from 0 till we hit the middle (j = i) for the same reason
		for (int j = 0; j < i; j++)
			// use the normal method to check if they are unequal
			if ( abs(to_check(i, j) - to_check(j, i)) > EPSILON ) 
				return false;

	// runtime is O(nC2)
	return true;
}

bool check_orthogonal_and_eigen(MatrixXd initial_mat, MatrixXd eigen_vectors, VectorXd eigen_values)
{
	// first check the orthogonality of the eigen vectors
	for (int i = 0; i < eigen_vectors.cols() - 1; i++)
	{
		for (int j = i + 1; j < eigen_vectors.cols(); j++)
		{
			double dot_prod = eigen_vectors.col(i).dot(eigen_vectors.col(j));

			if (abs(dot_prod) > EPSILON)
				return false;
		}
	}

	// then check the property Cu = lambda u
	for (int i = 0; i < eigen_values.rows(); i++)
	{
		MatrixXd lhs = initial_mat * eigen_vectors.col(i);
		MatrixXd rhs = eigen_values(i, 0) * eigen_vectors.col(i);

		if ((lhs - rhs).norm() > EPSILON)
			return false;
	}

	return true;
}

bool check_orthogonal(MatrixXd eigen_vectors)
{
	int k = 0;
	// first check the orthogonality of the eigen vectors
	for (int i = 0; i < eigen_vectors.cols() - 1; i++)
	{
		for (int j = i + 1; j < eigen_vectors.cols(); j++)
		{
			double dot_prod = eigen_vectors.col(i).dot(eigen_vectors.col(j));

			// cout << "Dot prod " << i << " " << j << ": " << dot_prod << endl;

			if (abs(dot_prod) > EPSILON && k == 1)
				return false;
			else
				k++;
		}
	}

	return true;
}

bool check_average_reconstruction_error(VectorXd initial_data, VectorXd avg_face, MatrixXd eigen_vectors)
{
	// first center initial data on the origin
	VectorXd transformed = initial_data - avg_face;

	// next project the vector into the eigen space and save the coefficients
	VectorXd projected_coefficients(eigen_vectors.cols());

	for (int i = 0; i < eigen_vectors.cols(); i++)
		projected_coefficients(i) = transformed.transpose() * eigen_vectors.col(i);


	// now reconstruct the vector in the initial space
	VectorXd reconstruction(transformed.rows());

	for (int i = 0; i < projected_coefficients.rows(); i++)
		reconstruction = reconstruction + projected_coefficients[i] * eigen_vectors.col(i);

	// add back in the avg_face
	reconstruction = reconstruction + avg_face;

	// then the difference between the initial data and the reconstruction is the reconstruction error vector
	VectorXd error = initial_data - reconstruction;

	cout << "Reconstruction error: " << error.norm() / initial_data.rows() << endl;

	// if norm is less than the threshold the test is passed
	return ( error.norm() / initial_data.rows() ) < RECONSTRUCTION_ERROR_THRESHOLD;
}

vector<MatrixXd> get_sorted_orthonormal_eigenvectors(MatrixXd inp)
{
	// use eigen lib built in solver
	EigenSolver<MatrixXd> solver(inp);

	// each row is 1 eigen value
	MatrixXcd vals = solver.eigenvalues();
	// each column is a different eigen vector (eigenval (k, 0) corresponds to col k)
	MatrixXcd vectors = solver.eigenvectors();

	// sort them to the output
	MatrixXd vals_to_ret(vals.rows(), vals.cols());
	MatrixXd vectors_to_ret(vectors.rows(), vectors.cols());

	vector<int> chosen_indices;

	for (int i = 0; i < vectors.cols(); i++)
	{
		int min_eigenval_index = -1;
		double min_val = numeric_limits<double>::infinity();

		// find min index not already chosen
		for (int j = 0; j < vals.rows(); j++)
		{
			if (vals(j, 0).real() < min_val && (std::find(chosen_indices.begin(), chosen_indices.end(), j) == chosen_indices.end()) && ( chosen_indices.size() == 0 || chosen_indices.back() != j ) )
			{
				min_val = vals(j, 0).real();
				min_eigenval_index = j;
			}
		}

		chosen_indices.push_back(min_eigenval_index);

		double vector_norm = vectors.col(min_eigenval_index).norm();

		// copy the smallest to the furthest column to the right
		for (int j = 0; j < vectors.rows(); j++)
		{
			vals_to_ret( ( vals.rows() - 1 ) - i, 0) = vals(min_eigenval_index, 0).real();
			vectors_to_ret(j, ( vectors.cols() - 1 ) - i) = vectors(j, min_eigenval_index).real() / vector_norm;
		}
	}

	return {vals_to_ret, vectors_to_ret};
}

MatrixXd get_covariance_matrix(MatrixXd inp_data)
{
	MatrixXd to_ret = inp_data.transpose() *  inp_data;

	return to_ret / to_ret.cols();
}

VectorXd get_average(MatrixXd inp_data)
{
	VectorXd average = inp_data.col(0);

	for (int i = 1; i < inp_data.cols(); i++)
	{
		VectorXd xi = inp_data.col(i);

		average = average + (1 / (i)) * (xi - average);
	}

	return average;
}

MatrixXd transform_to_d(MatrixXd data, MatrixXd eigen_vectors)
{
	MatrixXd transformed(data.rows(), eigen_vectors.cols());

	for (int i = 0; i < eigen_vectors.cols(); i++)
	{
		transformed.col(i) = data * eigen_vectors.col(i);

		transformed.col(i) /= transformed.col(i).norm();
	}

	return transformed;
}

MatrixXd subtract_average_face(MatrixXd data, VectorXd avg_face)
{
	MatrixXd transformed_face(data.rows(), data.cols());

	for (int i = 0; i < data.cols(); i++)
	{
		transformed_face.col(i) = data.col(i) - avg_face;
	}

	return transformed_face;
}

MatrixXd project_data_to_eigen_space(MatrixXd data, MatrixXd eigen_vectors)
{
	MatrixXd projected_data(eigen_vectors.cols(), data.cols());

	for (int i = 0; i < data.cols(); i++)
	{
		VectorXd coefficients(eigen_vectors.cols());

		for (int j = 0; j < eigen_vectors.cols(); j++)
		{
			coefficients(j) = data.col(i).transpose() * eigen_vectors.col(j);
		}

		projected_data.col(i) = coefficients;
	}

	return projected_data;
}

void visualize_vector_as_image_and_display(VectorXd image_data, int image_width, int image_height)
{

}

void save_results_to_file(string output_directory_path, VectorXd average_face, MatrixXd eigen_faces, MatrixXd coefficient_labels, MatrixXd projected_coefficients)
{
	ofstream o_file;
	ostringstream eigen_faces_file_path;
	ostringstream projected_coefficients_file_path;
	ostringstream average_face_file_path;

	eigen_faces_file_path << output_directory_path << "eigen_faces.csv";
	projected_coefficients_file_path << output_directory_path << "projected_coefficients.csv";
	average_face_file_path << output_directory_path << "average_face.csv";

	o_file.open(eigen_faces_file_path.str());

	o_file << eigen_faces << endl;

	o_file.close();

	o_file.open(projected_coefficients_file_path.str());

	o_file << coefficient_labels.transpose() << endl;
	o_file << projected_coefficients << endl;

	o_file.close();

	o_file.open(average_face_file_path.str());

	o_file << average_face << endl;

	o_file.close();


}

int main (int argc, char **argv)
{
	if (argc < 2)
		return -1;

	int width, height;

	width = height = 0;

	vector<MatrixXd> data = load_images_from_path(string(argv[1]), width, height);

	VectorXd avg_face = get_average(data[1]);
	VectorXd reconstruction_test_face = data[1].col(3);

	data[1] = subtract_average_face(data[1], avg_face);

	MatrixXd sigma = get_covariance_matrix(data[1]);

	if ( ! check_symmetric(sigma) ) 
	{
		cout << "Covariance matrix is not symmetric, returning in error!" << endl;
		return -1;
	}

	vector<MatrixXd> eigen_stuff = get_sorted_orthonormal_eigenvectors(sigma);

	if ( ! check_orthogonal_and_eigen(sigma, eigen_stuff[1], eigen_stuff[0].col(0)) )
	{
		cout << "Eigen vectors are not orthogonal before reconstruction, returning in error!" << endl;

		return -1;
	}

	eigen_stuff[1] = transform_to_d(data[1], eigen_stuff[1]);

	if ( ! check_orthogonal(eigen_stuff[1]) )
	{
		cout << "Eigen vectors are not orthogonal, returning in error!" << endl;

		return -1;
	}

	if ( ! check_average_reconstruction_error(reconstruction_test_face, avg_face, eigen_stuff[1]) )
	{
		cout << "Average reconstruction error is too high, returning in error!" << endl;

		return -1;
	}

	MatrixXd projected_coefficients = project_data_to_eigen_space(data[1], eigen_stuff[1]);

	// now I just save everything to a file and it should be good
	save_results_to_file("./model/", avg_face, eigen_stuff[1], data[0], data[1]);

	return 0;
}
