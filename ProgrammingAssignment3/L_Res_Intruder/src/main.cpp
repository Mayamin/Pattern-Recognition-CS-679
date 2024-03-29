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
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

// opencv2 libraries
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/eigen.hpp>

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

VectorXd vectorize_matrix(MatrixXd inp)
{
	VectorXd to_ret(inp.rows() * inp.cols());

	for (int i = 0; i < inp.rows(); i++)
	{
		for (int j = 0; j < inp.cols(); j++)
		{
			to_ret(i * inp.cols() + j) = inp(i, j);
		}
	}

	return to_ret;
}

MatrixXd matricize_vector(VectorXd inp, int width, int height)
{
	MatrixXd to_ret(height, width);

	int row = 0;
	int col = 0;

	for (int i = 0; i < inp.rows(); i++, col++)
	{
		if (col > width - 1)
		{
			col = 0;
			row++;
		}

		to_ret(row, col) = inp(i);
	}

	return to_ret;
}

vector<MatrixXd> load_images_from_path(string directory_path, int& width, int& height)
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

	width = img.cols;
	height = img.rows;

	if (img.empty())
	{
		cout << "Could not read image..." << endl;

		return to_ret;
	}

	// create a D x M matrix to contain all of the data
	MatrixXd data(img.rows * img.cols, file_paths.size());

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

		MatrixXd tmp(height, width);
		
		vector<Mat> channels(3);

		split(img, channels);

		cv2eigen(channels[0], tmp);

		VectorXd vec = vectorize_matrix(tmp);

		data.col(image_num) = vec;
	
	}

	// cout << data << endl;

	to_ret.push_back(labels);
	to_ret.push_back(data);

	return to_ret;
}

// to check is assumed to be square as only square matrices can be symmetric
bool check_symmetric (const MatrixXd to_check)
{
	return to_check.isApprox(to_check.transpose());
}

bool check_orthogonal(const MatrixXd eigen_vectors)
{
	// cout << eigen_vectors.transpose() * eigen_vectors << endl;
	return (eigen_vectors.transpose() * eigen_vectors).isIdentity(EPSILON);
}

bool check_average_reconstruction_error(const VectorXd initial_data, const VectorXd avg_face, const MatrixXd eigen_vectors)
{
	// first center initial data on the origin
	VectorXd transformed = initial_data - avg_face;

	// next project the vector into the eigen space and save the coefficients
	VectorXd projected_coefficients =  eigen_vectors.transpose() * transformed;

	// now reconstruct the vector in the initial space
	VectorXd reconstruction =  eigen_vectors * projected_coefficients;

	// for (int i = 0; i < projected_coefficients.rows(); i++)
	// 	reconstruction = reconstruction + projected_coefficients[i] * eigen_vectors.col(i);

	// add back in the avg_face
	reconstruction = reconstruction + avg_face;

	// then the difference between the initial daa and the reconstruction is the reconstruction error vector
	VectorXd error = initial_data - reconstruction;

	cout << "Reconstruction error: " << error.norm() / initial_data.rows() << endl;

	// if norm is less than the threshold the test is passed
	return ( error.norm() / initial_data.rows() ) < RECONSTRUCTION_ERROR_THRESHOLD;
}

vector<MatrixXd> get_sorted_orthonormal_eigenvectors(MatrixXd inp)
{
	// use eigen lib built in solver
	SelfAdjointEigenSolver<MatrixXd> solver(inp);

	// each row is 1 eigen value
	MatrixXd vals = solver.eigenvalues();
	// each column is a different eigen vector (eigenval (k, 0) corresponds to col k)
	MatrixXd vectors = solver.eigenvectors();

	// cout << "=" << vectors.cols() << endl;

	// they come in sorted smallest to largest, so reverse the orders and return them
	return {vals.colwise().reverse(), vectors.rowwise().reverse()};
}

MatrixXd get_covariance_matrix(MatrixXd inp_data)
{
	return (inp_data.transpose() *  inp_data) / inp_data.cols();
}

VectorXd get_average(MatrixXd inp_data)
{
	return inp_data.rowwise().mean();
}

MatrixXd transform_to_d(MatrixXd data, MatrixXd eigen_vectors)
{
	MatrixXd to_ret = ( data * eigen_vectors ).leftCols(min(data.rows(), data.cols()));

	to_ret.colwise().normalize();

	return to_ret;
}

MatrixXd subtract_average_face(MatrixXd data, VectorXd avg_face)
{
	return data.colwise() - avg_face;
}

MatrixXd project_data_to_eigen_space(MatrixXd data, MatrixXd eigen_vectors)
{
	return (data.transpose() * eigen_vectors).transpose();
}

void save_results_to_file(string output_directory_path, VectorXd average_face, MatrixXd eigen_faces, MatrixXd eigen_vals, MatrixXd coefficient_labels, MatrixXd projected_coefficients)
{
	ofstream o_file;
	ostringstream eigen_faces_file_path;
	ostringstream projected_coefficients_file_path;
	ostringstream average_face_file_path;

	eigen_faces_file_path << output_directory_path << "eigen_faces.csv";
	projected_coefficients_file_path << output_directory_path << "projected_coefficients.csv";
	average_face_file_path << output_directory_path << "average_face.csv";

	o_file.open(eigen_faces_file_path.str());

	o_file << eigen_vals.transpose() << endl;
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

void save_vector_image_to_file(VectorXd image, int width, int height, string file_name)
{
	cout << "Saving to file: " << file_name << endl;
	Mat cv_image(height, width, CV_8U);

	Matrix<uint8_t, -1, -1> tmp = matricize_vector(image, width, height).cast<uint8_t>();

	eigen2cv(tmp, cv_image);

	imwrite(file_name, cv_image);
}

int main (int argc, char **argv)
{
	if (argc < 2)
		return -1;

	int width, height;

	vector<MatrixXd> data = load_images_from_path(string(argv[1]), width, height);

	VectorXd avg_face = get_average(data[1]);

	VectorXd reconstruction_test_face = data[1].col(3);

	data[1] = subtract_average_face(data[1], avg_face);

	cout << "Calculating Covariance Matrix..." << endl;

	MatrixXd sigma = get_covariance_matrix(data[1]);

	if ( ! check_symmetric(sigma) ) 
	{
		cout << "Covariance matrix is not symmetric, returning in error!" << endl;
		return -1;
	}

	cout << "Estimating eigen vectors..." << endl;

	vector<MatrixXd> eigen_stuff = get_sorted_orthonormal_eigenvectors(sigma);

	if ( ! check_orthogonal(eigen_stuff[1]) )
	{
		cout << "Eigen vectors are not orthogonal before transformation, returning in error!" << endl;

		return -1;
	}

	cout << "Transforming eigen vectors to d dimensions..." << endl; 

	eigen_stuff[1] = transform_to_d(data[1], eigen_stuff[1]);

	if ( ! check_orthogonal(eigen_stuff[1]) )
	{
		cout << "Eigen vectors are not orthogonal before transformation, returning in error!" << endl;

		return -1;
	}

	cout << "Estimating reconstruction error..." << endl;

	if ( ! check_average_reconstruction_error(reconstruction_test_face, avg_face, eigen_stuff[1]) )
	{
		cout << "Average reconstruction error is too high, returning in error!" << endl;

		return -1;
	}

	cout << "Saving results..." << endl; 

	MatrixXd projected_coefficients = project_data_to_eigen_space(data[1], eigen_stuff[1]);

	// now I just save everything to a file and it should be good
	save_results_to_file("./model/", avg_face, eigen_stuff[1], eigen_stuff[0], data[0], projected_coefficients);

	save_vector_image_to_file(avg_face, width, height, "./images/avg_face.pgm");

	ostringstream eigen_faces_file_path;

	for (int i = 0; i < 10; i++)
	{
		eigen_faces_file_path.str("");
		
		eigen_faces_file_path << "./images/eigen_face_highest_" << i << ".pgm";

		VectorXd to_save = eigen_stuff[1].col(i).array() - eigen_stuff[1].col(i).minCoeff();

		to_save = 255 * ( to_save ) / (eigen_stuff[1].col(i).maxCoeff() - eigen_stuff[1].col(i).minCoeff());

		save_vector_image_to_file(to_save, width, height, eigen_faces_file_path.str());

		eigen_faces_file_path.str("");
		
		eigen_faces_file_path << "./images/eigen_face_lowest_" << i << ".pgm";	

		to_save = eigen_stuff[1].col(eigen_stuff[1].cols() - 1 - i).array() - eigen_stuff[1].col(eigen_stuff[1].cols() - 1 - i).minCoeff();

		to_save = 255 * ( to_save ) / (eigen_stuff[1].col(eigen_stuff[1].cols() - 1 - i).maxCoeff() - eigen_stuff[1].col(eigen_stuff[1].cols() - 1 - i).minCoeff());

		save_vector_image_to_file(to_save, width, height, eigen_faces_file_path.str());
	}

	cout << "All done" << endl;

	return 0;
}
