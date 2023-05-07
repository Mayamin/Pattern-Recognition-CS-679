import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import cv2
import copy
from PIL import Image
from os import listdir
from numpy import linalg as LA
from scipy.spatial import distance

dimensions = 2880

def get_k(eigen_vals, threshold):
    sum_eigen_values = sum(eigen_vals)
    k = 0
    numerator = 0
    
    for i in range (len(eigen_vals)):

        numerator = numerator + eigen_vals[i]
        temp_k = numerator / sum_eigen_values
        
        if temp_k >= threshold:
            k = i+1
            break

    return k

def load_eigen_stuff (file_name):
    to_ret = np.array([])

    with open(file_name, 'r') as csv_file:
        # Create a CSV reader object
        csv_reader = csv.reader(csv_file)
        
        #first row is taken next time loop will start from second row
        first_row = next( csv_reader)
    
        eigen_values = first_row[0].split()
        eigen_values = np.transpose(np.atleast_2d([ float(i) for i in eigen_values ]))
        
        to_ret = np.zeros((dimensions, eigen_values.shape[0]))

        row_num = 0

        for row in csv_reader:
            #splitting the elements of first column of each row
            row_arr = row[0].split()
            
            #converting values in each row to float
            row_arr = np.array([ float(i) for i in row_arr ])

            for i in range(len(row_arr)):
                to_ret[row_num, i] = row_arr[i]

            row_num += 1

        print(row_num)
        
        
    return to_ret, eigen_values

def load_proj_coeffs (file_name):
    to_ret = np.array([])

    with open(file_name,'r') as csv_file:
    
        csv_reader = csv.reader(csv_file)   
        # saving train labels
        #taking first row so all iterations in file later on will skip the first row
        first_row = next(csv_reader)
        train_labels = first_row[0].split()
        train_labels = np.array([ int(i) for i in train_labels ])

        to_ret = np.zeros((1204, len(train_labels)))

        row_num = 0

        for row in csv_reader:
            row_arr = row[0].split()
            row_arr = np.array([ float(i) for i in row_arr ])

            for i in range(len(row_arr)):
                to_ret[row_num, i] = row_arr[i]

            row_num += 1
        

    return to_ret, train_labels

def load_avg_face (file_name):
    to_ret = []

    with open(file_name, 'r') as csv_file:
        # Create a CSV reader object
        csv_reader = csv.reader(csv_file)
        
        # Iterate over each row in the CSV file
        for row in csv_reader:
            to_ret.append( float(row[0]))

    return np.transpose(np.atleast_2d(to_ret))

def load_test_images(path, avg_face):
    to_ret = np.zeros((dimensions, 1196))
    test_ids = []

    i = 0

    for file_name in os.listdir(path):

        file_name_with_path = os.path.join(path, file_name)
        img = cv2.imread(file_name_with_path)
        temp = file_name.split('_')
        
        test_ids.append(int(temp[0]))

        if img is not None:
            image = Image.open(file_name_with_path)
            img_arr = np.array(image).flatten()

            tmp = np.transpose(np.atleast_2d([ float(i) for i in img_arr ]))
            tmp -= avg_face

            for j in range(to_ret.shape[0]):
                to_ret[j, i] = tmp[j]

        i += 1

    return to_ret, test_ids

def transform_to_eigen_space( test_mat , eigen_matrix ):
    return np.transpose(np.matmul(np.transpose(test_mat), eigen_matrix))

def get_maha_distance( vec1, train_image_matrix, eigen_values):

    sum = vec1 - train_image_matrix

    sum = np.square(sum)

    sum = sum / eigen_values

    return np.sum(sum)

def get_top_50_labels(distances, train_labels):
    to_ret = []
    already_picked = []

    for i in range(50):
        min = float('inf')
        mindex = -1

        for j in range(len(distances)):

            if distances[j] < min and (j not in already_picked):
                min = distances[j]
                mindex = j
        
        already_picked.append(mindex)
        to_ret.append(train_labels[mindex])

    return to_ret

def get_min_distance(distances):
    min = float('inf')
    mindex = -1

    for i in range(50):
        for j in range(len(distances)):

            if distances[j] < min :
                min = distances[j]
                mindex = j

    return distances[mindex]
    

if __name__ == '__main__':

    # load files
    eigen_vector_matrix, eigen_vals = load_eigen_stuff('./model/eigen_faces.csv')
    projected_coefficient_matrix, train_labels = load_proj_coeffs('./model/projected_coefficients.csv')
    average_face_vector = load_avg_face('./model/average_face.csv')
    test_images_matrix, test_labels = load_test_images('./data/fb_H', average_face_vector)

    # transform them
    test_images_matrix = transform_to_eigen_space(test_images_matrix, eigen_vector_matrix[:2880, :])

    thresholds = []

    k_vals = [ get_k(eigen_vals, .95) ]

    distances_matrix = np.zeros((test_images_matrix.shape[1], projected_coefficient_matrix.shape[1]))

    eigen_mat = np.linalg.inv(np.diag(eigen_vals[:k_vals[0], 0]))

    # generate thresholds
    for i in range(test_images_matrix.shape[1]):

        k = k_vals[0]
        test_img = np.transpose(np.atleast_2d(test_images_matrix[:k, i] ))

        for j in range(projected_coefficient_matrix.shape[1]):
            distances_matrix[i, j] = distance.mahalanobis(test_img, projected_coefficient_matrix[:k, j], eigen_mat)

    # print(distances_matrix[:, :3])

    thresholds = copy.deepcopy(distances_matrix[:, 0])

    thresholds.sort()

    # min = thresholds[0]
    # max = thresholds[-1]

    # step = (max - min) / 30

    # thresholds = [ i * step + min for i in range(30) ]


    # True if the test image is not an intruder
    gt_intruder = [ i in train_labels for i in test_labels ]
    num_intruders = 0
    num_non_intruders = 0

    for i in gt_intruder:
        if i:
            num_non_intruders += 1
        else:
            num_intruders += 1

    # TP is 1
    # FP is 2
    # FN is 3
    # TN is 4
    verdicts = [ [] for i in range(len(thresholds))]

    # print(thresholds)

    for t_iter in range(len(thresholds)):
        t = thresholds[t_iter]

        for i in range(test_images_matrix.shape[1]):

            min = distances_matrix[i, 0]

            # if this is greater than t then we rule intruder
            verdict = min < t
            # print(f'{min} < {t}')

            # TP
            if verdict and gt_intruder[i]:
                verdicts[t_iter].append(1)
            # FP
            elif verdict and not gt_intruder[i]:
                verdicts[t_iter].append(2)
            # FN
            elif not verdict and gt_intruder[i]:
                verdicts[t_iter].append(3)
            # TN
            else:
                verdicts[t_iter].append(4)

    true_positive_rates = [ 0 for i in range(len(thresholds)) ]
    false_positive_rates = [ 0 for i in range(len(thresholds)) ]

    # print(verdicts)

    for i in range(len(verdicts)):
        for j in range(len(verdicts[i])):
            if verdicts[i][j] == 1:
                true_positive_rates[i] += 1
            if verdicts[i][j] == 2:
                false_positive_rates[i] += 1

        true_positive_rates[i] /= num_non_intruders
        false_positive_rates[i] /= num_intruders

    # print(true_positive_rates)
    # print(false_positive_rates)

    plt.plot(false_positive_rates, true_positive_rates, c='r')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC for Intruder Detection")
    plt.savefig("images/ROC.pdf", format="pdf", bbox_inches="tight")
    plt.show()