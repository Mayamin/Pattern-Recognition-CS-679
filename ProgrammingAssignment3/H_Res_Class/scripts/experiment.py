import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import cv2
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
    

if __name__ == '__main__':

    # load files
    eigen_vector_matrix, eigen_vals = load_eigen_stuff('./model/eigen_faces.csv')
    projected_coefficient_matrix, train_labels = load_proj_coeffs('./model/projected_coefficients.csv')
    average_face_vector = load_avg_face('./model/average_face.csv')
    test_images_matrix, test_labels = load_test_images('./data/fb_H', average_face_vector)

    # transform them
    test_images_matrix = transform_to_eigen_space(test_images_matrix, eigen_vector_matrix[:2880, :])

    k_vals = [ get_k(eigen_vals, .8), get_k(eigen_vals, .9), get_k(eigen_vals, .95) ]

    test = projected_coefficient_matrix[:, 0]

    distances = [ [] , [] , [] ]

    eigen_mat = [ np.linalg.inv(np.diag(eigen_vals[:k, 0])) for k in k_vals ]

    # classify
    for i in range(test_images_matrix.shape[1]):

        for j in range(len(k_vals)):
            k = k_vals[j]
            test_img = np.transpose(np.atleast_2d(test_images_matrix[:k, i] ))

            distances_i = []

            # print(eigen_vals[:k].shape)
            for x in range(projected_coefficient_matrix.shape[1]):
                distances_i.append(distance.mahalanobis(test_img, projected_coefficient_matrix[:k, x], eigen_mat[j]))

            labels = get_top_50_labels(distances_i, train_labels)

            distances[j].append([test_labels[i], labels ])


    num_correct = [ [ 0 for i in range(50) ] for p in range(len(k_vals)) ]

    num1 = 0 
    num2 = 0

    for k in range(len(k_vals)):
        for r in range(50):
            for i in range(len(distances[k])):
                for j in range(r):
                    if distances[k][i][0] == distances[k][i][1][j]:
                        if (r == 1) and num1 < 3:
                            print(distances[k][i][0], " pos ", k)

                            num1 += 1

                        num_correct[k][r] += 1
                        break
                    
                    if (r == 1) and num2 < 3:
                        print(distances[k][i][0], " neg ", k)

                        num2 += 1

            # print(f'acc with {k_vals[k]} info and rank {r}: {num_correct[k][r] / len(distances[k])}')
            num_correct[k][r] /= len(distances[k])
        
        num1 = 0
        num2 = 0

    # plot results
    plt.plot(range(50), num_correct[0], 'r', label='80% info')
    plt.plot(range(50), num_correct[1], 'g', label='90% info')
    plt.plot(range(50), num_correct[2], 'b', label='95% info')
    plt.legend(loc='best')

    plt.xlabel('Rank (1-50)')
    plt.ylabel('Acc. (%)')
    plt.title('CMC curve for high res images')
    plt.savefig("images/CMC.pdf", format="pdf", bbox_inches="tight")
    plt.show()