import numpy as np
import csv
import PIL
from PIL import Image
import math
import sys
from numpy import linalg as LA
import os
from os import listdir
import cv2
import glob

def centering_img(x,average_face):
    
    centered_image = []
    centered_image = x - average_face

    return centered_image


def mahalanobis_dist (eig_vals, test_coefficients, train_coefficients, k):
    #Q1 -> Should I take k number of eigen values ? did so
    k_eig_vals = []
    k_eig_vals = np.array( eig_vals[:k])

    tmp1 =  np.transpose(np.atleast_2d(1/k_eig_vals))
    # print("tmp1.shape",tmp1.shape)
    # print("train_coefficients.shape ",train_coefficients.shape)
    #there was an error here with train_coefficient slicing [:k,] 1st half means take k rows and next empty half means take all columns which is true since we are considering all columns
    tmp2 = test_coefficients[0:k]-train_coefficients[ :k,]
    tmp3 = np.square(tmp2)
    # print("tmp3.shape",tmp3.shape)

    #Q2 -> Should I Check if it should be temp1 and tmp3 inside np.dot? did that
    tmp4 = np.dot(tmp1, tmp3)

    # return np.sum(tmp4) 
    return tmp4 #sayed suggestion

# print("Hello")
# with open('model/average_face.csv', 'r') as file:
#     # for i in file.rows():
#     #     print(row[])
  
#     csv_reader = csv.reader(csv_file)
    
#     # Iterate over each row in the CSV file
#     for row in csv_reader:

# image = Image.open('00001_930831_fa_a.pgm')
image = Image.open('00002_930831_fa.pgm')


#convert image to numpy array
image_array = np.array(image)
average_face = []

eigen_values = np.array([])
train_coefficeints = np.array([])

# Flatten the 2D array to a 1D vector
pixel_vector = np.transpose(np.atleast_2d(image_array.flatten()))

# print("pixel_vector ", pixel_vector.shape )

#new image array
centered_image = np.array([])
counter = 0

eigen_numerator = 0
temp_k = 0
k = 0 

train_H_labels = np.array([])
eigen_vector = np.array([])

eigen_matrix = []

numerator = 0

target_info = 0.9
tolerance = 1e-6


#subtrafting average face
# Open the CSV file in read mode
with open('average_face.csv', 'r') as csv_file:
    # Create a CSV reader object
    csv_reader = csv.reader(csv_file)
    
    # Iterate over each row in the CSV file
    for row in csv_reader:

        # Access the values in each row by index
        # Assuming the first column contains string values and the second column contains integer values
        column1 = row[0]
        # centered_image = np.append(centered_image, float((pixel_vector[counter]-int(row[0])).astype(float))
        average_face.append( float(row[0]))

        counter = counter + 1

average_face = np.transpose(np.atleast_2d(np.array(average_face)))
low_res_averageFace = average_face 
# print(average_face)

# print("="*30)
# print(pixel_vector.shape)
# print(average_face.shape)

centered_image = pixel_vector - average_face

#from training images
#finding k using eigen value
#keeping track of the total number of rows and columns in eigenfaces.csv
with open('eigen_faces.csv', 'r') as csv_file:
    # Create a CSV reader object
    csv_reader = csv.reader(csv_file)
    
    #first row is taken next time loop will start from second row
    first_row = next( csv_reader)
    # print("first_row ", first_row )
 
    eigen_values = first_row[0].split()
    eigen_values = np.array([ float(i) for i in eigen_values ])
    eig_values_2d = np.transpose(np.atleast_2d(eigen_values))

    # print(eigen_values)
    # print(" eigen_values.shape ",eig_values_2d.shape)
    
    sum_eigen_values = sum(eigen_values)
    denominator = sum_eigen_values 
    # print(denominator)
 
    # print( eigen_values[0])
    
    #deciding values of k
    for i in range (len(eigen_values)):

        numerator = numerator + eigen_values[i]
        temp_k = numerator / denominator
        # print(" temp_k, i ",temp_k, i  )
        
        if temp_k >= 0.9:
            # print("k ",temp_k)
            k = i+1
            break
     
    eigen_list = [ [] for i in range(len(eigen_values)) ]
    # print("len of  eigen list",len(eigen_list))
    # print("Eigen list ", eigen_list)
            # skip reading values from first row
        # if first:
        #     print("row ", row)
        #     first = False
        #     continue

   #it start reading from the second row by default     
    for row in csv_reader:

        # print(row)
        #splitting the elements of first column of each row
        row_arr = row[0].split()
        #converting values in each row to float
        row_arr = np.array([ float(i) for i in row_arr ])
        # print(row_arr)
        # print(len(row_arr))

        for i in range(len(row_arr)):
            eigen_list[i].append(row_arr[i])
    
    eigen_matrix = np.transpose(np.array(eigen_list))
    # print("Eigen list ",eigen_list)     
       
    r,c = eigen_matrix.shape
    # print("Eigen matrix",eigen_matrix)   
    # print(eigen_matrix[:, 0])

    # print(r,c)
    # print("shape of  eigen matrix value of r and c ", eigen_matrix.shape[0],eigen_matrix.shape[1])
   
#saving projection co-efficients of training images in a matrix
with open('projected_coefficients.csv','r') as csv_file:
   
    csv_reader = csv.reader(csv_file)   
    # saving train labels
    #taking first row so all iterations in file later on willskip the first row
    first_row = next(csv_reader)
    train_labels = first_row[0].split()
    train_labels = np.array([ float(i) for i in train_labels ])

    train_coefficient_list = [ [] for i in range(len(train_labels)) ]

    for row in csv_reader:
        row_arr = row[0].split()
        row_arr = np.array([ float(i) for i in row_arr ])

        for i in range(len(row_arr)):
          train_coefficient_list[i].append(row_arr[i])
    
    train_coefficient_matrix = np.array(train_coefficient_list)
    # print("shape of train coefficient matrix value of r and c ", train_coefficient_matrix.shape[0],train_coefficient_matrix.shape[1])
    
    r,c = train_coefficient_matrix.shape 
    # print(r,c)
    # print(len(eigen_matrix[0]))
 
#calculating k projection co-efficients for a test image
# y_temp = np.array([])

transposed_centered_image = np.transpose(centered_image)

# print(transposed_centered_image.shape)
# print(eigen_matrix.shape)

test_projection_coefficients = np.transpose(np.matmul(transposed_centered_image, eigen_matrix))

print("="*30)
print(test_projection_coefficients.shape)   
# computing er

# test_passed = False
# test = 0 

# for i in range (train_coefficient_matrix.shape[1]):
#     #find min later on
#     if mahalanobis_dist(eigen_values, test_projection_coefficients, train_coefficient_matrix, train_coefficient_matrix.shape[0]) < .001:
#         print("Found train image in training set ", i)
#         test_passed = True
#         break
# if not test_passed:
#     print("Test not passed train image not found in training set")
#     sys.exit(-1)

# #reading low resolution test images
path_lowRes_test = "./fb_L" 
path_lowRes_train = "./fa_L"
test_images = []
temp_test_coeff = 0
r = 50
train_id = []
train_id_50 = []
test_id = []
correct_Predictions_lowRes = 0
total_low_res = 1204

#saving training data file names in an array
for file_name in os.listdir(path_lowRes_train):
    path = os.path.join(path_lowRes_train,file_name)
    img = cv2.imread(path) 
    temp = file_name.split('_')
    train_id.append(temp[0])

def get_min_sorted_values_with_indices(arr, num_values):

    # Sort the input array in ascending order
    sorted_indices = np.argsort(arr)

    # Retrieve the indices of the num_values smallest values
    min_indices = sorted_indices[:num_values]

    # Retrieve the minimum sorted values and their indices
    min_values = arr[min_indices]

    return min_values, min_indices

#for every test image in test folder compute distance and sort them
for file_name in os.listdir(path_lowRes_test):
    
    path = os.path.join(path_lowRes_test,file_name)
    img = cv2.imread(path)
    temp = file_name.split('_')
    test_id = temp[0]

    if img is not None:
        # test_images.append(img)
        # cv2.imshow("image",img)
        # cv2.waitKey(10)
        #centering each test image
        image = Image.open(path)
        img_arr = np.array(image)
        # print("img_arr ",img_arr)
        x = np.transpose(np.atleast_2d(img_arr.flatten()))
        # print("X",x)
        x_bar = low_res_averageFace
        # print("X_BAR ",x_bar)
        traspose_centered_img = np.transpose(centering_img(x, x_bar))
        # print(" traspose_centered_img ",traspose_centered_img )
        # print(traspose_centered_img.shape)# bug is here it is zero
        # print(eigen_matrix.shape)
        test_coefficients = np.transpose(np.matmul(traspose_centered_img, eigen_matrix))
        dis_arr = np.array([])
   
        # print("K ", k)
        # print(train_coefficient_matrix.shape[1])
        for i in range (train_coefficient_matrix.shape[1]):
        #find min later on
            dis = mahalanobis_dist(eig_values_2d, test_coefficients, train_coefficient_matrix, k) 
            # print(dis)
            dis_arr = np.append(dis_arr,dis)
                
    # # train_id_np = np.array(train_id)
    # dis_arr_sorted_indices = np.argsort(dis_arr)
    # # sorting the distances in ascending order and also keeping the indices for matching identification result
    # dis_arr_sorted = np.sort(dis_arr)
    # print("dis_arr_sorted",dis_arr_sorted)
    r = 50
    #we have 50 indexes and distance values that are all sorted 
    min_values, min_indices = get_min_sorted_values_with_indices(dis_arr, r)
    dis_arr_sorted_50  = min_indices


    #  #when r = 50
    # dis_arr_sorted_50 = dis_arr_sorted[:50]
    # # print("dis_arr_sorted_50",dis_arr_sorted_50)
    # dis_arr_sorted_indices_50 = np.array(dis_arr_sorted_indices[:50]) 
    train_50_list = []
    #for every index value stored in dis_ar... save corresponding train_id to train_50_list
    #1st
    input_list = train_id
    #indices
    input_indices = min_indices
    # print("len(dis_arr_sorted_indices_50)",len(dis_arr_sorted_indices_50))
    # print("len(train_id)",len(train_id))
    # train_50_list.append([input_list[i]for i in input_indices])
    # train_50_list = np.array(train_50_list )

    for i in range(len(train_id)):
        if  any(i==value for value in train_id):
            train_id_50.append(train_id[i])


    found = False 
    if any(value==test_id for value in train_id_50):
        found = True 
        correct_Predictions_lowRes = correct_Predictions_lowRes + 1
        break
  
    print("correct_Predictions_lowRes ", correct_Predictions_lowRes)
     
    #emptying array before distance calculation for next test image
    dis_arr_sorted = np.array([])
    dis_arr = np.array([])
    found = False

