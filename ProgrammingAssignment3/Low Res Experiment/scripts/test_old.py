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
import operator
import glob
import matplotlib.pyplot as plt

def centering_img(x,average_face):
    
    centered_image = []
    centered_image = x - average_face
    return centered_image

def mahalanobis_dist (eig_vals, test_coefficients, train_coefficients, k):

    # k_eig_vals = []
    # k_eig_vals = np.array( eig_vals[:k])
    # print("k_eig_vals",k_eig_vals)
    # print("k_eig_vals.shape",k_eig_vals.shape)
    # tmp1 =  np.transpose(np.atleast_2d(1/k_eig_vals))
    # tmp1 =  np.atleast_2d(1/eig_vals)
    # print(" tmp1", tmp1)
    # print(" tmp1.shape = 1/ k_eigen_values", tmp1.shape)
    # print("tmp1.shape",tmp1.shape)
    # print("train_coefficients.shape ",train_coefficients.shape)
    #there was an error here with train_coefficient slicing [:k,] 1st half means take k rows and next empty half means take all columns which is true since we are considering all columns
    # print("test_coefficients.shape",test_coefficients.shape)
    # print("train_coefficients.shape",train_coefficients.shape)
    # print("test_coefficients[0:k].shape",test_coefficients[0:k].shape)
    # print("train_coefficients[0:k].shape",np.transpose(np.atleast_2d(train_coefficients[0:k,0])).shape)
  
    tmp2 = test_coefficients[0:k]-train_coefficients[ :k,]
    # print(" tmp2", tmp2)
    # print(" tmp2.shape = k_test_coeff - k_train_coeff", tmp2.shape)
    tmp3 = np.square(tmp2)
    # print("tmp3",tmp3)
    # print("tmp3.shape = square (k_test_coeff - k_train_coeff)",tmp3.shape)
   
    # tmp4 = np.dot(np.transpose(tmp1), tmp3)
    tmp4 = tmp3 / eig_vals
    # print("tmp4",tmp4)
    # print("tmp4.shape = 1/k_eigen_values . square (k_test_coeff - k_train_coeff)",tmp4.shape)
    # return np.sum(tmp4) 
    return np.sum(tmp4) 

# print("Hello")
# with open('model/average_face.csv', 'r') as file:
#     # for i in file.rows():
#     #     print(row[])
  
#     csv_reader = csv.reader(csv_file)
    
#     # Iterate over each row in the CSV file
#     for row in csv_reader:

# image = Image.open('../data/fa_H/00023_930831_fa.pgm')
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
eigen_val_cov = []

#subtrafting average face
# Open the CSV file in read mode
with open('../model/average_face.csv', 'r') as csv_file:
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
len_eigen_values = 0
#from training images
#finding k using eigen value
#keeping track of the total number of rows and columns in eigenfaces.csv
with open('../model/eigen_faces.csv', 'r') as csv_file:
    # Create a CSV reader object
    csv_reader = csv.reader(csv_file)
    
    #first row is taken next time loop will start from second row
    first_row = next( csv_reader)
    # print("first_row ", first_row )
 
    eigen_values = first_row[0].split()
    len_eigen_values = len(eigen_values)
    eigen_values = np.array([ float(i) for i in eigen_values ])
    eigen_val_cov = eigen_values
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
    covaraince_low_res = np.cov(eigen_matrix, rowvar = False)
    # print("covaraince_low_res",covaraince_low_res)
    # print("covaraince_low_res.shape",covaraince_low_res) 

       
    r,c = eigen_matrix.shape
    # print("Eigen matrix",eigen_matrix)   
    # print(eigen_matrix[:, 0])

    # print(r,c)
    # print("shape of  eigen matrix value of r and c ", eigen_matrix.shape[0],eigen_matrix.shape[1])

# covaraince_low_res = np.zeros((len_eigen_values,len_eigen_values))
# print(covaraince_low_res)
# print(covaraince_low_res.shape) 

#saving projection co-efficients of training images in a matrix
with open('../model/projected_coefficients.csv','r') as csv_file:
   
    csv_reader = csv.reader(csv_file)   
    # saving train labels
    #taking first row so all iterations in file later on willskip the first row
    first_row = next(csv_reader)
    train_labels = first_row[0].split()
    train_labels = [ int(i) for i in train_labels ]

    train_coefficient_list = [ [] for i in range(len(train_labels)) ]

    for row in csv_reader:
        row_arr = row[0].split()
        # print("len(row_arr)",len(row_arr))
        row_arr = np.array([ float(i) for i in row_arr ])

        for i in range(len(row_arr)):
          train_coefficient_list[i].append(row_arr[i])
        # print("len(train_coefficient_list[0])",len(train_coefficient_list[0]))
    
    train_coefficient_matrix = np.transpose(np.array(train_coefficient_list))
    # print("shape of train coefficient matrix value of r and c ", train_coefficient_matrix.shape[0],train_coefficient_matrix.shape[1])
    
    r,c = train_coefficient_matrix.shape 
    # print("r,c",r,c)
    # print(len(eigen_matrix[0]))
 
#calculating k projection co-efficients for a test image
# y_temp = np.array([])
transposed_centered_image = np.transpose(centered_image)

# print(transposed_centered_image.shape)
# print(eigen_matrix.shape)

test_projection_coefficients = np.transpose(np.matmul(transposed_centered_image, eigen_matrix))

# print("="*30)
# print("test_projection_coefficients.shape",test_projection_coefficients.shape)   
# computing er

test_passed = False
test = 0 
#    train_coeff =np.transpose(np.atleast_2d(train_coefficient_matrix[:k,i])) 
#             test_coeff = np.transpose(np.atleast_2d(test_coefficients[0:k,test_img_counter]))
# print("train_coefficient_matrix.shape",train_coefficient_matrix.shape)
for i in range (train_coefficient_matrix.shape[1]):
    #find min later on
    # print(train_coefficient_matrix[:,i].shape)

    train_coeff = np.transpose(np.atleast_2d(train_coefficient_matrix[:,i]))
    # print("train_coeff.shape ", train_coeff.shape)
    test_coeff = np.transpose(np.atleast_2d(test_projection_coefficients[0:,0]))
    # print("test_coeff.shape ", test_coeff.shape)
    if mahalanobis_dist(eigen_values,  test_coeff , train_coeff , train_coefficient_matrix.shape[0]) < .001:
        # print("Found train image in training set ", i)
        test_passed = True
        break
if not test_passed:
    # print("Test not passed train image not found in training set")
    sys.exit(-1)

# #reading low resolution test images
# path_lowRes_test = "../data/fb_L/" 
# path_lowRes_train = "../data/fa_L/"

path_lowRes_test = "../data/fb_L/" 
path_lowRes_train = "../data/fa_L/"
found_counter=0

test_images = []
temp_test_coeff = 0
r = 50

train_id_50 = []
test_id = []
correct_Predictions_lowRes = 0
total_low_res = 1204
code_book = {}

results = []

#for every test image in test folder compute distance and sort them
for file_name in os.listdir(path_lowRes_test):   
    # print(path_lowRes_test + file_name)
    path = os.path.join(path_lowRes_test,file_name)
    img = cv2.imread(path)
    temp = file_name.split('_')
    test_id = int(temp[0])

    if img is not None:
        # test_images.append(img)
        # cv2.imshow("image",img)
        # cv2.waitKey(0)
        #centering each test image
        image = Image.open(path)
        img_arr = np.array(image)
        # print("img_arr ",img_arr)
        x = np.transpose(np.atleast_2d(img_arr.flatten()))
        # print("X",x)
        x_bar = low_res_averageFace
        # print("X_BAR ",x_bar)
        transpose_centered_img = np.transpose(centering_img(x, x_bar))
        # transpose_centered_img = np.transpose(x - x_bar)
        # print(" traspose_centered_img ",transpose_centered_img )
        # print(traspose_centered_img.shape)# bug is here it is zero
        # print(eigen_matrix.shape)
        test_coefficients = np.transpose(np.dot(transpose_centered_img, eigen_matrix))
        # print(test_coefficients)
        # for i in range 
        # print("test_coefficients.shape",test_coefficients.shape)
        # print("K ", k)
        # print("train_coefficient_matrix.shape",train_coefficient_matrix.shape)
        same_test_coeff_column = np.zeros((320,1204))

        for i in range(320):
            for j in range(1204):
                same_test_coeff_column[i][j] =  test_coefficients[i][0]
            if i == 19:
                break
        
        # print("same_test_coeff_column.shape ",(same_test_coeff_column[:19,]).shape)
        # print(same_test_coeff_column)

        #Alternate Mahalanobis distance
        # train = train_coefficient_matrix[:k,] #19x1204
        # test = same_test_coeff_column[:k,]
        # dis_all = []
        # eigen_k = 1 / eigen_values[:k]
        # diff = (train-test)**2
        # all_dis = np.array(np.sort(np.dot(eigen_k,diff)))
        # all_dis_indices = np.argsort(all_dis) #indices of ascending order elements
        # # print("all_dis.shape",all_dis.shape)
        # selected_ids = []
        
        # for i in range (len(train_labels)):
        #    if i in all_dis_indices[:r]:
        #        selected_ids.append(train_labels[i])
        #        if test_id in selected_ids:
        #            print("Found")
        #            found_counter += 1
        #            break
     
        for i in range (train_coefficient_matrix.shape[1]):
        #find min later on
            train_coeff =np.transpose(np.atleast_2d(train_coefficient_matrix[:k,i])) 
            # print("train_coeff.shape",train_coeff.shape)
            test_coeff = np.transpose(np.atleast_2d(test_coefficients[0:k,0]))

            # print("test_coeff.shape",test_coeff.shape)
            # tmp2 = test_coefficients[0:k]-train_coefficients[ :k,]
            # dis = mahalanobis_dist = np.sqrt(np.dot(np.transpose((test_coeff - train_coefficient_matrix)), np.linalg.inv(covaraince_low_res) ,(test_coeff - train_coeff)))
            
            dis = mahalanobis_dist(eigen_values[:k], test_coeff , train_coeff , k) 
            # print(dis)
            # code_book[train_labels[i]] = dis[0][0]
            if train_labels[i] in code_book.keys() and dis < code_book[train_labels[i]]:
                code_book[train_labels[i]] = dis
            elif not (train_labels[i] in code_book.keys()):
                code_book[train_labels[i]] = dis
           
            # print("test_coefficients.shape",test_coefficients.shape )
            # print(" test_coeff.shape ",test_coeff.shape) 
            # print(dis)
            # dis_arr = np.append(dis_arr,dis)

        # print(code_book)

        sorted_code_book = sorted(code_book.items(), key=lambda kv: kv[1])

        results.append([test_id, sorted_code_book[:50]])

        # print(sorted_code_book)
        # print(sorted_code_book[0])
        # print(code_book[sorted_code_book[0]])
        # break

# print(len(results))

num_correct = [0 for i in range(50)]       
for r in range (50):
    
    for i in range (len(results)):
        for j in range(r):
            # print('='*30)
            # print(results[i][1][j][0])
            # print(results[i][0])
            if results[i][1][j][0]==results[i][0]:
                num_correct[r] += 1

                break
    # print(num_correct)
    num_correct[r] /= len(results)
# print(num_correct)
plt.plot(range(50), num_correct, 'b', label = "derfew")
plt.show()

# test id
# print(results[0][0])##################################### 1
# key value pair for the 0th element of sorted code book
# print(results[0][1][0])
# key for the 0th element of the sorted code book
# [0][1][0][0] = image_no, codebook_no, codebook_pair_index(which training image), inside a cell of codebook 0 = key 1 = distnace
# print(results[0][1][0][0])################################# 2
# print(results[image_number][1][0][0])
# compare 1 with 2

# print("new_result ", (found_counter/1204) * 100)


# pick r

# check if the first r have test_id as a label

# mark correct if so

# calculate proprotion correct
      
                
    # # train_id_np = np.array(train_id)
    # dis_arr_sorted_indices = np.argsort(dis_arr)
    # # sorting the distances in ascending order and also keeping the indices for matching identification result
    # dis_arr_sorted = np.sort(dis_arr)
    # print("dis_arr_sorted",dis_arr_sorted)
    # r = 50
    # #we have 50 indexes and distance values that are all sorted 
    # # min_values, min_indices = get_min_sorted_values_with_indices(dis_arr, r)
    # # dis_arr_sorted_50  = min_indices


    # #  #when r = 50
    # # dis_arr_sorted_50 = dis_arr_sorted[:50]
    # # # print("dis_arr_sorted_50",dis_arr_sorted_50)
    # # dis_arr_sorted_indices_50 = np.array(dis_arr_sorted_indices[:50]) 
    # train_50_list = []
    # #for every index value stored in dis_ar... save corresponding train_id to train_50_list
    # #1st
    # input_list = train_id
    # #indices
    # input_indices = min_indices
    # # print("len(dis_arr_sorted_indices_50)",len(dis_arr_sorted_indices_50))
    # # print("len(train_id)",len(train_id))
    # # train_50_list.append([input_list[i]for i in input_indices])
    # # train_50_list = np.array(train_50_list )

    # for i in range(len(train_id)):
    #     if  any(i==value for value in train_id):
    #         train_id_50.append(train_id[i])


    # found = False 
    # if any(value==test_id for value in train_id_50):
    #     found = True 
    #     correct_Predictions_lowRes = correct_Predictions_lowRes + 1
    #     break
  
    # print("correct_Predictions_lowRes ", correct_Predictions_lowRes)
     
    # #emptying array before distance calculation for next test image
    # dis_arr_sorted = np.array([])
    # dis_arr = np.array([])
    # found = False

