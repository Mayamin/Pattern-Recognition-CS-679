import numpy as np
import csv
import PIL
from PIL import Image
import math

def read_csv_to_matrix(file_path):

    matrix = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            matrix.append(row)
    return matrix


# print("Hello")
# with open('model/average_face.csv', 'r') as file:
#     # for i in file.rows():
#     #     print(row[])
  
#     csv_reader = csv.reader(csv_file)
    
#     # Iterate over each row in the CSV file
#     for row in csv_reader:

image = Image.open('00001_930831_fa_a.pgm')

#convert image to numpy array
image_array = np.array(image)

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


########################################################################################  

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
        centered_image = np.append(centered_image, (pixel_vector[counter]-int(row[0])).astype(float))

        counter = counter + 1


########################################################################################  

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

    # print(eigen_values)
    # print(eigen_values.shape)
    
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
            print("k ",temp_k)
            k = i+1
            break
    
    
    
    eigen_list = [ [] for i in range(len(eigen_values)) ]
    print("len of  eigen list",len(eigen_list))
    
    # first = True
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

        for i in range(len(row_arr)):
            # np.append(eigen_matrix[i], row_arr)
         eigen_list[i].append(row_arr)
       
    eigen_matrix = np.array(eigen_list)
    r,c,d = eigen_matrix.shape
    print(r,c,d)
    print("shape of  eigen matrix value of r and c ", eigen_matrix.shape[0],eigen_matrix.shape[1])
########################################################################################    
   
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
          train_coefficient_list[i].append(row_arr)
    
    train_coefficient_matrix = np.array(train_coefficient_list)
    print("shape of train coefficient matrix value of r and c ", train_coefficient_matrix.shape[0],train_coefficient_matrix.shape[1])
    #why do we have a three dimension matrix
    r,c,d= train_coefficient_matrix.shape 
    print(r,c,d)
    # print(len(eigen_matrix[0]))
  
########################################################################################  

#calculating k projection co-efficients for a test image
# y_temp = np.array([])
y = []
y_temp = 0
transposed_centered_image = np.transpose(centered_image)

for i in range (len(eigen_matrix[0])):
    
    y_temp = np.matmul(transposed_centered_image,eigen_matrix[i])
    y.append(y_temp)
  
test_projection_coefficients = y

print(len(y))   


############################################################################################
#computing er
er = []

for i in range (len(test_projection_coefficients)):
 er.append(abs(test_projection_coefficients-train_coefficient_matrix[i]))










# vector = np.empty(eigen_matrix.shape[0])

# # Iterate through each column in the matrix
# for col_idx in range(eigen_matrix.shape[1]):
#     # Extract the row eigen_matrixvalues for the current column
#     col_values = eigen_matrix[:, col_idx]
    
#     # Add the extracted values to the vector
#     vector = np.vstack((vector, col_values))

# # Remove the first row of the vector (which was used for initialization)
# vector = np.delete(vector, 0, axis=0)

# # Transpose the vector to convert it into a column vector
# vector = vector.T
# vector = np.array(vector)
# vector  = np.delete(vector , 0, axis=0)
# row, col = vector.shape
# # print("matrix row and column number ", row, col)

# c = centered_image.shape
# # print(c)

# transposed_centered_image = np.transpose(centered_image)
# # print(transposed_centered_image.shape)

# # print("type", type(transposed_centered_image ))

# test_eigen_coefficients = []
# eigen_multiply = []
# # print(k)
# for value in range (k):
#     # test_eigen_coefficients[k] = np.matmul(transposed_centered_image,vector[k])
#     test_eigen_coefficients[k] = np.multiply(transposed_centered_image.astype(float),vector[k].astype(float))


    
    
     















 

        
    # print(" len(eigen_values) ", len(eigen_values))        
    # print(k)














    
#     # Convert the CSV reader object to a list
#     csv_list = list(csv_reader)
    
#     # Count the number of rows and columns
#     num_rows = len(csv_list)
#     num_cols = len(csv_list[0])

#     # Get the first row of the CSV file
#     eigen_values = next(csv_reader)
#     print(eigen_values)
#     # sum_of_all_eigen_values = sum(eigen_values)
#     # denominator = sum_of_all_eigen_values
#     # print("Type", type())
#     numerator = 0
#     temp_k = 0
#     k = 0 

#     # for column_value in eigen_values:       
#     #     numerator = numerator + column_value 
#     #     print("column_value Type", type(column_value))
#     #     # temp_k = numerator/denominator
#     #     if(temp_k==.8):
#     #        k = temp_k

# print("Value of k for experiment 3 a ii is ", k)
# # Print the total number of rows and columns
# print(f"The eigen_faces.csv file has {num_rows} rows and {num_cols} columns.")

# with open('eigen_faces.csv', 'r') as csv_file:
#     # Create a CSV reader object
#     csv_reader = csv.reader(csv_file)
    
#     # Get the first row of the CSV file
#     first_row = next(csv_reader)
    
#     # Convert the first row to a NumPy array
#     first_row_array = np.array(first_row)

# print(first_row_array)



#calculating k from eigen values for experiment 3 (a.II)




# print("centered_image ", centered_image)

 


