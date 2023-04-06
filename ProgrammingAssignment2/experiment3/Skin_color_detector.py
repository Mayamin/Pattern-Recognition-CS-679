import time
import math
import numpy as np
import cv2
import warnings
import scipy
import random
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

warnings.filterwarnings("ignore", category=RuntimeWarning)
#open cv reads in the order BGR

# def compute_covariance(N, x, mu):
#    for i in range(N):
#       sigma = (x-mu) * np.transpose(x-mu)

# Calculate the Gaussian probability density function for a given x, mean (mu) and variance (var)
def gaussian_pdf(x, mu, cov):

   #print(str(np.sqrt(((2*math.pi)**2) *  np.linalg.det(cov))))

   # print("="*30)
   # print(str(((np.transpose(x-mu) * np.linalg.inv(cov)))))
   # print("="*30)

   pdf = (1/ np.sqrt(((2*math.pi)**2) *  np.linalg.det(cov))) * np.exp((-0.5) *  np.transpose(x-mu).dot( np.linalg.inv(cov).dot((x-mu)))) 
     
   return pdf

#read training image and convert from bgr to rgb
path = r'/home/mraha/PA_2/P2_Data/Data_Prog2/Training_1.ppm'
train_bgr = cv2.imread(path) 
train_rgb = cv2.cvtColor(train_bgr, cv2.COLOR_BGR2RGB)

#read reference training image and convert from bgr to rgb
path_2_Ref = r'/home/mraha/PA_2/P2_Data/Data_Prog2/ref1.ppm'
train_ref_bgr = cv2.imread(path_2_Ref) 
train_ref_rgb = cv2.cvtColor(train_ref_bgr, cv2.COLOR_BGR2RGB)

#find out size of original image and create a tensor with three channels 
width,height = train_rgb.shape[0], train_rgb.shape[1]
ground_truth_1 = np.zeros((width,height))
chromatic = np.zeros((train_rgb.shape[0],train_rgb.shape[1],train_rgb.shape[2]))

#r g b channel values for red and green
red = np.array([3, 3, 255])
white = np.array([255, 255, 255])
black = np.array([0,0,0])

red_channel = np.array([])
green_channel = np.array([])
vector = np.vectorize(np.double)

# saving values in chromatic in RGB order
for i in range(width):
  for j in range(height):
    
    comparison_1 = train_ref_rgb[i][j] == red
    comparison_2 = train_ref_rgb[i][j] == white
    
    #denominator = (train_rgb[i,j][0] + train_rgb[i,j][1] + train_rgb[i,j][2])
    denominator = sum(train_rgb[i,j])
    if denominator == 0:
       denominator = 0.0000001
      #  print("encountered divide by zero")
     

    chromatic[i,j,0] = np.divide(train_rgb[i,j][0], denominator)
    chromatic[i,j,1] = np.divide(train_rgb[i,j][1], denominator)

    #keeping the face pixel values in red and green channel and marking these locations as 1 in groud truth array
    if (comparison_1.all() ) or ( comparison_2.all()):

      # numerator = train_rgb[i,j][1] 
      red_channel = np.append(red_channel, chromatic[i,j,0])
      # numerator = train_rgb[i,j][1] 
      green_channel = np.append(green_channel, chromatic[i,j,1])
      
      #1 = face
      ground_truth_1[i,j] = 1
      
      #debuggin print statements
      if green_channel[-1] > 1 or red_channel[-1] > 1:
        print("="*30)
        print(f'{train_rgb[i,j][0]} + {train_rgb[i,j][1]} + {train_rgb[i,j][2]} == {train_bgr[i,j][0]} + {train_bgr[i,j][1]} + {train_bgr[i,j][2]}')
      #   print(numerator)
        print("denominator ", denominator)
        print("Red channel values in face region ",red_channel)
        print("Green channel values in face region ", green_channel)
        print("="*30)
    
     
# print("lenght of red = length of green channel  = N", red_channel.size, green_channel.size)
print("Conversion to chromatic space done! ")
N = red_channel.size

#Compute Mean
red_channel_mean = np.mean(red_channel, dtype = np.float64)
green_channel_mean = np.mean(green_channel, dtype = np.float64)
train_mean = np.array([red_channel_mean,green_channel_mean])
print("Mean calculated! ")

# Compute Covariance 
covariance_matrix = np.array(np.cov(red_channel, green_channel))
print("Covariance Calculated! ")

constant_factor = 1 / ((2*math.pi) * math.sqrt(np.linalg.det(covariance_matrix)))
step_size = constant_factor/20 
thresholds = []
thresholds = [0 for i in range(20)] 
number_of_thresholds = 20

print("20 thresholds created! ")

for i in range (1,20,1):
    thresholds[i] = (thresholds[i-1] + step_size)

print("search for best threshold started")

#finding optimum threshold at equal error rate
temp_labels = np.array([])
result = np.zeros(train_rgb.shape[0])
# TP = 1, TN =2, FP = 3, FN = 4
# rows = width
# cols = height
# my_array = []
# for i in range(rows):
#     row = []
#     for j in range(cols):
#         row.append([0] * 20)
#     my_array.append(row)

FPR_list = []
FNR_list = []

FP_count = FN_count = TP_count = TN_count = 0
#for every threshold in the image find FP FN TP TN values
for t in range (20):
   
   label = 0

   for i in range(width):
     for j in range(height):
        
        # x = [r,g]
        x = np.array([chromatic[i,j,0],chromatic[i,j,1]])
        mew = np.array(train_mean)

        if gaussian_pdf(x, mew,covariance_matrix) > thresholds[t]:
          label = 1
        # print(" i j ", i, j)
        #TP
        if ground_truth_1[i,j] == 1 and label == 1:
          # my_array[i][j][t] = 1
          TP_count = TP_count + 1

        # #TN   
        elif ground_truth_1[i,j] == 0 and label == 0:
          # my_array[i][j][t] = 2
          TN_count = TN_count + 1

        #FP   
        elif ground_truth_1[i,j] == 0 and label == 1:
          # my_array[i][j][t] = 3
          FP_count = FP_count + 1

        #FN   
        else:
          # my_array[i][j][t] = 4
          FN_count = FN_count + 1

  #  print(" :( ")
   FPR_list = np.append(FPR_list,FP_count/(FP_count+TN_count))
   FNR_list = np.append(FNR_list,FN_count/(FN_count+TP_count))
   FN_count = FP_count = TP_count = TN_count = 0


# print("Size of FPR_list ", len(FPR_list))
# print("Size of FNR_list", len(FPR_list))
best_threshold_index = 0
min = FPR_list[0]-FNR_list[0]

for i in range (1,20,1):
  if abs(FPR_list[i]-FNR_list[i]) < min:
     best_threshold_index = i
  

best_threshold = thresholds[best_threshold_index]
print("Best threshold found ",best_threshold )

#read test image

#read test image 1 and convert from bgr to rgb
path_2 = r'/home/mraha/PA_2/P2_Data/Data_Prog2/Training_3.ppm'
test1_bgr = cv2.imread(path_2) 
test1_rgb = cv2.cvtColor(train_bgr, cv2.COLOR_BGR2RGB)

#read reference test image 1 and convert from bgr to rgb
path_22_Ref = r'/home/mraha/PA_2/P2_Data/Data_Prog2/ref3.ppm'
test1_ref_bgr = cv2.imread(path_22_Ref) 
test1_ref_rgb = cv2.cvtColor(test1_ref_bgr, cv2.COLOR_BGR2RGB)       
           
#read test image 2 and convert from bgr to rgb
path_3 = r'/home/mraha/PA_2/P2_Data/Data_Prog2/Training_6.ppm'
test2_bgr = cv2.imread(path) 
test2_rgb = cv2.cvtColor(test2_bgr, cv2.COLOR_BGR2RGB)

#read reference test image 2 and convert from bgr to rgb
path_22_Ref = r'/home/mraha/PA_2/P2_Data/Data_Prog2/ref6.ppm'
test2_ref_bgr = cv2.imread(path_22_Ref) 
test2_ref_rgb = cv2.cvtColor(test2_ref_bgr, cv2.COLOR_BGR2RGB)



ground_truth_t1 = np.zeros((width,height))
test1_label = 0
test1_FP_count = test1_FN_count = test1_TP_count = test1_TN_count = 0
test1_FP_list = test1_FN_list = test1_TP_list = test1_TN_list = np.array([])

y_true = np.array([])
y_scores = np.array([])

#classifying test image 
for i in range(width):
  for j in range(height):
     
    comparison_1 = train_ref_rgb[i][j] == red
    comparison_2 = train_ref_rgb[i][j] == white
       
    x_test = np.array([test1_rgb[i,j][0], test1_rgb[i,j][1]])

    #experimental 255 if white 0 if black
    # y_true = np.append(y_true,test1_rgb[i,j][0])

    #assiging groud_ truth values to a 2D array
    if (comparison_1.all() ) or ( comparison_2.all()):
       ground_truth_t1[i,j] = 1
       y_true = np.append(y_true,1)
    else:
      y_true = np.append(y_true,0)

    #classification
    if gaussian_pdf(x_test, mew,covariance_matrix) > best_threshold:
      
      test1_label = 1
      y_scores = np.append(y_scores,1)
    
    else:
      y_scores = np.append(y_scores,0)

    #     #TP
    # if ground_truth_t1[i,j] == 1 and test1_label == 1:
    #       # my_array[i][j][t] = 1
    #   test1_TP_count = test1_TP_count + 1
    #     #TN   
    # elif ground_truth_t1[i,j] == 0 and test1_label == 0:
    #       # my_array[i][j][t] = 2
    #   test1_TN_count = test1_TN_count + 1

    #     #FP   
    # elif ground_truth_t1[i,j] == 0 and test1_label == 1:
       
    #   test1_FP_count = test1_FP_count + 1

    #     #FN   
    # elif ground_truth_t1[i,j] == 1 and test1_label == 0:
         
    #   test1_FN_count = test1_FN_count + 1
    
    # #only keeping the the estimated face pixels in test image and making everything else black 
    # if ground_truth_t1 != test1_label:
    #    test1_rgb[i,j] = np.array([0,0,0]) 

    # test1_FP_list = np.append(test1_FP_list,test1_FP_count)
    # test1_FN_list = np.append(test1_FN_list,test1_FN_count)
    # test1_TP_list = np.append(test1_TP_list,test1_TP_count)
    # test1_TN_list = np.append(test1_TN_list,test1_TN_count)




# test_FPR = test1_FP_count / (test1_FP_count + test1_TN_count)
# test_FNR = test1_FN_count / (test1_FN_count + test1_TP_count)

sorted_indices = np.argsort(y_scores)[::-1]
y_true_sorted = y_true[sorted_indices]
y_scores_sorted = y_scores[sorted_indices]

# Initialize the true positive rate and false positive rate arrays
tpr = [0]
fpr = [0]

# Initialize the number of true positives and false positives to 0
tp = 0
fp = 0

# Iterate through each predicted score and update tp and fp accordingly
for i in range(len(y_scores_sorted)):
    if y_true_sorted[i] == 1:
        tp += 1
    else:
        fp += 1
    tpr.append(tp / sum(y_true))
    fpr.append(fp / sum(1 - y_true))

# Plot the ROC curve
plt.plot(fpr, tpr)
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()



start = time.time()

# print(23*2.3)

end = time.time()
print("time elapsed ",end - start)






# print("FPR_list ",FPR_list)
# print("FNR_list ",FNR_list)

# plt.scatter(FPR_list,FNR_list)
# plt.title('Scatter Plot of FP vs FN')
# plt.xlabel('FP')
# plt.ylabel('FN')
# plt.savefig('ROC_curve.png')
# plt.show()








        #print("mu ", mew)

# optimum_t = np.argmax(equal_error_t) 
# print("optimum t found to be ",optimum_t)
# equal_error_t =    np.empty(equal_error_t)     
# labels_1 = np.zeros((width,height))
# print("classification started")
# #classification with optimum threshold value
# for i in range(width):
#     for j in range(height):
       
#       x = np.array([chromatic[i,j,0],chromatic[i,j,1]])
#       mew = np.array(train_mean)
       
#       #assigning label value as 1 face  
#       if gaussian_pdf(x,mew,covariance_matrix) > optimum_t:
#        labels_1[i,j] = 1
      
#       #for each pixel checking assigned label with ground truth
#       if (ground_truth_1[i,j] == 0 ) and (labels_1[i,j] == 1):
#        FP = FP+1
#        FP_plot = np.append(FP_plot,FP)

#       elif(ground_truth_1[i,j] == 1 ) and (labels_1[i,j] == 0):
#        FN = FN + 1
#        FN_plot = np.append(FN_plot,FN)

# print("Classification done!")
# plt.scatter(FP_plot, FN_plot, c= "blue")
# plt.show()
 
# plt.scatter(FP_plot, FN_plot,c = "blue")
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.savefig('ROC_curve.png')
# plt.show()

# start = time.time()

# print(23*2.3)

# end = time.time()
# print("time elapsed ",end - start)

