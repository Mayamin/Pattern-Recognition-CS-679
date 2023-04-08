import time
import math
import numpy as np
import cv2
import warnings
import scipy
import random
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import roc_curve
warnings.filterwarnings("ignore", category=RuntimeWarning)
#open cv reads in the order BGR

# Calculate the Gaussian probability density function for a given x, mean (mu) and variance (var)
def gaussian_pdf(x, mu, lhs, cov_inv):
  #  print(lhs)
  #  print(x)
  #  print(mu)
  #  print(cov_inv)

   pdf = lhs * np.exp((-0.5) *  np.dot(np.transpose(x-mu), ( np.dot(cov_inv, ((x-mu)))))) 
   
   return pdf

def calc_mean( x: list):
  sum = 0
  for i in x:
    sum += x
  return sum / len(x)

def calc_covariance(x, mu):
  sum = 0

  for i in len(x[0]):
    nx = np.array(x[0][i], x[1][i])
    sum += np.dot((nx - mu), ( np.transpose(nx - mu)))

  return sum / len(x[0])

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
yCC_clr_space = np.zeros((train_rgb.shape[0],train_rgb.shape[1],train_rgb.shape[2]))

#r g b channel values for red and green
red = np.array([3, 3, 255])
white = np.array([255, 255, 255])
black = np.array([0,0,0])

red_channel = np.array([])
green_channel = np.array([])
cB_channel = np.array([])
cR_channel = np.array([])
vector = np.vectorize(np.double)

#read test image 1 and convert from bgr to rgb
path_2 = r'/home/mraha/PA_2/P2_Data/Data_Prog2/Training_3.ppm'
test1_bgr = cv2.imread(path_2) 
test1_rgb = cv2.cvtColor(test1_bgr, cv2.COLOR_BGR2RGB)

test1_chromatic = np.zeros((test1_rgb.shape[0],test1_rgb.shape[1],test1_rgb.shape[2]))
test1_yCC = np.zeros((test1_rgb.shape[0],test1_rgb.shape[1],test1_rgb.shape[2]))

yCC_path_2 = r'/home/mraha/PA_2/P2_Data/Data_Prog2/yCC_train/Training_3.ppm'
yCC_test1_bgr =cv2.imread(yCC_path_2)
yCC_test1_rgb = cv2.cvtColor(yCC_test1_bgr, cv2.COLOR_BGR2RGB)

#read reference test image 1 and convert from bgr to rgb
path_22_Ref = r'/home/mraha/PA_2/P2_Data/Data_Prog2/ref3.ppm'
test1_ref_bgr = cv2.imread(path_22_Ref) 
test1_ref_rgb = cv2.cvtColor(test1_ref_bgr, cv2.COLOR_BGR2RGB)       
ground_truth_t1 = np.zeros((test1_ref_rgb.shape[0],test1_ref_rgb.shape[1]))

#read test image 2 and convert from bgr to rgb
path_3 = r'/home/mraha/PA_2/P2_Data/Data_Prog2/Training_6.ppm'
test2_bgr = cv2.imread(path_3) 
test2_rgb = cv2.cvtColor(test2_bgr, cv2.COLOR_BGR2RGB)

###########
# yCC_test2_rgb = cv2.cvtColor(test2_bgr, cv2.COLOR_BGR2RGB)
####

test2_chromatic = np.zeros((test2_rgb.shape[0],test2_rgb.shape[1],test2_rgb.shape[2]))
test2_yCC = np.zeros((test2_rgb.shape[0],test2_rgb.shape[1],test2_rgb.shape[2]))

yCC_path_3 = r'/home/mraha/PA_2/P2_Data/Data_Prog2/yCC_train/Training_6.ppm'
yCC_test2_bgr =cv2.imread(yCC_path_3)
yCC_test2_rgb = cv2.cvtColor(yCC_test2_bgr, cv2.COLOR_BGR2RGB)

#read reference test image 2 and convert from bgr to rgb
path_33_Ref = r'/home/mraha/PA_2/P2_Data/Data_Prog2/ref6.ppm'
test2_ref_bgr = cv2.imread(path_33_Ref) 
test2_ref_rgb = cv2.cvtColor(test2_ref_bgr, cv2.COLOR_BGR2RGB)
ground_truth_t2 = np.zeros((test2_ref_rgb.shape[0],test2_ref_rgb.shape[1]))

# saving values in chromatic in RGB order
for i in range(width):
  for j in range(height):
    
    comparison_1 = train_ref_rgb[i][j] == red
    comparison_2 = train_ref_rgb[i][j] == white
    
    test1_comparison_1 = test1_ref_rgb[i][j] == red 
    test1_comparison_2 = test1_ref_rgb[i][j] == white

    test2_comparison_1 = test2_ref_rgb[i][j] == red 
    test2_comparison_2 = test2_ref_rgb[i][j] == white
   
    #train image convert to chromatic space 
    denominator = sum(train_rgb[i,j])
    if denominator == 0:
       denominator = 0.0000001
     
    chromatic[i,j,0] = np.divide(train_rgb[i,j][0], denominator)
    chromatic[i,j,1] = np.divide(train_rgb[i,j][1], denominator)

    #train image convert to YCC color space
    #Cb
    yCC_clr_space[i,j,1] = (-0.169 * train_rgb[i,j][0]) - (0.332 * train_rgb[i,j][1]) + (0.500 * train_rgb[i,j][2])
    #Cr
    yCC_clr_space[i,j,2] = ( 0.500 * train_rgb[i,j][0]) - (0.419 * train_rgb[i,j][1]) - (0.081 * train_rgb[i,j][2])
    
    #test1 image convert to chromatic space
    test1_denominator = sum(test1_rgb[i,j])
    if test1_denominator == 0:
       test1_denominator = 0.0000001

    test1_chromatic[i,j,0] =np.divide(test1_rgb[i,j][0],test1_denominator)
    test1_chromatic[i,j,1] =np.divide(test1_rgb[i,j][1],test1_denominator)

    #test1 image convert to yCC space
    test1_yCC[i,j,1] = (-0.169 * test1_rgb[i,j][0]) - (0.332 * test1_rgb[i,j][1]) + (0.500 * test1_rgb[i,j][2])
    test1_yCC[i,j,2] = ( 0.500 * test1_rgb[i,j][0]) - ( 0.419 * test1_rgb[i,j][1]) - (0.081 * test1_rgb[i,j][2])

    #test2 image convert to chromatic space
    test2_denominator = sum(test2_rgb[i,j])
    if test2_denominator == 0:
       test2_denominator = 0.0000001

    test2_chromatic[i,j,0] =np.divide(test2_rgb[i,j][0],test2_denominator)
    test2_chromatic[i,j,1] =np.divide(test2_rgb[i,j][1],test2_denominator)

    #test2 image convert to yCC space
    test2_yCC[i,j,1] = (-0.169 * test2_rgb[i,j][0]) - (0.332 * test2_rgb[i,j][1]) + (0.500 * test2_rgb[i,j][2])
    test2_yCC[i,j,2] = ( 0.500 * test2_rgb[i,j][0]) - ( 0.419 * test2_rgb[i,j][1]) - (0.081 * test2_rgb[i,j][2])
    
    #for train image
    #keeping the face pixel values in red and green channel and marking these locations as 1 in groud truth array
    if (comparison_1.all() ) or (comparison_2.all()):

      #from chromatic space
      red_channel = np.append(red_channel, chromatic[i,j,0])
      green_channel = np.append(green_channel, chromatic[i,j,1])
      #from yCC space
      cB_channel = np.append(cB_channel, yCC_clr_space[i,j,1])
      cR_channel = np.append(cR_channel, yCC_clr_space[i,j,2])

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

    
    #for test image mark face
    if(test1_comparison_1.all()) or (test1_comparison_2.all()):
      ground_truth_t1 [i,j] = 1
    
    #for test image mark face
    if(test2_comparison_1.all()) or (test2_comparison_2.all()):
      ground_truth_t2 [i,j] = 1
   
# print("lenght of red = length of green channel  = N", red_channel.size, green_channel.size)
# print("Conversion to chromatic and yCC space done! ")
N = red_channel.size

#Compute Mean
red_channel_mean = np.mean(red_channel, dtype = np.float64)
green_channel_mean = np.mean(green_channel, dtype = np.float64)
train_mean = np.array([red_channel_mean,green_channel_mean])

cB_channel_mean = np.mean(cB_channel,dtype=np.float64)
cR_channel_mean = np.mean(cR_channel,dtype=np.float64)
yCC_train_mean = np.array([cB_channel_mean,cR_channel_mean])
print("Mean for chromatic and yCC color space calculated! ")

# Compute Covariance 
covariance_matrix = np.array(np.cov(red_channel, green_channel))

yCC_covariance_matrix = np.array(np.cov(cB_channel, cR_channel))
print("Covariance matrix for chromatic and yCC color space calculated! ")

#creating threshold chromatic
constant_factor = 1 / ((2*math.pi) * math.sqrt(np.linalg.det(covariance_matrix)))
step_size = constant_factor/20 
thresholds = []
thresholds = [0 for i in range(20)] 
number_of_thresholds = 20

#creating threshold yCC
yCC_constant_factor = 1 / ((2*math.pi) * math.sqrt(np.linalg.det(yCC_covariance_matrix)))
yCC_step_size = yCC_constant_factor/20 
yCC_thresholds = []
yCC_thresholds = [0 for i in range(20)] 
yCC_number_of_thresholds = 20

for i in range (1,20,1):
    thresholds[i] = (thresholds[i-1] + step_size)
    yCC_thresholds[i] = (yCC_thresholds[i-1] + yCC_step_size)
print("20 thresholds created! ")

#finding optimum threshold at equal error rate
temp_labels = np.array([])
result = np.zeros(train_rgb.shape[0])

#for chromatic space
#for train images
FPR_list = [0]*20
FNR_list = [0]*20
FP_count = FN_count = TP_count = TN_count = 0

#for test image1
test1_label = 0
test1_FP_count = test1_FN_count = test1_TP_count = test1_TN_count = 0
test1_FPR_list = [0]*20
test1_FNR_list = [0]*20

#for test image2
test2_label = 0
test2_FP_count = test2_FN_count = test2_TP_count = test2_TN_count = 0
test2_FPR_list = [0]*20
test2_FNR_list = [0]*20


#for yCC color space
#for train images
yCC_FPR_list = [0]*20
yCC_FNR_list = [0]*20
yCC_FP_count = yCC_FN_count = yCC_TP_count = yCC_TN_count = 0

#for test image1
yCC_test1_label = 0
yCC_test1_FP_count = yCC_test1_FN_count = yCC_test1_TP_count = yCC_test1_TN_count = 0
yCC_test1_FPR_list = [0]*20
yCC_test1_FNR_list = [0]*20

#for test image2
yCC_test2_label = 0
yCC_test2_FP_count = yCC_test2_FN_count = yCC_test2_TP_count = yCC_test2_TN_count = 0
yCC_test2_FPR_list = [0]*20
yCC_test2_FNR_list = [0]*20


#forward calculation of variables so we dont have to do it more than once
#for chromatic space
cov_det = np.linalg.det(covariance_matrix)
cov_inv = np.linalg.inv(covariance_matrix)
lhs = 1/ np.sqrt(((2*math.pi)**2) * cov_det)

#for yCC space
yCC_cov_det = np.linalg.det(yCC_covariance_matrix)
yCC_cov_inv = np.linalg.inv(yCC_covariance_matrix)
yCC_lhs = 1/np.sqrt(((2*math.pi)**2) * yCC_cov_det)


#train images
#for every threshold in the image find FP FN TP TN values
for t in range (20):
  label = 0
  yCC_label = 0

  for i in range(width):
     for j in range(height):
        
        ########train image
        # x = [r,g]
        x = np.array([chromatic[i,j,0],chromatic[i,j,1]])
        mew = np.array(train_mean)

        yCC_x = np.array([yCC_clr_space[i,j,1],yCC_clr_space[i,j,2]])
        yCC_mew = np.array(yCC_train_mean)

        if gaussian_pdf(x, mew, lhs, cov_inv) > thresholds[t]:
          label = 1
        else:
          label = 0 
        
        if gaussian_pdf(yCC_x, yCC_mew, yCC_lhs, yCC_cov_inv) > yCC_thresholds[t]:
          yCC_label = 1
        else:
          yCC_label = 0 

               
        #for chromatic color space
        if ground_truth_1[i,j] == 1 and label == 1:
          TP_count = TP_count + 1
        
        elif ground_truth_1[i,j] == 0 and label == 0:
          TN_count = TN_count + 1
        
        elif ground_truth_1[i,j] == 0 and label == 1:
          FP_count = FP_count + 1
        
        else:
          FN_count = FN_count + 1
                
        #for yCC color space
        if ground_truth_1[i,j] == 1 and yCC_label == 1:
          yCC_TP_count = yCC_TP_count + 1
        
        elif ground_truth_1[i,j] == 0 and yCC_label == 0:
          yCC_TN_count = yCC_TN_count + 1
        
        elif ground_truth_1[i,j] == 0 and yCC_label == 1:
          yCC_FP_count = yCC_FP_count + 1
        
        else:
          yCC_FN_count = yCC_FN_count + 1
        
        #######test1 image
        x_t1 = np.array([test1_chromatic[i,j,0],test1_chromatic[i,j,1]])
        # print("x_t1",x_t1)
        yCC_x_t1 = np.array([test1_yCC[i,j,1],test1_yCC[i,j,2]])
        # print("yCC_x_t1 ",yCC_x_t1)
        #assigning labels for chromatic space
        if gaussian_pdf(x_t1,mew, lhs, cov_inv)>thresholds[t]:
          test1_label = 1
        else:
          # test1_rgb[i,j] = [255,255,255]
          test1_label = 0
        
        #classification for yCC space
        if gaussian_pdf(yCC_x_t1,yCC_mew, yCC_lhs, yCC_cov_inv)>yCC_thresholds[t]:
          yCC_test1_label = 1
        else:
          # yCC_test1_rgb[i,j] = [255,255,255]
          yCC_test1_label = 0

        #compute FN FP TN TP for first test image in chromatic space
        if ground_truth_t1[i,j] == 1 and test1_label == 1:
          test1_TP_count = test1_TP_count + 1
        elif ground_truth_t1[i,j] == 0 and test1_label == 0:
          test1_TN_count = test1_TN_count + 1
        elif ground_truth_t1[i,j] == 0 and test1_label == 1:
          test1_FP_count = test1_FP_count + 1
        else:
          test1_FN_count = test1_FN_count + 1
        
        #compute FN FP TN TP for first test image in yCC space
        if ground_truth_t1[i,j] == 1 and yCC_test1_label == 1:
          yCC_test1_TP_count = yCC_test1_TP_count + 1
        elif ground_truth_t1[i,j] == 0 and yCC_test1_label == 0:
          yCC_test1_TN_count = yCC_test1_TN_count + 1
        elif ground_truth_t1[i,j] == 0 and yCC_test1_label == 1:
          yCC_test1_FP_count = yCC_test1_FP_count + 1
        else:
          yCC_test1_FN_count = yCC_test1_FN_count + 1


        #######test2 image
        x_t2 = np.array([test2_chromatic[i,j,0],test2_chromatic[i,j,1]])
        yCC_x_t2 = np.array([test2_yCC[i,j,1],test2_yCC[i,j,2]])

        #classification of second test image in chromatic space
        if gaussian_pdf(x_t2,mew, lhs, cov_inv)>thresholds[t]:
          test2_label = 1
        else:
          # test2_rgb[i,j] = [255,255,255]
          test2_label = 0
        
        #classification for yCC space
        if gaussian_pdf(yCC_x_t2,yCC_mew, yCC_lhs, yCC_cov_inv)>yCC_thresholds[t]:
          yCC_test2_label = 1
        else:
          # yCC_test2_rgb[i,j] = [255,255,255]
          yCC_test2_label = 0

        #compute FN FP TN TP for second test image in chromatic space
        if ground_truth_t2[i,j] == 1 and test2_label == 1:
          test2_TP_count = test2_TP_count + 1
        elif ground_truth_t2[i,j] == 0 and test2_label == 0:
          test2_TN_count = test2_TN_count + 1
        elif ground_truth_t2[i,j] == 0 and test2_label == 1:
          test2_FP_count = test2_FP_count + 1
        else:
          test2_FN_count = test2_FN_count + 1
        
        #compute FN FP TN TP for second test image in yCC space
        if ground_truth_t2[i,j] == 1 and yCC_test2_label == 1:
          yCC_test2_TP_count = yCC_test2_TP_count + 1
        elif ground_truth_t2[i,j] == 0 and yCC_test2_label == 0:
          yCC_test2_TN_count = yCC_test2_TN_count + 1
        elif ground_truth_t2[i,j] == 0 and yCC_test2_label == 1:
          yCC_test2_FP_count = yCC_test2_FP_count + 1
        else:
          yCC_test2_FN_count = yCC_test2_FN_count + 1
  # print(f"t = {t}, {FP_count} {TN_count} || {FPR_list}" )
  #balancing divide by zero error for FPR and FNR calculation for train and test images in chromatic space
  if(FP_count+TN_count)==0:
    FPR_list[t] = 0
  else:
    FPR_list[t] = FP_count/(FP_count+TN_count)

  if(FN_count+TP_count)==0:
    FNR_list[t] = 0
  else:
    FNR_list[t] = FN_count/(FN_count+TP_count)

  if(test1_FP_count+test1_TN_count)==0:
    test1_FPR_list[t] = 0
  else:
    test1_FPR_list[t] = test1_FP_count/(test1_FP_count+test1_TN_count)
  
  if(test1_FN_count+test1_TP_count)==0:
     test1_FNR_list[t] = 0
  else:
    test1_FNR_list[t] = test1_FN_count/(test1_FN_count+test1_TP_count)


  if(test2_FP_count+test2_TN_count)==0:
    test2_FPR_list[t] = 0
  else:
    test2_FPR_list[t] = test2_FP_count/(test2_FP_count+test2_TN_count)
  
  if(test2_FN_count+test2_TP_count)==0:
     test2_FNR_list[t] = 0
  else:
    test2_FNR_list[t] = test2_FN_count/(test2_FN_count+test2_TP_count)
  
  ############################ balancing divide by zero error for FPR and FNR calculation for train and test images in yCC space
  if(yCC_FP_count+yCC_TN_count)==0:
    yCC_FPR_list[t] = 0
  else:
    yCC_FPR_list[t] = yCC_FP_count/(yCC_FP_count+yCC_TN_count)

  if(yCC_FN_count+yCC_TP_count)==0:
    yCC_FNR_list[t] = 0
  else:
    yCC_FNR_list[t] = yCC_FN_count/(yCC_FN_count+yCC_TP_count)

  if(yCC_test1_FP_count+yCC_test1_TN_count)==0:
    yCC_test1_FPR_list[t] = 0
  else:
    yCC_test1_FPR_list[t] = yCC_test1_FP_count/(yCC_test1_FP_count+yCC_test1_TN_count)
  
  if(yCC_test1_FN_count+yCC_test1_TP_count)==0:
     yCC_test1_FNR_list[t] = 0
  else:
    yCC_test1_FNR_list[t] = yCC_test1_FN_count/(yCC_test1_FN_count+yCC_test1_TP_count)


  if(yCC_test2_FP_count+yCC_test2_TN_count)==0:
    yCC_test2_FPR_list[t] = 0
  else:
    yCC_test2_FPR_list[t] = yCC_test2_FP_count/(yCC_test2_FP_count+yCC_test2_TN_count)
  
  if(yCC_test2_FN_count+yCC_test2_TP_count)==0:
     yCC_test2_FNR_list[t] = 0
  else:
    yCC_test2_FNR_list[t] = yCC_test2_FN_count/(yCC_test2_FN_count+yCC_test2_TP_count)
  
  #reinitializing values to zero for evaluation with respect to the next thresholds in chromatic space
  FN_count = FP_count = TP_count = TN_count = 0
  test1_FP_count = test1_FN_count = test1_TP_count = test1_TN_count = 0
  test2_FP_count = test2_FN_count = test2_TP_count = test2_TN_count = 0

  #reinitializing values to zero for evaluation with respect to the next thresholds in chromatic space
  yCC_FN_count = yCC_FP_count = yCC_TP_count = yCC_TN_count = 0
  yCC_test1_FP_count = yCC_test1_FN_count = yCC_test1_TP_count = yCC_test1_TN_count = 0
  yCC_test2_FP_count = yCC_test2_FN_count = yCC_test2_TP_count = yCC_test2_TN_count = 0

#
min_t1 = abs(test1_FPR_list[0]-test1_FNR_list[0])
min_t1_index = 0

min_t2 = abs(test2_FPR_list[0]-test2_FNR_list[0])
min_t2_index = 0

yCC_min_t1 = abs(yCC_test1_FPR_list[0]-yCC_test1_FNR_list[0])
yCC_min_t1_index = 0

yCC_min_t2 = abs(yCC_test2_FPR_list[0]-yCC_test2_FNR_list[0])
yCC_min_t2_index = 0

for t in range(1,20,1):
  if abs(test1_FPR_list[t]-test1_FNR_list[t]) < min_t1 :
   min_t1 = abs(test1_FPR_list[t]-test1_FNR_list[t])
   min_t1_index = t
  
  if abs(test2_FPR_list[t]-test2_FNR_list[t]) < min_t2 :
   min_t2 = abs(test2_FPR_list[t]-test2_FNR_list[t])
   min_t2_index = t
  
  if abs(yCC_test1_FPR_list[t]-yCC_test1_FNR_list[t]) < yCC_min_t1 :
   yCC_min_t1 = abs(yCC_test1_FPR_list[t]-yCC_test1_FNR_list[t])
   yCC_min_t1_index = t
  
  if abs(yCC_test2_FPR_list[t]-yCC_test2_FNR_list[t]) < yCC_min_t2 :
   yCC_min_t2 = abs(yCC_test2_FPR_list[t]-yCC_test2_FNR_list[t])
   yCC_min_t2_index = t

error_rate_t1_chromatic = test1_FNR_list[min_t1_index]
error_rate_t2_chromatic = test2_FNR_list[min_t2_index]

error_rate_t1_yCC = yCC_test1_FNR_list[yCC_min_t1_index]
error_rate_t2_yCC = yCC_test2_FNR_list[yCC_min_t2_index]

print('='*30)
print(f'error_rate_t1_chromatic {error_rate_t1_chromatic}')
print(f'error_rate_t2_chromatic {error_rate_t2_chromatic}')
print(f'error_rate_t1_yCC {error_rate_t1_yCC}')
print(f'error_rate_t2_yCC {error_rate_t2_yCC}')

print('='*30)
print(f'Optimal threshold for test1 chromatic: {thresholds[min_t1_index]}')
print(f'Optimal threshold for test1 yCC: {yCC_thresholds[yCC_min_t1_index]}')
print(f'Optimal threshold for test2 chromatic: {thresholds[min_t2_index]}')
print(f'Optimal threshold for test2 yCC: {yCC_thresholds[yCC_min_t2_index]}')
print('='*30)


for i in range(width):
  for j in range (height):
    #######test1 image
        x_t1 = np.array([test1_chromatic[i,j,0],test1_chromatic[i,j,1]])
        yCC_x_t1 = np.array([test1_yCC[i,j,1],test1_yCC[i,j,2]])
       
        #assigning labels for chromatic space
        if gaussian_pdf(x_t1,mew, lhs, cov_inv)<=thresholds[min_t1_index]:
          test1_rgb[i,j] = [255,255,255]
         
        #classification for yCC space
        if gaussian_pdf(yCC_x_t1,yCC_mew, yCC_lhs, yCC_cov_inv)<=yCC_thresholds[yCC_min_t1_index]:
          yCC_test1_rgb[i,j] = [255,255,255]
    
    #######test2 image
        x_t2 = np.array([test2_chromatic[i,j,0],test2_chromatic[i,j,1]])
        yCC_x_t2 = np.array([test2_yCC[i,j,1],test2_yCC[i,j,2]])
       
        #assigning labels for chromatic space
        if gaussian_pdf(x_t2,mew, lhs, cov_inv)<=thresholds[min_t2_index]:
          test2_rgb[i,j] = [255,255,255]
         
        #classification for yCC space
        if gaussian_pdf(yCC_x_t2,yCC_mew, yCC_lhs, yCC_cov_inv)<=yCC_thresholds[yCC_min_t2_index]:
          yCC_test2_rgb[i,j] = [255,255,255]
          
Image.fromarray(test1_rgb).save('/home/mraha/PA_2/P2_Data/Data_Prog2/test1_chromatic.ppm')  
Image.fromarray(test2_rgb).save('/home/mraha/PA_2/P2_Data/Data_Prog2/test2_chromatic.ppm')
Image.fromarray(yCC_test1_rgb).save('/home/mraha/PA_2/P2_Data/Data_Prog2/test1_yCC.ppm')  
Image.fromarray(yCC_test2_rgb).save('/home/mraha/PA_2/P2_Data/Data_Prog2/test2_yCC.ppm')

# FPR&FNR vs Thresholds
plt.plot(thresholds,FPR_list, 'r',label = "chromatic_FP")
plt.plot(thresholds,FNR_list,'b',label = "chromatic_FN" )
plt.title('Train ROC Curve FPR & FNR Vs Thresholds')
plt.xlabel('Threshold')
plt.ylabel('Error Rate')
plt.legend(loc="lower right")
plt.savefig('Train_ROC_chromatic_FPR & FNR Vs T.png')
plt.show()

plt.plot(yCC_thresholds,yCC_FPR_list, 'g',label = "yCC_FP")
plt.plot(yCC_thresholds,yCC_FNR_list,'y',label = "yCC_FN" )
plt.title('Train ROC Curve FPR & FNR Vs Thresholds')
plt.xlabel('Threshold')
plt.ylabel('Error Rate')
plt.legend(loc="lower right")
plt.savefig('Train_ROC_yCC_FPR & FNR Vs T.png.png')
plt.show()


plt.plot(thresholds,test1_FPR_list, 'r',label = "chromatic_test1_FP")
plt.plot(thresholds,test1_FNR_list,'b',label = "chromatic_test1_FN" )
plt.title('Test1 ROC Curve FPR&FNR Vs Thresholds')
plt.xlabel('Threshold')
plt.ylabel('Error Rate')
plt.legend(loc="lower right")
plt.savefig('Test1_ROC_chromatic_FPR & FNR Vs T.png')
plt.show()

plt.plot(yCC_thresholds,yCC_test1_FPR_list, 'g',label = "yCC_test1_FP")
plt.plot(yCC_thresholds,yCC_test1_FNR_list,'y',label = "yCC_test1_FN" )
plt.title('Test1 ROC Curve FPR & FNR Vs T yCC')
plt.xlabel('Threshold')
plt.ylabel('Error Rate')
plt.legend(loc="lower right")
plt.savefig('Test1_ROC_FPR & FNR Vs T yCC.png')
plt.show()

plt.plot(thresholds, test2_FPR_list,'r',label = "chromatic_test2_FP")
plt.plot(thresholds,test2_FNR_list,'b',label = "chromatic_test2_FN")
plt.title('Test2 ROC Curve FPR & FNR Vs Thresholds')
plt.xlabel('Threshold')
plt.ylabel('Error Rate')
plt.legend(loc="lower right")
plt.savefig('Test2_ROC_curve_FPR & FNR Vs Thresholds_chromatic.png')
plt.show()

plt.plot(yCC_thresholds, yCC_test2_FPR_list,'g',label = "yCC_test2_FP")
plt.plot(yCC_thresholds,yCC_test2_FNR_list,'y',label = "yCC_test2_FN")
plt.xlabel('Threshold')
plt.ylabel('Error Rate')
plt.legend(loc="lower right")
plt.savefig('Test2_ROC_curve_FPR & FNR Vs Thresholds yCC.png')
plt.show()

#FNRvsFPR
plt.plot(FPR_list,FNR_list, 'r',label = "chromatic")
plt.plot(yCC_FPR_list,yCC_FNR_list, 'b',label = "yCC")
plt.title('Train ROC Curve FNR Vs FPR')
plt.xlabel('FPR')
plt.ylabel('FNR')
plt.legend(loc="lower right")
plt.savefig('Train_ROC_curve_FNR vs FPR.png')
plt.show()

plt.plot(test1_FPR_list,test1_FNR_list, 'r',label = "chromatic")
plt.plot(yCC_test1_FPR_list,yCC_test1_FNR_list, 'g',label = "yCC")
plt.title('Test1 ROC Curve FNR vs FPR')
plt.xlabel('FPR')
plt.ylabel('FNR')
plt.legend(loc="lower right")
plt.savefig('Test1_ROC_curve_FNR vs FPR.png')
plt.show()

plt.plot(test2_FPR_list,test2_FNR_list,'r',label = "chromatic")
plt.plot(yCC_test2_FPR_list,yCC_test2_FNR_list,'g',label = "yCC")
plt.title('Test2 ROC Curve FNR Vs FPR')
plt.xlabel('FPR')
plt.ylabel('FNR')
plt.legend(loc="lower right")
plt.savefig('Test2_ROC_curve_FNR vs FPR.png')
plt.show()

