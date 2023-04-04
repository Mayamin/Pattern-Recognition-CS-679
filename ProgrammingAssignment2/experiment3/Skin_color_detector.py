# import PIL
# from PIL import Image
import math
import numpy as np
import cv2
import warnings
import scipy
import random
from scipy.stats import multivariate_normal
from decimal import Decimal, getcontext
# import tensorflow as tf

# warnings.filterwarnings("ignore", category=RuntimeWarning)
#open cv reads in the order BGR

# Divide two numbers with a tolerance for very small denominators.
# def safe_divide(num, denom, epsilon=1e-10):
    
#     if abs(denom) < epsilon:
#         return 0.0  
#     else:
#         return num / denom

# Calculate the Gaussian probability density function for a given x, mean (mu) and variance (var)
def gaussian_pdf(x, mu, cov):
        #transpose = (x-mew)^T
        # x = x.astype(float)
        # mu = mu.astype(float)
        # traspose = np.transpose(x-mu)
        pdf = (1/ math.sqrt(((2*math.pi)**2) * np.linalg.det(cov))) * math.exp( (-0.5) * np.transpose(x-mu) * np.linalg.inv(cov) * (x-mu)) 

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
# ground_truth_1 = np.empty(width,height)
ground_truth_1 = np.zeros((width,height))
chromatic = np.zeros((train_rgb.shape[0],train_rgb.shape[1],train_rgb.shape[2]))

#r g b channel values for red and green
red = np.array([3, 3, 255])
white = np.array([255, 255, 255])

red_channel = np.array([])
green_channel = np.array([])

# only convert the pixels in training image that correspond to white or red pixels as per reference image mapping
# saving values in chromatic in RGB order
for i in range(width):
  for j in range(height):
    
    comparison_1 = train_ref_rgb[i][j] == red
    comparison_2 = train_ref_rgb[i][j] == white
    
    #denominator = (train_rgb[i,j][0] + train_rgb[i,j][1] + train_rgb[i,j][2])
    denominator = sum(train_rgb[i,j])

    chromatic[i,j,0] = train_rgb[i,j][0] / denominator
    chromatic[i,j,1] = train_rgb[i,j][1] / denominator

    #keeping the face pixel values in red and green channel and marking these locations as 1 in groud truth array
    if (comparison_1.all() ) or ( comparison_2.all()):

      # #red channel
      # numerator = train_rgb[i,j][0]
      # chromatic[i,j,0] = safe_divide(numerator,denominator)

      # #green channel
      numerator = train_rgb[i,j][1] 
      # chromatic[i,j,1] = safe_divide(numerator,denominator)

      #red channel
      # numerator = train_rgb[i,j][0]
      # chromatic[i,j,0] = numerator / denominator
      red_channel = np.append(red_channel, chromatic[i,j,0])
      
      #green channel
      numerator = train_rgb[i,j][1] 
      # # print(numerator)
      # chromatic[i,j,1] = numerator / denominator
      # print(chromatic[i,j,1])
      green_channel = np.append(green_channel, chromatic[i,j,1])

      ground_truth_1[i,j] = 1
      
      #debuggin print statements
      if green_channel[-1] > 1 or red_channel[-1] > 1:
        print("="*30)
        print(f'{train_rgb[i,j][0]} + {train_rgb[i,j][1]} + {train_rgb[i,j][2]} == {train_bgr[i,j][0]} + {train_bgr[i,j][1]} + {train_bgr[i,j][2]}')
        print(numerator)
        print("denominator ", denominator)
        print("Red channel values in face region ",red_channel)
        print("Green channel values in face region ", green_channel)
        print("="*30)
    
    #marking the non face regions in groud truth array as 0
    else:
       ground_truth_1[i,j] = 0
       

# red_channel = normalize(red_channel,0,1)
# green_channel = normalize(green_channel,0,1)

print(red_channel, green_channel)

red_channel_mean = np.mean(red_channel, dtype=np.float64)
green_channel_mean = np.mean(green_channel, dtype=np.float64)

getcontext().prec = 20

# red_channel_dec = [Decimal(str(num)) for num in red_channel]
# red_channel_mean = sum(red_channel_dec)/Decimal(len(red_channel))

# green_channel_dec = [Decimal(str(num)) for num in green_channel]
# green_channel_mean = sum(green_channel_dec)/Decimal(len(green_channel))


# green_channel_mean = np.mean(green_channel, dtype=np.float64)
# print(chromatic.shape)
print(red_channel_mean, green_channel_mean)

# print("Mean of Red channel  is ", red_channel_mean,"\n Mean of Green channel is ",  green_channel_mean)
# print(red_channel_mean.shape, green_channel_mean.shape)

train_mean = np.array([red_channel_mean,green_channel_mean])
# print(" Sample mean = mew_r and mew_g ", red_channel_mean, green_channel_mean)
print(f"mean = {train_mean} " )

# # Compute the sample covariance matrix of the channels
covariance_matrix = np.array(np.cov(red_channel, green_channel))

# covariance_matrix = np.cov(train_mean)
print(" Sample covariance matrix ", covariance_matrix)

# # # multivariate normal distribution
# distribution = multivariate_normal(mean = train_mean, cov = covariance_matrix) 

# # log-likelihood 
# likelihoods = distribution.logpdf([red_channel, green_channel])    
r_g_channel_combined = np.array([])

for i in range (red_channel.shape[0]):
  r_g_channel_combined = np.append(red_channel[i], green_channel[i])

# #calculating likelihood .i.e. g(x)
# likelihoods = distribution.logpdf(r_g_channel_combined) 
# print("Log-Likelihoods:\n", likelihoods)

# intermval_max = 
constant_factor = 1 / ((2*math.pi) * math.sqrt(np.linalg.det(covariance_matrix)))
step_size = constant_factor/20 
thresholds = []
thresholds = [0 for i in range(20)] 
number_of_thresholds = 20
# label = np.array([][])


print("constant factor ", constant_factor)
print("step size ", step_size)
print ("thresholds", thresholds)

for i in range (1,20,1):
    thresholds[i] = (thresholds[i-1] + step_size)

# print(thresholds[1]-thresholds[2])
for i in range (number_of_thresholds):
  print(thresholds[i])

labels_1 = np.zeros((width,height))

print("len(thresholds)) ", len(thresholds))
t = 0

for i in range(width):
    for j in range(height):
        
        # x = [r,g]
        x = np.array([chromatic[i,j][0],chromatic[i,j][1]])
        #mew = [rbar, gbar]
        mew = np.array(train_mean)
        # var = [covariance_matrix[0,0], covariance_matrix[1,1]]
        t = random.sample(thresholds,1)
       
       #assigning 0 or 1 prediction labels 
        if gaussian_pdf(x,mew, covariance_matrix) > t:
           # label = 1 means face, label = 0 means no face
           labels_1[i,j] = 1
        else:
           labels_1[i,j] = 0

FP = FN = TP = train_bgr  = 0

for i in range(width):
    for j in range(height):
       
       if(ground_truth_1[i,j] == 0 ) and (labels_1[i,j] == 0):
          TN = TN + 1
       elif(ground_truth_1[i,j] == 1 ) and (labels_1[i,j] == 1):
           TP = TP + 1
       elif(ground_truth_1[i,j] == 0 ) and (labels_1[i,j] == 1):
          FP = FP + 1
       elif(ground_truth_1[i,j] == 1 ) and (labels_1[i,j] == 0):
          FN = FN + 1  

print(ground_truth_1)   
print("labels ",labels_1)
           
           
               














# red_channel_mean = sum(red_channel)/red_channel.shape(0)