# import PIL
# from PIL import Image
import time
import math
import numpy as np
import cv2
import warnings
import scipy
import random
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
# from scipy.stats import multivariate_normal
# from decimal import Decimal, getcontext

warnings.filterwarnings("ignore", category=RuntimeWarning)
# warnings.filterwarnings("ignore", category=TypeError)
#open cv reads in the order BGR

def compute_covariance(N, x, mu):
   for i in range(N):
      sigma = (x-mu) * np.transpose(x-mu)

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
# print("width ,height ", width, height)
# ground_truth_1 = np.empty(width,height)
ground_truth_1 = np.zeros((width,height))
chromatic = np.zeros((train_rgb.shape[0],train_rgb.shape[1],train_rgb.shape[2]))

#r g b channel values for red and green
red = np.array([3, 3, 255])
white = np.array([255, 255, 255])

red_channel = np.array([])
green_channel = np.array([])
vector = np.vectorize(np.double)

# only convert the pixels in training image that correspond to white or red pixels as per reference image mapping
# saving values in chromatic in RGB order
for i in range(width):
  for j in range(height):
    
    comparison_1 = train_ref_rgb[i][j] == red
    comparison_2 = train_ref_rgb[i][j] == white
    
    #denominator = (train_rgb[i,j][0] + train_rgb[i,j][1] + train_rgb[i,j][2])
    denominator = sum(train_rgb[i,j])

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
    
   #  #marking the non face regions in groud truth array as 0
   #  else:
   #     #0 = no face
   #     ground_truth_1[i,j] = 0
       
# print("lenght of red = length of green channel  = N", red_channel.size, green_channel.size)

N = red_channel.size

#Compute Mean
red_channel_mean = np.mean(red_channel, dtype = np.float64)
green_channel_mean = np.mean(green_channel, dtype = np.float64)
train_mean = np.array([red_channel_mean,green_channel_mean])
# print(" Sample mean = mew_r and mew_g ", red_channel_mean, green_channel_mean)
# print(f"mean = {train_mean} " )

# Compute Covariance 
covariance_matrix = np.array(np.cov(red_channel, green_channel))

# covariance_matrix = np.cov(train_mean)
# print("Covariance matrix ", covariance_matrix)

constant_factor = 1 / ((2*math.pi) * math.sqrt(np.linalg.det(covariance_matrix)))
step_size = constant_factor/20 
thresholds = []
thresholds = [0 for i in range(20)] 
number_of_thresholds = 20
# label = np.array([][])

# print("constant factor ", constant_factor)
# print("step size ", step_size)

for i in range (1,20,1):
    thresholds[i] = (thresholds[i-1] + step_size)

# print ("thresholds", thresholds)
temp_labels_1 = np.zeros((width,height))
# print("len(thresholds)) ", len(thresholds))
t = 0
FP = FN = TP = TN = train_bgr  = 0
FP_temp = FN_temp = TP_temp = TN_temp   = 0

FN_plot = np.zeros([])
FP_plot = np.zeros([])

FN_plot_temp = np.zeros([])
FP_plot_temp = np.zeros([])

equal_error_t = np.array([])

t_plot = np.array([])

#finding optimum threshold at equal error rate
for i in range(width):
    for j in range(height):
        # x = [r,g]
        x = np.array([chromatic[i,j,0],chromatic[i,j,1]])
        mew = np.array(train_mean)
        #print("mu ", mew)
      #   print("gaussian_pdf",gaussian_pdf(x,mew,covariance_matrix))
      #   print("t ",t)
      #   if gaussian_pdf(x,mew,covariance_matrix) > t: 
      #       temp_labels_1[i,j] = 1
          
        for k in range(20):
          t = random.sample(thresholds,k)
          t_plot = np.append(t_plot,t)

          #assigning 1 for face prediction labels 
          if(ground_truth_1[i,j] == 0 ) and (temp_labels_1[i,j] == 1):
            FP_temp = FP_temp + 1
            # FP_plot = np.append(FP_plot, FP)
       
          elif(ground_truth_1[i,j] == 1 ) and (temp_labels_1[i,j] == 0):
            FN_temp = FN_temp + 1
            # FN_plot = np.append(FN_plot, FN)
         
          if FP_temp == FN_temp:
            equal_error_t = np.append(equal_error_t, t)

optimum_t = np.argmax(equal_error_t) 
equal_error_t =    np.empty(equal_error_t)     
labels_1 = np.zeros((width,height))

#classification 
for i in range(width):
    for j in range(height):
       
      x = np.array([chromatic[i,j,0],chromatic[i,j,1]])
      mew = np.array(train_mean)
       
      #assigning label value as 1 face  
      if gaussian_pdf(x,mew,covariance_matrix) > optimum_t:
       labels_1[i,j] = 1
      
      #for each pixel checking assigned label with ground truth
      if (ground_truth_1[i,j] == 0 ) and (labels_1[i,j] == 1):
       FP = FP+1
       FP_plot = np.append(FP_plot,FP)

      elif(ground_truth_1[i,j] == 1 ) and (labels_1[i,j] == 0):
       FN = FN + 1
       FN_plot = np.append(FN_plot,FN)

plt.scatter(FP_plot, FN_plot, c= "blue")
plt.show()
 
#precision
# TPR = TP/TP+FN
# #specificity
# TNR = TN/TN+FP
# FPR = FP/FP+TN

plt.scatter(FP_plot, FN_plot,c = "blue")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

start = time.time()

print(23*2.3)

end = time.time()
print("time elapsed ",end - start)




           
           
               














# red_channel_mean = sum(red_channel)/red_channel.shape(0)