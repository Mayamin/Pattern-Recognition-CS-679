

#works

#!/ usr / bin /python3

import os
import pandas as pd
from matplotlib import pyplot as plt

# Set the figure size
plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True

# Make a list of columns
columns = ['x ', 'y']

# Read a CSV file
df_part1 = pd.read_csv("/home/mraha/Desktop/myproject/src/dataset_A_part1.csv")

df_part1.plot(kind = 'scatter', x = 'x', y = 'y', figsize=(6,6),c = 'green')
plt.savefig('figure_1.pdf', dpi=300)
df_part1 = pd.read_csv("/home/mraha/Desktop/myproject/src/dataset_A_part2.csv")
df_part1.plot(kind = 'scatter', x = 'x', y = 'y', figsize=(6,6),c = 'red')
plt.savefig('figure_2.pdf', dpi=300)
# Plot the lines

# df_part2.plot(kind = 'scatter', x = 'x', y = 'y', figsize=(6,6),c = 'red')
# plt.scatter(df_part1,df_part2 c=cat_col)
plt.show()




# not working
# import pandas as pd
# import matplotlib.pyplot as plt
# import csv

# # df_1= pd.DataFrame(data = data_value, columns = ['x', 'y']); 
# # df_1 = pd.read_csv('/home/mraha/Desktop/myproject/src/dataset_A_part1.csv')
# # df_1.plot.scatter(x = 'x', y = 'y', title = "Scatter plot pandas variable for X and Y axis"); 

# X1, Y1 = [], []
# X2, Y2 = [], []

# with open('/home/mraha/Desktop/myproject/src/dataset_A_part1.csv','r') as csvfile:
#     lines = csv.reader(csvfile, delimiter=',')
#     for row in lines:
#         X1.append(row[0])
#         Y1.append(float(row[1]))

# with open('/home/mraha/Desktop/myproject/src/dataset_A_part2.csv','r') as csvfile:
#     lines = csv.reader(csvfile, delimiter=',')
#     for row in lines:
#         X2.append(row[0])
#         Y2.append(float(row[1]))        
  
# plt.scatter(X1, Y1, color = 'g',s = 100)
# plt.scatter(X2, Y2, color = 'r',s = 100)
# plt.xticks(rotation = 25)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Data Distribution', fontsize = 20)
# plt.savefig("data_visualization.pdf", format="pdf", bbox_inches="tight")
  
# plt.show()

# import seaborn as sb
# import pandas as pd
# import matplotlib.pyplot as plt

# dataFrame = pd.read_csv("/home/mraha/Desktop/myproject/src/dataset_A_part1.csv")

# # plotting scatterplot with Age and Weight
# # weight in kgs
# sb.scatterplot(dataFrame['x'],dataFrame['y'])

# # plt.ylabel("Weight (kgs)")
# plt.show()




# works

# import sys
# import matplotlib
# # matplotlib.use('Agg')

# import pandas as pd
# import matplotlib.pyplot as plt

# df = pd.read_csv('/home/mraha/Desktop/myproject/src/dataset_A_part1.csv')

# df.plot(kind = 'scatter', x = 'x', y = 'y')
# # print(df._1)
# # plt.scatter()

# plt.show()

# #Two  lines to make our compiler able to draw:
# # plt.savefig(sys.stdout.buffer)
# # sys.stdout.flush()
