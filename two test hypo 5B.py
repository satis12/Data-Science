# Python program to implement Independent T-Test on the two independent samples  
  
# Importing the required libraries  
from scipy.stats import ttest_ind  
import numpy as np  
  
# Creating the data groups  
data_group1 = np.array([12, 18, 12, 13, 15, 1, 7,  
                        20, 21, 25, 19, 31, 21, 17,  
                        17, 15, 19, 15, 12, 15])  
data_group2 = np.array([23, 22, 24, 25, 21, 26, 21,  
                        21, 25, 30, 24, 21, 23, 19,  
                        14, 18, 14, 12, 19, 15])  
  
# Calculating the mean of the two data groups  
mean1 = np.mean(data_group1)  
mean2 = np.mean(data_group2)  
  
# Print mean values  
print("Data group 1 mean value:", mean1)  
print("Data group 2 mean value:", mean2)  
  
# Calculating standard deviation  
std1 = np.std(data_group1)  
std2 = np.std(data_group2)  
  
# Printing standard deviation values  
print("Data group 1 std value:", std1)  
print("Data group 2 std value:", std2)  
  
# Implementing the t-test  
t_test,p_val = ttest_ind(data_group1, data_group2)  
print("The P-value is: ", p_val)  
  
# taking the threshold value as 0.05 or 5%  
if p_val < 0.05:      
    print("We can reject the null hypothesis")  
else:  
    print("We can accept the null hypothesis")













