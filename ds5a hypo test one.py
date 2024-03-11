# import the necessary libraries
from scipy import stats
import numpy as np

# Sample
sample_A = np.array([1,2,4,4,5,5,6,7,8,8])
sample_B = np.array([1,2,2,3,3,4,5,6,7,7])

# Perform independent sample t-test
t_statistic, p_value = stats.ttest_ind(sample_A, sample_B)

# Set the significance level (alpha)
alpha = 0.05

# Compute the degrees of freedom (df) (n_A-1)+(n_b-1)
df = len(sample_A)+len(sample_B)-2

# Calculate the critical t-value
# ppf is used to find the critical t-value for a two-tailed test
critical_t = stats.t.ppf(1 - alpha/2, df)


# Print the results
print("T-value:", t_statistic)
print("P-Value:", p_value)
print("Critical t-value:", critical_t)

# Decision
print('With T-value')
if np.abs(t_statistic) >critical_t:
	print('There is significant difference between two groups')
else:
	print('No significant difference found between two groups')

print('With P-value')
if p_value >alpha:
	print('No evidence to reject the null hypothesis that a significant difference between the two groups')
else:
	print('Evidence found to reject the null hypothesis that a significant difference between the two groups')







