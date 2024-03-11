import pandas as pd
import matplotlib.pyplot as plt

# Sample Data (replace this with your actual data)
data = {'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May'],
        'Sales': [50, 45, 60, 55, 70],
        'Expenses': [30, 35, 40, 45, 50]}

df = pd.DataFrame(data)

# Visualization: Line Plot
plt.figure(figsize=(10, 6))
plt.plot(df['Month'], df['Sales'], marker='o', label='Sales')
plt.plot(df['Month'], df['Expenses'], marker='o', label='Expenses')

plt.title('Monthly Sales and Expenses')
plt.xlabel('Month')
plt.ylabel('Amount ($)')
plt.legend()
plt.grid(True)
plt.show()

# Storytelling
print("In the first five months of the year:")
print(f"- Sales increased steadily, reaching a peak in May.")
print(f"- Expenses also showed an upward trend, with a slight spike in April.")
print(f"- Despite the increase in expenses, the company managed to maintain a positive sales trend.")

# Additional analysis or visualizations can be added based on the dataset and storytelling goals.
