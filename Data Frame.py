import pandas as pd

# Set option for future behavior
pd.set_option('future.no_silent_downcasting', True)

# Assign data
data = {'Name': ['Jai', 'Princi', 'Gaurav',
                'Anuj', 'Ravi', 'Natasha', 'Riya'],
        'Age': [17, 17, 18, 17, 18, 17, 17],
        'Gender': ['M', 'F', 'M', 'M', 'M', 'F', 'F'],
        'Marks': [90, 76, 'NaN', 74, 65, 'NaN', 71]}

# Convert into DataFrame
df = pd.DataFrame(data)

# Display data
print(df)

# Compute average
c = avg = 0
for ele in df['Marks']:
    if str(ele).isnumeric():
        c += 1
        avg += int(ele)
avg /= c

# Replace missing values and infer data types
df['Marks'] = df['Marks'].replace(to_replace="NaN", value=avg).astype(float).infer_objects(copy=False)

# Display data
print(df)

# Categorize gender
df['Gender'] = df['Gender'].map({'M': 0, 'F': 1}).astype(float)

# Display data
print(df)

# Filter top scoring students
df = df[df['Marks'] >= 75].copy()

# Remove age column from filtered DataFrame
df.drop('Age', axis=1, inplace=True)

# Display data
print(df)

# Import module
import pandas as pd

# creating DataFrame for Student Details
details = pd.DataFrame({
    'ID': [101, 102, 103, 104, 105, 106,
        107, 108, 109, 110],
    'NAME': ['Jagroop', 'Praveen', 'Harjot',
            'Pooja', 'Rahul', 'Nikita',
            'Saurabh', 'Ayush', 'Dolly', "Mohit"],
    'BRANCH': ['CSE', 'CSE', 'CSE', 'CSE', 'CSE',
            'CSE', 'CSE', 'CSE', 'CSE', 'CSE']})

# printing details
print(details)

# Creating Dataframe for Fees_Status
fees_status = pd.DataFrame(
    {'ID': [101, 102, 103, 104, 105,
            106, 107, 108, 109, 110],
    'PENDING': ['5000', '250', 'NIL',
                '9000', '15000', 'NIL',
                '4500', '1800', '250', 'NIL']})

# Printing fees_status
print(fees_status)
