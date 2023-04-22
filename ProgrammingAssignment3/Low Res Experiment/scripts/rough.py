import csv
import numpy as np

# Open the CSV file in read mode
with open('dummy .csv', 'r') as csvfile:
    # Create a CSV reader object
    reader = csv.reader(csvfile)

    # Initialize an empty list to hold the row data
    rows = []

    # Iterate through each row in the CSV file
    for row in reader:
        # Split the row contents by a delimiter, e.g. comma
        values = row[0].split(',') # assuming the row has only one column

        # Append the values to the rows list as a new row
        rows.append(values)

    # Convert the rows list to a NumPy array
    arr = np.array(rows)

    # You can now access the values in the NumPy array as needed
    # For example, you can print the array
    print(arr)