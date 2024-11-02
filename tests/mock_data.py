import numpy as np

# Create some example data
array1 = np.array([[1, 2], [3, 4]])
array2 = np.array([5, 6, 7])

# Save arrays to an NPZ file
np.savez('../data.npz', array1=array1, array2=array2)

# Load arrays from the NPZ file
data = np.load('../data.npz')
print(data['array1'])  # Access the first array
print(data['array2'])  # Access the second array

if __name__ == "__main__":
    # This block will run only if the script is executed directly
    print("Data has been saved to 'data.npz' and loaded successfully.")
