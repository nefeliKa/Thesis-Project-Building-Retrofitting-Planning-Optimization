# import numpy as np
# from scipy.sparse import csr_matrix

# # Load the saved sparse matrices
# list_probs = np.load('probabilities_np.npy', allow_pickle=True)

# # Iterate over the loaded array and access each sparse matrix
# for action_matrices in list_probs:
#     for sparse_matrix in action_matrices:
#         # Convert each element of the NumPy array to a CSR sparse matrix
#         sparse_matrix = csr_matrix(sparse_matrix.item())  # .item() retrieves the single element from the NumPy array
#         # You can perform operations like getting the shape or converting to dense array
#         print("Shape of sparse matrix:", sparse_matrix.shape)
#         dense_matrix = sparse_matrix.toarray()  # Convert to dense array
#         sm = np.sum(dense_matrix)
#         print("Dense matrix:")
#         print(dense_matrix)


from scipy.sparse import load_npz
import numpy as np

# Load the sparse matrix
sparse_matrix_loaded = load_npz('sparse_matrix_action_7.npz')

states = np.load('state_space.npy')


# Save NumPy array to CSV
np.savetxt('states.csv', states, delimiter=',', fmt='%d')
# Convert the sparse matrix to a dense matrix
# dense_matrix = sparse_matrix_loaded.toarray()
# list = []
# for i in range(len(states)): 
#     sm = np.sum(dense_matrix[i])
#     if sm > 1.0000000000000004 or sm < 0.994:
#         list.append(i)


# print(list)
# print(list)
# p= np.load('optimal_policy.npy')
# v =np.load('q_values.npy')

# # action_ar = np.load('action_array.npy')
# n =action_ar[7]
print('bla')
# Now you can use sparse_matrix_loaded as a regular CSR format sparse matrix
