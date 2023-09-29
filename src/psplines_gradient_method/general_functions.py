import numpy as np
import matplotlib.pyplot as plt


def create_precision_matrix(P):
    Omega = np.zeros((P, P))
    # fill the main diagonal with 2s
    np.fill_diagonal(Omega, 2)
    # fill the subdiagonal and superdiagonal with -1s
    np.fill_diagonal(Omega[1:], -1)
    np.fill_diagonal(Omega[:, 1:], -1)
    # set the last element to 1
    Omega[-1, -1] = 1
    return Omega

def create_first_diff_matrix(P):
    D = np.zeros((P-1, P))
    # fill the main diagonal with -1s
    np.fill_diagonal(D, -1)
    # fill the superdiagonal with 1s
    np.fill_diagonal(D[:, 1:], 1)
    return D


def create_second_diff_matrix(P):
    D = np.zeros((P-2, P))
    # fill the main diagonal with 1s
    np.fill_diagonal(D, 1)
    # fill the subdiagonal and superdiagonal with -2s
    np.fill_diagonal(D[:, 2:], 1)
    np.fill_diagonal(D[:, 1:], -2)
    # set the last element to 1
    D[-1, -1] = 1
    return D


def create_masking_matrix(N, M):
    mat = np.zeros((N, N * M), dtype=int)  # Start with a kxkm matrix of zeros
    rows = np.repeat(np.arange(N), M)
    cols = np.arange(N * M).reshape(N, M).ravel()
    mat[rows, cols] = 1
    return mat


def reshape_beta_for_sparse_multiplication(beta, format, width):
    L, P = beta.shape
    # Create an array of column indices for beta using broadcasting and modulo operation
    col_indices = (np.arange(width) + format[:, None]) % P
    # Use advanced indexing to fill beta_star
    beta_star = beta[:, col_indices]
    return np.transpose(beta_star, (0, 2, 1))


def reshape_ylambda_for_sparse_multiplication(ylambda, formats, width):
    L, P = ylambda.shape
    # Initialize a list to hold the 3D arrays created for each row of formats
    ylambda_star_subs = []
    # Loop over each row of formats and apply the existing logic
    for i, format in enumerate(formats):
        # Create an array of column indices for beta using broadcasting and modulo operation
        col_indices = (np.arange(width) + format[:, None]) % P
        # Use advanced indexing to fill beta_star
        ylambda_star = ylambda[i, col_indices]
        # Append the transposed beta_star to the list
        ylambda_star_subs.append(ylambda_star.T)
    # Stack the list of 3D arrays along a new axis to create the final 3D array
    return np.stack(ylambda_star_subs)


def reshape_bpsi_for_sparse_multiplication(b_psi, K):
    KP, T = b_psi.shape
    P = KP//K
    B_split = [b_psi[i * P: (i + 1) * P, :] for i in range(K)]
    # Create a list to store the aligned matrices corresponding to each element in B_split
    aligned_mats = []
    # Create a list to store the row number of the first non-zero entry for each b_split
    first_non_zero_rows = []
    for b_split in B_split:
        # Create a new matrix to store the aligned matrix for the current b_split
        aligned_mat = np.zeros((P, T), dtype=b_psi.dtype)
        # Create a vector to store the row number of the first non-zero entry for the current b_split
        first_non_zero_vector = np.full(T, -1, dtype=int)  # Initialize with -1 to represent columns with all zeros
        for t in range(T):
            # Find the indices of non-zero elements in the column
            non_zero_indices = np.nonzero(b_split[:, t])[0]
            if non_zero_indices.size > 0:
                # Find the index of the first non-zero element in the column
                first_non_zero_index = non_zero_indices[0]
                # Store the row number of the first non-zero entry in the vector
                first_non_zero_vector[t] = first_non_zero_index
                # Shift the elements in the column upwards
                aligned_mat[:P - first_non_zero_index, t] = b_split[first_non_zero_index:, t]
        # Append the vector to the list
        first_non_zero_rows.append(first_non_zero_vector)
        # Find the index of the last row that contains a non-zero element
        last_non_zero_row = 0
        for p in range(P - 1, -1, -1):
            if np.any(aligned_mat[p, :]):
                last_non_zero_row = p
                break
        # Remove the bottom rows that contain all zeros
        aligned_mat = aligned_mat[:last_non_zero_row + 1, :]
        # Append the aligned matrix for the current b_split to the list
        aligned_mats.append(aligned_mat)
    # Concatenate the list of aligned matrices horizontally
    concatenated_aligned_mat = np.concatenate(aligned_mats, axis=1)
    # Concatenate the list of first non-zero row vectors horizontally
    concatenated_first_non_zero_rows = np.concatenate(first_non_zero_rows)
    return concatenated_aligned_mat, concatenated_first_non_zero_rows


def reshape_bpsiTranspose_for_sparse_multiplication(b_psi, K):
    KP, T = b_psi.shape
    P = KP // K
    B_split = [b_psi[i * P: (i + 1) * P, :] for i in range(K)]
    # Create a list to store the aligned matrices corresponding to each element in B_split
    aligned_mats = []
    # Create a list to store the column number of the first non-zero entry for each b_split
    first_non_zero_cols = []
    max_cols = 0  # To store the maximum number of columns among all aligned_mat

    for b_split in B_split:
        # Create a new matrix to store the aligned matrix for the current b_split
        aligned_mat = np.zeros((P, T), dtype=b_psi.dtype)
        # Create a vector to store the column number of the first non-zero entry for the current b_split
        first_non_zero_vector = np.full(P, -1, dtype=int)  # Initialize with -1 to represent rows with all zeros
        for p in range(P):
            # Find the indices of non-zero elements in the row
            non_zero_indices = np.nonzero(b_split[p, :])[0]
            if non_zero_indices.size > 0:
                # Find the index of the first non-zero element in the row
                first_non_zero_index = non_zero_indices[0]
                # Store the column number of the first non-zero entry in the vector
                first_non_zero_vector[p] = first_non_zero_index
                # Shift the elements in the row to the left
                aligned_mat[p, :T - first_non_zero_index] = b_split[p, first_non_zero_index:]
        # Append the vector to the list
        first_non_zero_cols.append(first_non_zero_vector)
        # Find the index of the last column that contains a non-zero element
        last_non_zero_col = T - 1
        for t in range(T - 1, -1, -1):
            if np.any(aligned_mat[:, t]):
                last_non_zero_col = t
                break
        # Remove the rightmost columns that contain all zeros
        aligned_mat = aligned_mat[:, :last_non_zero_col + 1]
        # Append the aligned matrix for the current b_split to the list
        aligned_mats.append(aligned_mat)

        # Update max_cols
        if aligned_mat.shape[1] > max_cols:
            max_cols = aligned_mat.shape[1]

    # Pad the matrices with zeros to make them have the same number of columns
    for i in range(len(aligned_mats)):
        num_cols_to_pad = max_cols - aligned_mats[i].shape[1]
        if num_cols_to_pad > 0:
            padding = np.zeros((P, num_cols_to_pad), dtype=b_psi.dtype)
            aligned_mats[i] = np.hstack((aligned_mats[i], padding))

    # Now concatenate the padded matrices vertically
    concatenated_aligned_mat = np.concatenate(aligned_mats, axis=0)
    # Concatenate the list of first non-zero column vectors vertically
    concatenated_first_non_zero_cols = np.concatenate(first_non_zero_cols)
    return concatenated_aligned_mat.T, concatenated_first_non_zero_cols


def compute_lambda(B_psi, d, G_star, beta):
    K = len(d)
    J = np.ones((K, B_psi.shape[1]))
    diagdJ_plus_GBetaB = d[:, np.newaxis] * J + G_star @ np.kron(np.eye(K), beta) @ B_psi
    lambda_ = np.exp(diagdJ_plus_GBetaB)
    return lambda_


def plot_spikes(spikes, x_offset=0):
    # Group entries by unique values of s[0]
    unique_s_0 = np.unique(spikes[0])
    grouped_s = []
    for i in unique_s_0:
        indices = np.where(spikes[0] == i)[0]
        values = (spikes[1][indices] - x_offset)/1000
        grouped_s.append((i, values))
    for group in grouped_s:
        plt.scatter(group[1], np.zeros_like(group[1]) + group[0], s=1, c='black')
    plt.show()


def plot_binned(binned):
    # plot binned spikes
    _, ax = plt.subplots()
    ax.imshow(binned)
    ax.invert_yaxis()
    plt.show()


def plot_bsplines(B, time):
    # plot bsplines
    start = 190
    num_basis = 10
    for i in range(num_basis):
        plt.plot(time[start:(start+num_basis)], B[i+start, start:(start+num_basis)])
    plt.show()
