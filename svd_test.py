import numpy as np
LA = np.linalg

a = np.array([[1, 3, 4], [5, 6, 9], [1, 2, 3], [7, 6, 8]])
print(a)
# [[1 3 4]
#  [5 6 9]
#  [1 2 3]
#  [7 6 8]]
U, s, Vh = LA.svd(a, full_matrices=False)
assert np.allclose(a, np.dot(U, np.dot(np.diag(s), Vh)))

s[2:] = 0
new_a = np.dot(U, np.dot(np.diag(s), Vh))
print(new_a)
# [[ 1.02206755  2.77276308  4.14651336]
#  [ 4.9803474   6.20236935  8.86952026]
#  [ 0.99786077  2.02202837  2.98579698]
#  [ 7.01104783  5.88623677  8.07335002]]