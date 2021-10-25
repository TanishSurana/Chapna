import numpy as np

wk = np.array([1, 0, 3, -4])

# checking if elements are positive

posi = wk[wk>0].sum()

print(posi.sum())

