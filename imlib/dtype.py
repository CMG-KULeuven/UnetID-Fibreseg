import numpy as np

def normalize8(I):
    mn = I.min()
    mx = I.max()
    mx -= mn
    I = ((I - mn)/mx) * 255
    return I.astype(np.uint8)
