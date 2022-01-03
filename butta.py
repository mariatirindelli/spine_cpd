constraints = [
    [  17.925,   252.5 ,    -69.2 ,      1.    ],
     [  18.0714,  244.5 ,    -76.2,       2.    ],
     [  21.0714 , 244.5 ,    -98.2 ,      2.    ],
     [  21.      ,231.5 ,   -104.223,     3.    ],
     [  21.      ,229.5 ,   -130.177 ,    3.    ],
     [  21.      ,224.5  ,  -141.222  ,   4.    ],
     [  21.      ,221.5 ,   -166.178   ,  4.    ],
     [  24.      ,223.5,    -177.221  ,   5.    ]]

import numpy as np
import pandas as pd

arr = np.array(constraints)
constr = []
for i in range(arr.shape[0]-1):
    slice = arr[i:i+1]
    if arr[i, -1] == arr[i+1, -1]:
        continue
    print(arr[i, -1], "  ", arr[i+1, -1], "  ", np.mean(slice, axis=0))

    pos = np.mean(slice, axis=0)
    constr.append({"node1": arr[i, -1],
                   "node2": arr[i+1, -1],
                   "x": pos[0],
                   "y":pos[1],
                   "z":pos[2]})

frame = pd.DataFrame(constr)
frame.to_csv("C:/Users/maria/OneDrive/Desktop/frame.csv")


