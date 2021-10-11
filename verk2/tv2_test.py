from tv2 import terminal
import numpy as np

def test_terminal():
    arr = np.array(
        [
            [1,0,0,0,0,1],
            [1,2,2,0,0,3],
            [2,1,0,0,0,2],
            [2,1,2,0,0,3],
            [1,1,0,0,0,2]
        ]
    )
    assert terminal(arr, p=1, i=4, j=1) == True
    assert terminal(arr, p=1, i=2, j=1) == True
    assert terminal(arr, p=2, i=3, j=2) == False

    arr = np.array(
        [
            [1,0,0,0,0,1],
            [1,2,0,0,0,2],
            [2,1,0,0,0,2],
            [2,1,2,1,2,5],
            [1,2,1,2,0,4]
        ]
    )
    assert terminal(arr, p=2, i=3, j=4) == False
    assert terminal(arr, p=1, i=2, j=1) == False
    
    arr = np.array(
        [
            [0,0,0,0,0,0],
            [1,0,0,0,0,1],
            [1,2,1,2,1,5],
            [2,1,2,1,2,5],
            [2,1,0,0,0,2]
        ]
    )
    assert terminal(arr, p=1, i=2, j=4) == False
    assert terminal(arr, p=2, i=3, j=4) == False