import numpy as np
from scipy import sparse

# two player game (1) versus (2)
getotherplayer = lambda p : 3-p # returns the other player

# the initial empty board, in matrix board we store legal indices in board[:,-1]
def iState(n = 5, m = 5):
    return np.zeros((n,m+1), dtype=np.uint16)

# perform move for player on board
def Action(board, move, player):
    board[move,board[move,-1]] = player # place the disc on board
    board[move,-1] += 1 # next legal drop

# determine if terminal board state, assuming last move was made by player p
# We only need to check if the last piece added wins the game, no need to scan the whole board
def terminal(board, p, i, j, n = 5, m = 5):
    # Horizontal
    t = -min(2, i)
    count = 0
    while (t <= min(2, n-1-i) and count < 3):
        if board[i+t,j] == p:
            count += 1
        else:
            count = 0
        t += 1
    if count == 3:
        return True
    
    # Vertical
    # only checks below last placed marker
    t = -min(2, j)
    count = 0
    while (t <= 0):
        if board[i,j+t] == p:
            count += 1
        else:
            count = 0
        t += 1
    if count == 3:
        return True
    
    # Main diagonal
    t = -min(2, min(i, j))
    count = 0
    while (t <= min(2, min(n-1-i, m-1-j)) and count < 3):
        if board[i+t,j+t] == p:
            count += 1
        else:
            count = 0
        t += 1
    if count == 3:
        return True

    # Antidiagonal
    t = -min(2, min(i, m-1-j))
    count = 0
    while (t <= min(2, min(n-1-i, j)) and count < 3):
        if board[i+t,j-t] == p:
            count += 1
        else:
            count = 0
        t += 1
    if count == 3:
        return True
    return False

# let's all use the same zobTable, so we set the random seed
np.random.seed(42)
zobTable = np.random.randint(1,2**(5*5)-1, size=(5,5,3), dtype = np.uint32)
# compute index from current board state
def computeHash(board, n = 5, m = 5):
    h = np.uint32(0)
    for i in range(n):
        for j in range(board[i,-1]):
            h ^= zobTable[i,j,board[i,j]]
    return h

PI = np.zeros(2**25-2)
maxgame = 5*5 + 1

# Get the next hash resulting from an action on tile i,j
def nextHash(old_hash, i, j, p):
    return old_hash ^ zobTable[i,j,p]

# Training our player with „TD(0)“
def learn(greedy1 = False, greedy2 = False):
    S = iState() # initial board state
    p = 1 # first player to move (other player is 2)
    a = np.random.choice(np.where(S[:,-2] == 0)[0], 1) # first move is random
    Action(S, int(a), p) # force first move to be random
    p = getotherplayer(p) # other player's turn
    h = computeHash(S)
    ct = 1
    while True:
        a = np.where(S[:,-2] == 0)[0] # All legal moves
        if 0 == len(a): # check if a legal move was possible, else bail out
            PI[h] = 0 # Draw
            return
        vals = [PI[nextHash(h, i, S[i,-1], p)] for i in a] # Values for all possible moves
        maxv = max(vals)
        PI[h] = -maxv #
        if (p == 1 and not greedy1) or (p == 2 and not greedy2):
            # random policy
            a = np.random.choice(a, 1)[0]
        else:
            # greedy policy
            a = a[np.argmax(vals)]

        ct += 1
        Action(S, a, p) # take action a and update the board state
        h1 = nextHash(h, a, S[a,-1] - 1, p)
        if terminal(S, p, a, S[a,-1] - 1):
            endval = maxgame - ct # In order to make a losing side end slowly and a winning side to finish fast
            PI[h] = -endval # The opponent can win in the next game
            PI[h1] = endval # You have won
            return
        h = h1 # Update the hash value
        p = getotherplayer(p) # other player's turn

# Simulation
# 85/5/5/5 split r-r, g-r, r-g, g-g
for s in range(100):
    for i in range(170000):
        learn()
    for j in range(10000):
        learn(greedy1=True)
    for j in range(10000):
        learn(greedy2=True)
    for j in range(10000):
        learn(greedy1=True, greedy2=True)

# Save player
sparse.save_npz("Teymi5.npz", sparse.csr_matrix(PI))
