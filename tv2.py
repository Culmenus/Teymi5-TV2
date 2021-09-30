import numpy as np

# two player game (1) versus (2)
getotherplayer = lambda p: 3 - p  # returns the other player

# the initial empty board, in matrix board we store legal indices in board[:,-1]
def iState(n=5, m=5):
    return np.zeros((n, m + 1), dtype=np.uint16)


# perform move for player on board
def Action(board, move, player):
    if board[move, -2] > 0:
        print("illegal move ", move, " for board ", np.flipud(board.T))
        raise
    else:
        board[move, board[move, -1]] = player  # place the disc on board
        board[move, -1] += 1  # next legal drop
    return board


# determine if terminal board state, assuming last move was made by player p
def terminal(board, p, i, j, n=5, m=5):
    # Horizontal
    t = -min(2, i)
    count = 0
    while t <= min(2, n - 1 - i) and count < 3:
        if board[i + t, j] == p:
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
    while t <= 0:
        if board[i, j + t] == p:
            count += 1
        else:
            count = 0
        t += 1
    if count == 3:
        return True

    # Main diagonal
    t = -min(2, min(i, j))
    count = 0
    while t <= min(2, min(n - 1 - i, m - 1 - j)) and count < 3:
        if board[i + t, j + t] == p:
            count += 1
        else:
            count = 0
        t += 1
    if count == 3:
        return True

    # Antidiagonal
    t = -min(2, min(i, m - 1 - j))
    count = 0
    while t <= min(2, min(n - 1 - i, j)) and count < 3:
        if board[i + t, j - t] == p:
            count += 1
        else:
            count = 0
        t += 1
    if count == 3:
        return True
    return False


# Some pretty way of displaying the board in the terminal
def pretty_print(board, n=5, m=5, symbols=" XO"):
    for num in range(1, n + 1):
        print(" " + str(num) + " ", end=" ")
    print()
    for j in range(m):
        for i in range(n):
            print(" " + symbols[board[i, m - 1 - j]] + " ", end=" ")
        print("")


# ---

"""
# let's all use the same zobTable, so we set the random seed
np.random.seed(42)
zobTable = np.random.randint(1,2**(5*5)-1, size=(5,5,3), dtype = np.uint32)
# compute index from current board state
def computeHash(board, n = 5, m = 5):
  h = 0
  for i in range(n):
    for j in range(board[i,-1]):
      h ^= zobTable[i,j,board[i,j]]
  return h
"""
np.random.seed(42)
zobTable = np.random.randint(1, 2 ** 64 - 1, size=(5, 5, 3), dtype=np.uint64)
# compute index from current board state
def computeHash(board, n=5, m=5):
    h = np.uint64(0)
    for i in range(n):
        for j in range(board[i, -1]):
            h ^= zobTable[i, j, board[i, j]]
    return h


# ---

maxHashValue = 2 ** (5 * 5) - 2
# V = np.zeros(maxHashValue, dtype=np.int16)
V = {}
maxgame = 5 * 5 + 1

# ---


def nextHash(old_hash, i, j, p):
    return old_hash ^ zobTable[i, j, p]


def getVal(h):
    if h in V:
        return V[h]
    return 0


# Save your policy, PI to file, see folder icon on left hand side to download!
# np.save("teymiX", PI)
# !ls

upd = [0]


def learn(greedy1=False, greedy2=False, pr=False):
    S = iState()  # initial board state
    p = 1  # first player to move (other player is 2)
    a = np.random.choice(np.where(S[:, -2] == 0)[0], 1)  # first move is random
    if pr:
        print(int(a), end="")
    S = Action(S, int(a), p)  # force first move to be random
    p = getotherplayer(p)  # other player's turn
    h = computeHash(S)
    b = pr
    ct = 1
    while True:
        a = np.where(S[:, -2] == 0)[0]
        if 0 == len(a):  # check if a legal move was possible, else bail out
            V[h] = 1
            if pr:
                print("draw, moves = {}".format(ct))
            return 0  # its a draw, return 0 and board
        vals = [getVal(nextHash(h, i, S[i, -1], p)) for i in a]
        # vals = [V[nextHash(h, i, S[i,-1], p)] for i in a]
        v = getVal(h)
        maxv = -min(vals)
        if v != maxv:
            upd[0] += 1
        V[h] = maxv
        if (p == 1 and not greedy1) or (p == 2 and not greedy2):
            # random policy
            a = np.random.choice(a, 1)[0]
        else:
            # greedy policy
            a = a[np.argmin(vals)]

        ct += 1
        S = Action(S, a, p)  # take action a and update the board state
        h1 = nextHash(h, a, S[a, -1] - 1, p)
        if terminal(S, p, a, S[a, -1] - 1):
            V[h] = maxgame - ct
            V[h1] = -maxgame + ct
            if pr:
                print("w = {}, moves = {}".format(p, ct))
                # pretty_print(S)
            return p
        if pr:
            if b:
                print(", {}: ".format(a), end="")
                b = False
            print(getVal(h), end=" ")
        h = h1  # Update the hash value
        p = getotherplayer(p)  # other player's turn


for s in range(5):
    for i in range(10):
        upd[0] = 0
        for j in range(10000):
            learn()
        print("{}: {}".format(i, upd[0]))
        for j in range(3):
            learn(greedy1=True, greedy2=True, pr=True)
    w = 0
    l = 0
    for j in range(10000):
        c = learn(greedy1=True)
        if c == 1:
            w += 1
        elif c == 2:
            l += 1
    print("Player 1 greedy:\nWins: {}\nLosses: {}\n".format(w, l))
    w = 0
    l = 0
    for j in range(10000):
        c = learn(greedy2=True)
        if c == 1:
            w += 1
        elif c == 2:
            l += 1
    print("Player 2 greedy:\nWins: {}\nLosses: {}\n".format(l, w))
    w = 0
    l = 0
    for j in range(10000):
        c = learn(greedy1=True, greedy2=True)
        if c == 1:
            w += 1
        elif c == 2:
            l += 1
    print("Both greedy:\nPlayer 1 win: {}\nPlayer 2 win: {}".format(w, l))
