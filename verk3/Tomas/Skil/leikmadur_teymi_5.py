import torch
from torch.autograd import Variable
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cards_on_board = np.matrix([[-1, 0,11,10, 9, 8, 7, 6, 5,-1],
                            [24,18,19,20,21,22,23,12, 4,13],
                            [35,17, 9, 8, 7, 6, 5,25, 3,14],
                            [34,16,10,43,42,41, 4,26, 2,15],
                            [33,15,11,44,37,40, 3,27, 1,16],
                            [32,14, 0,45,38,39, 2,28,36,17],
                            [31,13,24,46,47,36, 1,29,47,18],
                            [30,37,35,34,33,32,31,30,46,19],
                            [29,38,39,40,41,42,43,44,45,20],
                            [-1,28,27,26,25,12,23,22,21,-1]])

class leikmadur_teymi_5a:
  def __init__(self, player):
    self.name = "Teymi 5 Random"
    self.player = player

  def policy(self, discs_on_board, cards_in_hand, legal_moves, legal_moves_1J, legal_moves_2J):
    len1 = len(legal_moves)
    len2 = len1 + len(legal_moves_1J)
    if legal_moves_1J.size == 0:
      legal_moves_1J = np.array([]).reshape(0, 2)
    if legal_moves_2J.size == 0:
      legal_moves_2J = np.array([]).reshape(0, 2)
    all_moves = np.concatenate((legal_moves, legal_moves_1J, legal_moves_2J)).astype(np.int8)
    played_card = 0
    i, j = 0, 0
    if len(all_moves) > 0:
      k = np.random.randint(0, len(all_moves))
      i, j = all_moves[k]
      if k < len1:
        played_card = cards_on_board[i,j]
      elif k < len2:
        played_card = 48
      else:
        played_card = 49

      return (i, j), played_card
    else:
      return (None, None), None

class leikmadur_teymi_5b:
  def __init__(self, player):
    self.name = "Teymi 5 Megabot"
    self.player = player
    self.model = [None] * 4
    self.model[0] = torch.load('../b1_trained.pth')
    self.model[1] = torch.load('../w1_trained.pth')
    self.model[2] = torch.load('../b2_trained.pth')
    self.model[3] = torch.load('../w2_trained.pth')
    self.sequences = [0, 0]

  def update_sequences(self, discs_on_board, i, j, player):
    # Updates a set containing the positions of all
    # discs which are part of a sequence, given that
    # the last move was played in position (i, j)
    if discs_on_board[i,j] != player:
      return
    p = player - 1

    temp_board = discs_on_board.copy()
    # Crude search; checks if the game is relatively new
    temp_sum = np.sum(temp_board == player)
    if temp_sum < 4:
      self.sequences[p] = 0
      return
    temp_board[0,0], temp_board[0,9], temp_board[9,0], temp_board[9,9] = [self.player]*4

    # Horizontal
    t1 = 1
    while j + t1 < 10 and temp_board[i,j+t1] == player:
      t1 += 1
    t1 -= 1
    t2 = 1
    while j - t2 >= 0 and temp_board[i,j-t2] == player:
      t2 += 1
    t2 -= 1
    t = t1 + t2
    if t >= 4:
      if t >= 8:
        self.sequences[p] = 2
        return
      if t1 < 5 and t2 < 5:
        self.sequences[p] += 1

    # Vertical
    t1 = 1
    while i + t1 < 10 and temp_board[i+t1,j] == player:
      t1 += 1
    t1 -= 1
    t2 = 1
    while i - t2 >= 0 and temp_board[i-t2,j] == player:
      t2 += 1
    t2 -= 1
    t = t1 + t2
    if t >= 4:
      if t >= 8:
        self.sequences[p] = 2
        return
      if t1 < 5 and t2 < 5:
        self.sequences[p] += 1

    # Main diagonal
    t1 = 1
    while i + t1 < 10 and j + t1 < 10 and temp_board[i+t1,j+t1] == player:
      t1 += 1
    t1 -= 1
    t2 = 1
    while i - t2 >= 0 and j - t2 >= 0 and temp_board[i-t2,j-t2] == player:
        t2 += 1
    t2 -= 1
    t = t1 + t2
    if t >= 4:
      if t >= 8:
        self.sequences[p] = 2
        return
      if t1 < 5 and t2 < 5:
        self.sequences[p] += 1

    # Antidiagonal
    t1 = 1
    while i + t1 < 10 and j - t1 >= 0 and temp_board[i+t1,j-t1] == player:
      t1 += 1
    t1 -= 1
    t2 = 1
    while i - t2 >= 0 and j + t2 < 10 and temp_board[i-t2,j+t2] == player:
      t2 += 1
    t2 -= 1
    t = t1 + t2
    if t >= 4:
      if t >= 8:
        self.sequences[p] = 2
        return
      if t1 < 5 and t2 < 5:
        self.sequences[p] += 1

  def getfeatures(self, discs_on_board, cards_in_hand, pos, card, disc):
    i, j = pos
    temp_board = discs_on_board.copy()
    temp_board[i,j] = disc
    if self.player == 2:
      for i in range(10):
        for j in range(10):
          if temp_board[i,j] > 0:
            temp_board[i,j] = 3 - temp_board[i,j]
    temp_board = temp_board[temp_board != -1]
    temp_board = temp_board.flatten()
    one_hot_board = np.zeros((temp_board.size, 3))
    one_hot_board[np.arange(temp_board.size),temp_board] = 1
    one_hot_board = one_hot_board.flatten()

    cards = np.zeros(50, dtype=np.int8)
    np.add.at(cards, cards_in_hand, 1)
    cards[card] -= 1
    assert(cards[card] >= 0)

    old_sequences = self.sequences.copy()
    self.update_sequences(discs_on_board, i, j, 1)
    self.update_sequences(discs_on_board, i, j, 2)
    seqs = []
    if self.player == 1:
      seqs = np.array([self.sequences[0], self.sequences[1]])
    else:
      seqs = np.array([self.sequences[1], self.sequences[0]])

    if self.player == 2:
      for i in range(temp_board.size):
        if temp_board[i] > 0:
          temp_board[i] = 3 - temp_board[i]

    self.sequences = old_sequences

    return np.concatenate((one_hot_board, cards, seqs))

  def policy(self, discs_on_board, cards_in_hand, legal_moves, legal_moves_1J, legal_moves_2J):
    len1 = len(legal_moves)
    len2 = len1 + len(legal_moves_1J)
    if legal_moves_1J.size == 0:
      legal_moves_1J = np.array([]).reshape(0, 2)
    if legal_moves_2J.size == 0:
      legal_moves_2J = np.array([]).reshape(0, 2)
    all_moves = np.concatenate((legal_moves, legal_moves_1J, legal_moves_2J)).astype(np.int8)
    played_card = 0
    i, j = 0, 0
    if len(all_moves) > 0:
      # Find afterstate values
      x = np.zeros((340, len(all_moves)))
      t = 0
      for i in range(len(legal_moves)):
        x1, x2 = legal_moves[i]
        x[:,t] = self.getfeatures(discs_on_board, cards_in_hand, legal_moves[i], cards_on_board[x1,x2], self.player)
        t += 1
      for i in range(len(legal_moves_1J)):
        x[:,t] = self.getfeatures(discs_on_board, cards_in_hand, legal_moves_1J[i], 48, 0)
        t += 1
      for i in range(len(legal_moves_2J)):
        x[:,t] = self.getfeatures(discs_on_board, cards_in_hand, legal_moves_2J[i], 49, self.player)
        t += 1

      x = Variable(torch.tensor(x, dtype=torch.float, device=device))
      h = torch.mm(self.model[1], x) + self.model[0] @ torch.ones((1,len(all_moves)), device=device)
      h_tanh = h.tanh()
      y = torch.mm(self.model[3], h_tanh) + self.model[2]
      values = y.sigmoid().detach()

      k = int(torch.argmax(values))

      i, j = all_moves[k]
      if k < len1:
        played_card = cards_on_board[i,j]
      elif k < len2:
        played_card = 48
      else:
        played_card = 49

      self.update_sequences(discs_on_board, i, j, 0)
      self.update_sequences(discs_on_board, i, j, 1)
      return (i, j), played_card
    else:
      return (None, None), None