import time
import numpy as np
import torch
from torch.autograd import Variable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from leikmadur_teymi_5 import *

# which player is playing it and nr of players (n).
def fiveInRow(row, col, discs_on_board, player, n):
    tempWin = [[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
               [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
               [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
               [0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
               [0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
               [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]]
    other_player = player % n+1
    players_discs = discs_on_board.copy()
    players_discs[players_discs == -1] = other_player
    players_discs = players_discs == other_player

    rows = list(players_discs[row])
    cols = [i[col] for i in players_discs]
    # Make the diagonal lines
    temp_bdiag = []
    temp_fdiag = []
    temp_fdiag2 = []
    temp_bdiag2 = []
    for i in range(0, len(players_discs[0])):
        if col - i >= 0 and row - i >= 0:
            temp_fdiag2.append(players_discs[row - i][col - i])
        if col + i < 10 and row + i < 10:
            temp_fdiag.append(players_discs[row + i][col + i])
        if col + i < 10 and row - i >= 0:
            temp_bdiag2.append(players_discs[row - i][col + i])
        if col - i >= 0 and row + i < 10:
            temp_bdiag.append(players_discs[row + i][col - i])
    temp_fdiag2.reverse()
    temp_bdiag2.reverse()
    if len(temp_fdiag) > 1:
        if len(temp_fdiag2) > 0:
            temp_fdiag2 = temp_fdiag2 + temp_fdiag[1:]
        else:
            temp_fdiag2 = temp_fdiag

    if len(temp_bdiag) > 1:
        if len(temp_bdiag2) > 0:
            temp_bdiag2 = temp_bdiag2 + temp_bdiag[1:]
        else:
            temp_bdiag2 = temp_bdiag
    # Fill them up if shorter than 10
    if(len(temp_bdiag2) < 10):
        temp_bdiag2 = temp_bdiag2 + ([False]*(10 - len(temp_bdiag2)))
    if(len(temp_fdiag2) < 10):
        temp_fdiag2 = temp_fdiag2 + ([False]*(10 - len(temp_fdiag2)))

    lists = list(filter(lambda discs: np.sum(discs) >= 5,
                 [cols, rows, temp_fdiag2, temp_bdiag2]))
    for i in lists:
        for j in tempWin:
            # check if 5 in row and if the correct "reitur" is also being checked
            if sum(np.multiply(i, j)) >= 5 and j[col] == 1:
                # Oh no this is not allowed
                return True
    # Yay lets go
    return False

# (floki@hi.is)
# discs_on_board er 10x10 matrix með stöðunni eftir að búið er að leggja niður "disk"
# player er nr hvaða player er verið að skoða
# n er fjöldi leikmanna (bara til þess að tékka hvort það séu 2
def isTerminal(discs_on_board, player, n):
    # regular win conditons
    tempWin = [[1,1,1,1,1,0,0,0,0,0],
              [0,1,1,1,1,1,0,0,0,0],
              [0,0,1,1,1,1,1,0,0,0],
              [0,0,0,1,1,1,1,1,0,0],
              [0,0,0,0,1,1,1,1,1,0],
              [0,0,0,0,0,1,1,1,1,1]]
    # extra win conditions for 2 players
    extraWin = [[0,1,1,1,1,1,1,1,1,1],
                [1,1,1,1,1,1,1,1,1,0],
              ]
    temp_board = discs_on_board.copy()
    temp_board[temp_board == -1] = player
    test = temp_board == player

    # Find all "lines". Row, col and diagonal.
    max_col = len(test[0])
    max_row = len(test)
    cols = [[] for _ in range(max_col)]
    rows = [[] for _ in range(max_row)]
    fdiag = [[] for _ in range(max_row + max_col - 1)]
    bdiag = [[] for _ in range(len(fdiag))]
    min_bdiag = -max_row + 1

    for x in range(max_col):
        for y in range(max_row):
            cols[x].append(test[y][x])
            rows[y].append(test[y][x])
            fdiag[x+y].append(test[y][x])
            bdiag[x-y-min_bdiag].append(test[y][x])
    lists = cols + rows + bdiag + fdiag

    # make one list of all lines
    np_lists = np.array(lists, dtype=object)

    # filter out lines shorter than 5 and lines that contain less than 5 discs
    filt = []
    for i in range(0,len(lists)):
        filt_i = len(np_lists[i]) >= 5 and sum(np_lists[i]) >= 5
        filt.append(filt_i)

    #combine into one list of lists
    new_list = list(np_lists[filt])
    temp = []

    #fill lines up to 10 for array multiplication
    for i in new_list:
        if(len(i) < 10):
            i = i + ([0.0]*(10 - len(i)))
        temp.append(i)
    # check if there are no lines
    if(len(temp) == 0):
        return False

    # use this if only 2 players
    player_wins = 0
    # Check every line if win condition is fulfilled
    for i in temp:
      # Regular win conditions, checks if any of the regular win conditions is met
      wins = [sum(np.multiply(i,j)) >= 5 for j in tempWin]
      # if 2 players then player needs to fulfill 2 win conditions
      if n == 2:
        # Check if any line is 9 or more in a row
        extra_wins = [sum(np.multiply(i, j)) >= 9 for j in extraWin]
        # if atleast 1 win conditon fulfilled then add win to player_wins
        if np.sum(wins) >= 1:
          player_wins = player_wins + 1
        # extra_wins = lines with 9 or more in a row
        # player_wins = nr of win conditions fulfilled
        if np.sum(extra_wins) >= 1 or player_wins >= 2:
          return True
      # if nr players > 2 only one win condition needed
      else:
        if np.sum(wins) >= 1:
          return True
    return False

# (tpr@hi.is)
# some global variables used by the games, what we get in the box!
the_cards = ['AC','2C','3C','4C','5C','6C','7C','8C','9C','1C','QC','KC',
             'AS','2S','3S','4S','5S','6S','7S','8S','9S','1S','QS','KS',
             'AD','2D','3D','4D','5D','6D','7D','8D','9D','1D','QD','KD',
             'AH','2H','3H','4H','5H','6H','7H','8H','9H','1H','QH','KH',
             '1J','2J']
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
# initialize the game, n is the number of players
def initGame(n = 2):
  discs_on_board = np.zeros((10,10), dtype = 'int8') # empty!
  discs_on_board[np.ix_([0,0,9,9],[0,9,0,9])] = -1 # the corners are "-1"
  # There are two decks of cards each with 48 unique cards if we remove the Jacks lets label them 0,...,47
  # Let card 48 by one-eyed Jack and card 49 be two-eyed jack there are 4 each of these
  cards = np.hstack((np.arange(48),np.arange(48),48,48,48,48,49,49,49,49))
  deck = cards[np.argsort(np.random.rand(104))] # here we shuffle the cards, note we cannot use shuffle (we have non-unique cards)
  # now lets deal out the hand, each player gets m[n] cards
  m = [None, None, 7, 6, 6]
  hand = []
  for i in range(n):
    hand.append(deck[:m[n]]) # deal player i m[n] cards
    deck = deck[m[n]:] # remove cards from deck
  return deck, hand, discs_on_board
# printing the board is useful for debugging code...
def pretty_print(discs_on_board, hand):
  color = ["","*","*","*","*"]
  for i in range(10):
    for j in range(10):
      if (discs_on_board[i,j] <= 0):
        if cards_on_board[i,j] >= 0:
          print(the_cards[cards_on_board[i,j]], end = " ")
        else:
          print("-1", end = " ")
      else:
        print(color[discs_on_board[i,j]]+str(discs_on_board[i,j]), end = " ")
    print("")
  for i in range(len(hand)):
    print("player ", i+1, "'s hand: ", [the_cards[j] for j in hand[i]], sep = "")
# get all feasible moved for normal cards, one-eyed jacks and two-eyed jacks
def getMoves(discs_on_board, hand, player, n = 2, debug=False):
    # legal moves for normal playing cards
    # check for cards in hand
    iH = np.in1d(cards_on_board, hand[player-1]).reshape(10, 10)
    iA = (discs_on_board == 0)  # there is no disc blocking
    legal_moves = np.argwhere(iH & iA)
    # legal moves for one-eyed Jacks (they remove), BUG: NOT when we have a 5 in a row (two players!!!)
    if 48 in hand[player-1]:
        legal_moves_1J = []
        temp_legal_moves_1J = np.argwhere(
            (discs_on_board != -1) & (discs_on_board != 0) & (discs_on_board != player))
        if n == 2:
            for i in temp_legal_moves_1J:
                temp = fiveInRow(i[0], i[1], discs_on_board, player, n)
                if temp == False:
                    legal_moves_1J.append(i)
        legal_moves_1J = np.array(legal_moves_1J)
    else:
        legal_moves_1J = np.array([])
    # legal moves for two-eyed Jacks (they are wild)
    if 49 in hand[player-1]:
        legal_moves_2J = np.argwhere(discs_on_board == 0)
    else:
        legal_moves_2J = np.array([])
    if debug:
        print("legal_moves for player ", player)
        for i, j in legal_moves:
            print(the_cards[cards_on_board[i, j]], end=" ")
        print("")
    return legal_moves, legal_moves_1J, legal_moves_2J

def drawCard(deck, hand, card_played, debug = False):
  # remove card player from hand
  if len(deck) > 0:
    new_card = deck[0] # take top card from the deck
    deck = deck[1:] # remove the card from the deck
    i = np.where(hand == card_played) # find location of card played in hand
    if debug:
      print("Hand before change",hand)
    if len(i) > 0:
      hand[i[0][0]] = new_card # replace the card played with a new one
    else:
      print("drawCard, could not find this cards in the current hand?!")
      raise
    if debug:
      print("Hand after change", hand)
  else:
    i = np.where(hand == card_played) # find location of card played in hand
    if debug:
      print("Hand before change",hand)
    if len(i) > 0:
      hand = np.delete(hand, i[0][0]) # set invalid card
    else:
      print("drawCard, could not find this cards in the current hand?!")
      raise
    if debug:
      print("Hand after change", hand)

  return deck, hand

def getfeatures(discs_on_board, move, disc, card, normal_moves, len1J, len2J, kmove, player, p1J=0, p2J=0, n = 2):
  b = discs_on_board.copy()
  #(i,j) = move
  b[move[0],move[1]] = disc # the look-ahead
  if (True == isTerminal(b, player, n)):
    print("ATH: winning move (%d,%d) for player %d" % (move[0],move[1],player))
    winning_move = True
  else:
    winning_move = False
  winning_move = False
  otherplayer = player % n + 1
  # lets put our player in the corners
  b[0,0] = b[0,-1] = b[-1,0] = b[-1,-1] = player
  # other potential moves we can make with our current hand
  for k in range(len(normal_moves)):
    if k != kmove:
      (i,j) = normal_moves[k]
      b[i,j] = -1 # potential move for the player in the future!
  b = b.reshape(b.size)
  x = np.concatenate((b == 0, b == player, b == otherplayer, b == -1, np.array([len1J]), np.array([len2J]),np.array([p1J]), np.array([p2J])))
  return x, winning_move

def softmax_greedy_policy(xa,  model):
  (nx,na) = xa.shape
  x = Variable(torch.tensor(xa, dtype = torch.float, device = device))
  # now do a forward pass to evaluate the board's after-state value
  h = torch.mm(model[1],x) + model[0] @ torch.ones((1,na),device = device)  # matrix-multiply x with input weight w1 and add bias
  h_tanh = h.tanh() # squash this with a sigmoid function
  y = torch.mm(model[3],h_tanh) + model[2] # multiply with the output weights w2 and add bias
  va = y.sigmoid().detach() # .cpu()
  # now for the actor:
  pi = torch.mm(model[4],h_tanh).softmax(1)
  m = torch.argmax(pi) # soft
  return m

def softmax_policy(xa,  model):
  (nx,na) = xa.shape
  x = Variable(torch.tensor(xa, dtype = torch.float, device = device))
  # now do a forward pass to evaluate the board's after-state value
  h = torch.mm(model[1],x) + model[0] @ torch.ones((1,na),device = device)  # matrix-multiply x with input weight w1 and add bias
  h_tanh = h.tanh() # squash this with a sigmoid function
  y = torch.mm(model[3],h_tanh) + model[2] # multiply with the output weights w2 and add bias
  va = y.sigmoid().detach() # .cpu()
  # now for the actor:
  pi = torch.mm(model[4],h_tanh).softmax(1)
  m = torch.multinomial(pi, 1) # soft
  return m

def greedy_policy(xa,  model):
  (nx,na) = xa.shape
  x = Variable(torch.tensor(xa, dtype = torch.float, device = device))
  # now do a forward pass to evaluate the board's after-state value
  h = torch.mm(model[1],x) + model[0] @ torch.ones((1,na),device = device)  # matrix-multiply x with input weight w1 and add bias
  h_tanh = h.tanh() # squash this with a sigmoid function
  y = torch.mm(model[3],h_tanh) + model[2] # multiply with the output weights w2 and add bias
  va = y.sigmoid().detach() # .cpu()
  m = torch.argmax(va[0,:]) # greedy, should we break ties here, I assume there will be none...
  return m, va.data[0,:]

def competition(p1, p2, debug = False):
  # random game player to the end!
  n = 2 # number of players, they are numbered 1,2,3,4,...
  deck, hand, discs_on_board = initGame(n) # initial hand and empty discs on board!
  # lets get three types of legal moves, by normal playing cards, one-eyed Jacks (1J) and two-eyed Jacks (2J):
  player = 1 # first player to move
  pass_move = 0 # counts how many player in a row say pass!
  win = 0.5
  while True:
    legal_moves, legal_moves_1J, legal_moves_2J = getMoves(discs_on_board, hand, player)
    if 1 == player:
      (i,j), spil = p1.policy(discs_on_board, hand[player-1], legal_moves, legal_moves_1J, legal_moves_2J)
    else:
      (i,j), spil = p2.policy(discs_on_board, hand[player-1], legal_moves, legal_moves_1J, legal_moves_2J)
    if spil == None:
      pass_move += 1 # this is a pass move (does this really happen?)
    else:
      pass_move = 0 # zero pass counter
      if (spil == 48):
        discs_on_board[i,j] = 0
      else:
        discs_on_board[i,j] = player # here we actually update the board
      # now we need to draw a new card
      deck, hand[player-1] = drawCard(deck, hand[player-1], spil)
      # lets pretty print this new state og the game
      if debug:
        pretty_print(discs_on_board, hand)
    if (pass_move == n)  | (True == isTerminal(discs_on_board, player, n)):
      # Bætti við að það prentar út hnitin á síðasta spili sem var spilað. Léttara að finna hvar leikmaðurinn vann.
      if debug:
        print("pass_move = ", pass_move, " player = ", player, " cards in deck = ", len(deck)," last played card at coords: (",i,j,")")
        print("hand = ", hand[player-1])
        pretty_print(discs_on_board, hand)
        print(discs_on_board)
        print("random player is %d and player %d won" % (player, player))
      if pass_move == n:
        win = 0.5
      elif 2 == player:
        win = 0
      else:
        win = 1
      break
    player = player%n+1
  return win

model_1 = []
model_1.append(leikmadur_teymi_5a(1))
model_1.append(leikmadur_teymi_5b(1))
model_2 = []
model_2.append(leikmadur_teymi_5a(2))
model_2.append(leikmadur_teymi_5b(2))

for i in range(2):
  for j in range(2):
    war = 0
    for k in range(100):
      result = competition(model_1[i], model_2[j],  debug = False)
      war += result
    print("final number wins for player %s (first move) against %s (second move) is %.1f" % (model_1[i].name,model_2[j].name,war))