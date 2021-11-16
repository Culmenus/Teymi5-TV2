import time
import numpy as np
import torch
from torch.autograd import Variable

# Code snippets frá tpr
# Þarf að aðlaga kóða

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
  #m = torch.argmax(pi) # greedy
  value = va.data[0,m] # assuming the reward is zero this is the actual target value
  advantage = value - torch.sum(pi*va)
  xtheta_mean = torch.sum(torch.mm(h_tanh,torch.diagflat(pi)),1)
  h_tanh = torch.squeeze(h_tanh[:,m],1)
  grad_ln_pi = h_tanh.view(1,len(xtheta_mean)) - xtheta_mean.view(1,len(xtheta_mean))
  x_selected = Variable(torch.tensor(xa[:,m], dtype = torch.float, device = device)).view(nx,1)
  return va, m, x_selected, grad_ln_pi, value, advantage.item()

def learnit(model, alpha = [0.1, 0.001, 0.001], epsilon = 0.0, debug = False):
  # random game player to the end!
  n = 2 # number of players, they are numbered 1,2,3,4,...
  deck, hand, discs_on_board = initGame(n) # initial hand and empty discs on board!
  # lets get three types of legal moves, by normal playing cards, one-eyed Jacks (1J) and two-eyed Jacks (2J):
  player = np.random.choice([1,2],1)[0] # first player to move
  pass_move = 0 # counts how many player in a row say pass!
  nx = discs_on_board.size*3+2
  if debug:
    print("nx = ", nx)
  phi = np.zeros((nx,2))
  phiold = np.zeros((nx,2))
  # initialize all traces, to zero
  traces = traces = [len(model) * [None]] * n
  for p in range(n):
    for m in range(len(model)):
      traces[p][m] = torch.zeros(model[m].size(), device = device, dtype = torch.float)
  I = [1.0,1.0]
  grad_ln_pi = [None, None]
  advantage = [None, None]
  while True:
    legal_moves, legal_moves_1J, legal_moves_2J = getMoves(discs_on_board, hand, player)
    # now we start by finding all after-states for the possible moves
    lenJ1 = np.sum(hand[player-1] == 48)
    lenJ2 = np.sum(hand[player-1] == 49)
    na = len(legal_moves) + len(legal_moves_1J) + len(legal_moves_2J)
    move_played = [m for count in range(3) for m in [legal_moves,legal_moves_1J,legal_moves_2J][count]]
    # walk through all the possible moves, k = 0,1,...
    k = 0
    if na > 0:
      card_played = np.zeros(na)
      disc_player = np.zeros(na)
      x = np.zeros((nx,na))
      for ell in range(len(legal_moves)):  
        (i,j) = tuple(legal_moves[ell])
        card_played[k] = cards_on_board[i,j]
        disc_player[k] = player
        x[:,k] = getfeatures(discs_on_board, (i,j), player, cards_on_board[i, j], legal_moves, lenJ1, lenJ2, k, player = player)
        k += 1
      for ell in range(len(legal_moves_1J)):
        (i,j) = tuple(legal_moves_1J[ell])
        card_played[k] = 48
        disc_player[k] = 0
        x[:,k] = getfeatures(discs_on_board, (i,j), 0, 48, legal_moves, lenJ1-1, lenJ2, k, player = player)
        k += 1
      for ell in range(len(legal_moves_2J)):
        (i,j) = tuple(legal_moves_2J[ell])
        card_played[k] = 49
        disc_player[k] = player
        x[:,k] = getfeatures(discs_on_board, (i,j), player, 49, legal_moves, lenJ1, lenJ2-1, k, player = player)
        k += 1
    if k == 0:
      pass_move += 1 # this is a pass move (does this really happen?)
      #pretty_print(discs_on_board, hand)
    else:
      pass_move = 0 # zero pass counter
      # now choose an epsilon greedy move
      va, k, x_selected, grad_ln_pi[player-1], value, advantage[player-1] = softmax_policy(x, model)
      (i,j) = move_played[k] # get the actual corresponding epsilon greedy move
      discs_on_board[i,j] = disc_player[k] # here we actually update the board
      phi[:,player-1] = x[:,k] # lets keep a track of the current after-state
      # now we need to draw a new card
      deck, hand[player-1] = drawCard(deck, hand[player-1], card_played[k])
      # lets pretty print this new state og the game
      if debug:
        pretty_print(discs_on_board, hand)
    if (pass_move == n) | (True == isTerminal(discs_on_board, player, n)):
      # Bætti við að það prentar út hnitin á síðasta spili sem var spilað. Léttara að finna hvar leikmaðurinn vann.
      if pass_move == n:
        model, I[player-1] = update_model(model, traces[player-1], alpha, phiold[:,player-1], value=0.0, reward=0.5, I = I[player-1], gradlnpi = grad_ln_pi[player-1], advantage = advantage[player-1])
        model, I[player-1] = update_model(model, traces[player-1], alpha, phi[:,player-1], value=0.0, reward=0.5, I = I[player-1], gradlnpi = grad_ln_pi[player-1], advantage = advantage[player-1])
        model, I[player%n] = update_model(model, traces[player%n], alpha, phiold[:,player%n], value=0.0, reward=0.5, I = I[player%n], gradlnpi = grad_ln_pi[player%n], advantage = advantage[player%n])
      else:
        model, I[player-1] = update_model(model, traces[player-1], alpha, phiold[:,player-1], value=0.0, reward=1.0, I = I[player-1], gradlnpi = grad_ln_pi[player-1], advantage = advantage[player-1])
        model, I[player-1] = update_model(model, traces[player-1], alpha, phi[:,player-1], value=0.0, reward=1.0, I = I[player-1], gradlnpi = grad_ln_pi[player-1], advantage = advantage[player-1])
        model, I[player%n] = update_model(model, traces[player%n], alpha, phiold[:,player%n], value=0.0, reward=0.0, I = I[player%n], gradlnpi = grad_ln_pi[player%n], advantage = advantage[player%n])
      if debug:
        print("pass_move = ", pass_move, " player = ", player, " cards in deck = ", len(deck)," last played card at coords: (",i,j,")")
        print("hand = ", hand[player-1])
      break
          
    model, I[player-1] = update_model(model, traces[player-1], alpha, phiold[:,player-1], value=value, reward=0, I = I[player-1], gradlnpi = grad_ln_pi[player-1], advantage = advantage[player-1])
    phiold[:,player-1] = phi[:,player-1]
    player = player%n+1 # next player in line
  return model

def competition(model, debug = False):
  # random game player to the end!
  n = 2 # number of players, they are numbered 1,2,3,4,...
  deck, hand, discs_on_board = initGame(n) # initial hand and empty discs on board!
  # lets get three types of legal moves, by normal playing cards, one-eyed Jacks (1J) and two-eyed Jacks (2J):
  player = np.random.choice([1,2],1)[0] # first player to move
  rplayer = np.random.choice([1,2],1)[0] # the player to play random
  pass_move = 0 # counts how many player in a row say pass!
  nx = discs_on_board.size*3+2
  if debug:
    print("nx = ", nx)
  win = 0.5
  while True:
    legal_moves, legal_moves_1J, legal_moves_2J = getMoves(discs_on_board, hand, player)
    # now we start by finding all after-states for the possible moves
    lenJ1 = np.sum(hand[player-1] == 48)
    lenJ2 = np.sum(hand[player-1] == 49)
    na = len(legal_moves) + len(legal_moves_1J) + len(legal_moves_2J)
    move_played = [m for count in range(3) for m in [legal_moves,legal_moves_1J,legal_moves_2J][count]]
    # walk through all the possible moves, k = 0,1,...
    k = 0
    if na > 0:
      card_played = np.zeros(na)
      disc_player = np.zeros(na)
      x = np.zeros((nx,na))
      for ell in range(len(legal_moves)):  
        (i,j) = tuple(legal_moves[ell])
        card_played[k] = cards_on_board[i,j]
        disc_player[k] = player
        x[:,k] = getfeatures(discs_on_board, (i,j), player, cards_on_board[i, j], legal_moves, lenJ1, lenJ2, k, player = player)
        k += 1
      for ell in range(len(legal_moves_1J)):
        (i,j) = tuple(legal_moves_1J[ell])
        card_played[k] = 48
        disc_player[k] = 0
        x[:,k] = getfeatures(discs_on_board, (i,j), 0, 48, legal_moves, lenJ1-1, lenJ2, k, player = player)
        k += 1
      for ell in range(len(legal_moves_2J)):
        (i,j) = tuple(legal_moves_2J[ell])
        card_played[k] = 49
        disc_player[k] = player
        x[:,k] = getfeatures(discs_on_board, (i,j), player, 49, legal_moves, lenJ1, lenJ2-1, k, player = player)
        k += 1
    if k == 0:
      pass_move += 1 # this is a pass move (does this really happen?)
    else:
      pass_move = 0 # zero pass counter
    # now choose an epsilon greedy move
      if (rplayer != player):
        va, k = softmax_greedy_policy(x, model)
      else:
        k = np.random.choice(range(len(move_played)),1)[0]
      (i,j) = move_played[k] # get the actual corresponding epsilon greedy move
      discs_on_board[i,j] = disc_player[k] # here we actually update the board
      # now we need to draw a new card
      deck, hand[player-1] = drawCard(deck, hand[player-1], card_played[k])
      # lets pretty print this new state and the game
      if debug:
        pretty_print(discs_on_board, hand)
    if (pass_move == n)  | (True == isTerminal(discs_on_board, player, n)):
      # Bætti við að það prentar út hnitin á síðasta spili sem var spilað. Léttara að finna hvar leikmaðurinn vann.
      if debug:
        print("pass_move = ", pass_move, " player = ", player, " cards in deck = ", len(deck)," last played card at coords: (",i,j,")")
        print("hand = ", hand[player-1])
      if pass_move == n:
        win = 0.5
      elif rplayer == player:
        win = 0
      else:
        win = 1
      break
    player = player%n+1 
          
  return win 





#os.chdir('/content/gdrive/MyDrive/')
start = time.time()
#device = torch.device('cuda')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
  print(torch.cuda.current_device())
  print(torch.cuda.device(0))
  print(torch.cuda.device_count())
  print(torch.cuda.get_device_name(0))

# cuda will only create a significant speedup for large/deep networks and batched training
#device = torch.device('cpu')

# parameters for the training algorithm
alpha = [0.1, 0.001, 0.001]  # step size for PG and then each layer of the neural network

lam = 0.7 # lambda parameter in TD(lam-bda)
# define the parameters for the single hidden layer feed forward neural network
# randomly initialized weights with zeros for the biases
nx = 302
nh = np.int(nx/2)

# now perform the actual training and display the computation time
delta_train_steps = 1000
train_steps = 100000

model = 5 * [None]
if False:
  loadtrainstep = 1000
  model[0] = torch.load('./ac/b1_trained_'+str(loadtrainstep)+'.pth')
  model[1] = torch.load('./ac/w1_trained_'+str(loadtrainstep)+'.pth')
  model[2] = torch.load('./ac/b2_trained_'+str(loadtrainstep)+'.pth')
  model[3] = torch.load('./ac/w2_trained_'+str(loadtrainstep)+'.pth')
  model[4] = torch.load('./ac/theta_'+str(loadtrainstep)+'.pth')
  wins_against_random = np.load('./ac/wins_against_random.npy')
  wins_against_random = np.concatenate((wins_against_random, np.zeros(train_steps-loadtrainstep-1)))
  wins_against_random[loadtrainstep:] = 0.0
  comp_time = np.load('./ac/comp_time.npy')
  comp_time = np.concatenate((comp_time, np.zeros(train_steps-loadtrainstep-1)))
else:
  loadtrainstep = 0
  print("nx = %d, nh = %d" % (nx,nh))
  model[0] = Variable(torch.zeros((nh,1), device = device, dtype=torch.float), requires_grad = True)
  model[1] = Variable(0.1*torch.randn(nh,nx, device = device, dtype=torch.float), requires_grad = True)
  model[2] = Variable(torch.zeros((1,1), device = device, dtype=torch.float), requires_grad = True)
  model[3] = Variable(0.1*torch.randn(1,nh, device = device, dtype=torch.float), requires_grad = True)
  model[4] = Variable(0.1*torch.randn(1,nh, device = device, dtype=torch.float), requires_grad = True)
  wins_against_random = np.zeros(train_steps)
  comp_time = np.zeros(train_steps)

print(len(wins_against_random))
for trainstep in range(loadtrainstep,train_steps):
  print("Train step ", trainstep, " / ", train_steps)
  start = time.time()
  for k in range(delta_train_steps):
    model = learnit(model, alpha)
  for i in range(100):
    war = competition(model)
    wins_against_random[trainstep] += war
  print("wins against random = ", wins_against_random[trainstep]/100*100)
  torch.save(model[0], './ac/b1_trained_'+str(trainstep)+'.pth')
  torch.save(model[1], './ac/w1_trained_'+str(trainstep)+'.pth')
  torch.save(model[2], './ac/b2_trained_'+str(trainstep)+'.pth')
  torch.save(model[3], './ac/w2_trained_'+str(trainstep)+'.pth')
  torch.save(model[4], './ac/theta_'+str(trainstep)+'.pth')
  np.save('./ac/wins_against_random.npy', wins_against_random)
  # estimate the computation time to complete
  end = time.time()
  comp_time[trainstep] = np.round((end - start)/60)
  print("estimated time remaining: ", comp_time[trainstep]*(train_steps-loadtrainstep-trainstep+1)/60," (hours)")
  np.save('./ac/comp_time.npy', comp_time)
