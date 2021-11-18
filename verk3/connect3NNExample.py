import numpy as np
import torch
from torch.autograd import Variable

# Dæmi um fullbúið TD-lambda með tauganeti

def connect3TDLambda(number_episodes, w1, b1, w2, b2, epsilon = 0.1, alpha = 0.001, alpha2 = 0.1, lam = 0.9, gamma = 1.0):
  V = np.zeros(2**(5*5)) # this is *almost* a perfect function expressed by a table for player 2
  nx = 3*25
  for e in range(number_episodes):
    # zero the eligbility traces
    Z_w1 = torch.zeros(w1.size(), device = device, dtype = torch.float)
    Z_b1 = torch.zeros(b1.size(), device = device, dtype = torch.float)
    Z_w2 = torch.zeros(w2.size(), device = device, dtype = torch.float)
    Z_b2 = torch.zeros(b2.size(), device = device, dtype = torch.float)
    target = [0.0,0.0] # some memory space for targets, player 1 and 2
    S = iState() # initial board state
    p = 1 # first player to move (other player is 2)
    a = np.random.choice(np.where(S[:,-2]==0)[0],1) # first move is random
    S = Action(S,int(a),p) # force first move to be random
    s_index = computeHash(S)
    p = getotherplayer(p) # other player's turn
    old_index = s_index # this s_index will only be used by player to move first
    move = 0 # reset move counter
    play = True # I assume!
    while play: # let the game begin!
      move += 1 # increment move counter
      As = np.where(S[:,-2]==0)[0] # get index to all feasible moves
      na = len(As) # number of possible moves
      if 0 == na: # check if a legal move was possible, else bail out
        play = False
        target = [0.0, 0.5] # its a draw
        reward = 0.5
      else:
        va = np.zeros((na)) # the value of these afterstates
        if 1 == p:
          xa = np.zeros((na,nx)) # all after-states for the different moves
          for i in range(0, na):
            xa[i,:] = one_hot_encoding(Action(S.copy(),int(As[i]),p))
          x = Variable(torch.tensor(xa.transpose(), dtype = torch.float, device = device))
          # now do a forward pass to evaluate the board's after-state value
          h = torch.mm(w1,x) + b1 # matrix-multiply x with input weight w1 and add bias
          h_sigmoid = h.tanh() # squash this with a sigmoid function
          y = torch.mm(w2,h_sigmoid) + b2 # multiply with the output weights w2 and add bias
          va = y.sigmoid().detach().cpu().numpy().flatten()
        else:
          for i in range(0, na):
            a = int(As[i])
            s_index ^= zobTable[a,S[a,-1],p] # do lookup
            va[i] = V[s_index] # extract after-state value
            s_index ^= zobTable[a,S[a,-1],p] # undo lookup
        vmax = np.max(va)
        if np.random.rand() < epsilon: # epsilon greedy
          a = np.random.choice(As,1) # pure random policy
        else:
          a = np.random.choice(As[vmax == va],1) # greedy policy, break ties randomly
        S = Action(S,int(a),p) # take action a and update the board state
        if 1 == p:
          phi = one_hot_encoding(S) # we keep a record of the after-state attribues observed by player 1
          # phisym = one_hot_encoding(np.flipud(S))
        else:
          s_index = computeHash(S) # current board index (after-state for player p)
        if Terminal(S,p):
          play = False # return the winning player, game over
          if 1 == p:
            target = [0.0, 0.0]
            reward = 1.0
          else:
            target = [0.0, 1.0]
            reward = 0.0
        else:
          target = [va[np.where(a==As)[0]][0],vmax] # p1 uses SARSA
          reward = 0
      # now perform the TD update:
      if move > 2:
        if 1 == p:
        # lets do a forward past for the old board, this is the state we will update
          h = torch.mm(w1,xold) + b1 # matrix-multiply x with input weight w1 and add bias
          h_sigmoid = h.tanh() # squash this with a sigmoid function
          y = torch.mm(w2,h_sigmoid) + b2 # multiply with the output weights w2 and add bias
          y_sigmoid = y.sigmoid() # squash the output
          # now compute all gradients
          y_sigmoid.backward()
          # update the eligibility traces using the gradients
          Z_w1 = gamma * lam * Z_w1 + w1.grad.data
          Z_b1 = gamma * lam * Z_b1 + b1.grad.data
          Z_w2 = gamma * lam * Z_w2 + w2.grad.data
          Z_b2 = gamma * lam * Z_b2 + b2.grad.data
          # zero the gradients
          w1.grad.data.zero_()
          b1.grad.data.zero_()
          w2.grad.data.zero_()
          b2.grad.data.zero_()
          # perform now the update for the weights
          delta =  reward + gamma * target[0] - y_sigmoid.detach() # this is the usual TD error
          delta = torch.tensor(delta, dtype = torch.float, device = device)
          w1.data = w1.data + alpha * delta * Z_w1 # may want to use different alpha for different layers!
          b1.data = b1.data + alpha * delta * Z_b1
          w2.data = w2.data + alpha * delta * Z_w2
          b2.data = b2.data + alpha * delta * Z_b2
        else:
          V[old_index] = V[old_index] + alpha2 * (target[1] - V[old_index]) # Q-learning
      if (p == 1):
        xold = Variable(torch.tensor(phi.reshape((len(phi),1)), dtype = torch.float, device = device))
      else:
        old_index = s_index

      p = getotherplayer(p) # other player's turn
    
  return V, w1, b1, w2, b2 # the learned after-state value functions