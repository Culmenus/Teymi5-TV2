# Hugmynd 1:

- Spila nokkra leiki epsilon-greedy, safna upplýsingum (s, v) (hvaða target?)
- Fóðra tauganet með þessum upplýsingum, fá bætt virðisfall
- Endurtaka

<!-- -->

    nn = some_neural_network    # sennilega torch nn-hlutur, einhvers konar tauganet
    
    for i in some_range:
      v = np.array()
      for j in batch_size:
        v.append(play())        # play skilar reynslunni (s, v)
      nn.learn(v)               # Notum innbyggðar lærdómsaðferðir úr torch

# Hugmynd 2 (betri)

- Spila leiki epsilon-greedy, gera TD-lambda uppfærslur jafnóðum


<!-- Oddur að pæla eitthvað hér fyrir neðan-->


import SequenceEnv()

env = SequenceEnv()
state = env.gameInit(num_players = 2)

num_obersvable_features = len(np.flatten(state.discs_on_board)) + len(env.hand[0] #lengd á fletjuðu borði og init hendi player 1 sem fyrir
                                                                                 #tvo playera væri 7 að lengd.

state_values = np.zeros(num_obersvable_features) # value reita og handa?
eligibility = np.zeros(num_players, num_observable_features) #sama of fyrir ofan

for step in range(n_steps):

<!--hmm ætti tauganet ekki að sjá um elegibility??-->

# Meira

```
For episodes:
	S, hands <- Initialize board and hands
	z <- Initialize elegibility trace for each player
	player <- 1
	delta <- [0, 0]
	For steps in episode:
		moves <- getMoves(S, hands[player])
		for move in moves:
			**find maxValue of value(play(S,move))
		# action from epsilon greedy policy
		action <- maxValue or epsilon random
		# Observation of new state and reward
		Sˆ, R <- play(S, action) # R \in (0, 0.5, 1) in terminal states for (loss, tie, win)
		# Decay elegibility trace and update for
			previous state
		z[player] <- gamma*lamda*z[player] + value_gradient(S, w)
		# Create target
		delta[player] <- R + gamma*value(Sˆ, w) - value(S, w)
		# Update every other time so that the parameters are stable for one round
		if player=2:
			# Update parameters
			w <- w + alpha*delta[1]*z[1]
			w <- w + alpha*delta[2]*z[2]
		else if state is terminal:
			w <- w + alpha*delta[player]*z[player]
		If S is terminal:
			break
		S <- Sˆ
		player <- 3 - player # For hand and end board
		invert_board()
```

# Value function

The value() function will be a pytorch neural network object, with an input layer for a vector of the board and hand, two fully connected hidden layers and an output layer of a single value. The parameter update will therefore be stochastic gradient decent that takes elegibility traces into account. Additionally it would be possible to save pairs of states and their value approximations in order to squeeze more out of that data with batch learning.

The attributes will be represented by a one-dimensional array consisting of the following:
- The board, represented with a one-hot encoding of each tile
- The cards in hand, represented by a 50-element vector whose elements in turn represent the number of a given card in the current hand
- The trash pile (used cards), represented in the same way as the current hand

The reward function will return 0 for entering any state other than a terminal state. The reward for entering a terminal state will be 1 for a win and 0.5 for a draw. Note that entering a terminal state cannot result in a loss. The value for a terminal state will be around 0 (imperfections may result due to the approximate nature of the value function), and no reward will be obtained after entering a terminal state.