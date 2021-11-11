
# Actor-critic

```
For episodes:
	S, hands <- Initialize board and hands
	z <- Initialize elegibility trace for each player and both parameter sets, theta and w (policy and value approximator)
	player <- 1
	delta <- [0, 0]
	I <- 1
	For steps in episode:
		if S is terminal:
			#value is 0
			# ...
			break
		moves <- getMoves(S, hands[player])
		action <- policy(moves)
		# Observation of new state and reward
		Sˆ, R <- play(S, action) # R \in (0, 0.5, 1) in terminal states for (loss, tie, win)
		# Decay elegibility trace and update for previous state
		z[player][w] <- gamma*lambda_w*z[player] + value_gradient(S, w)
		z[player][theta] <- gamma*lambda*z[player][theta] + I*ln(policy_gradient(action, S, theta)
		# Create target
		delta[player] <- R + gamma*value(Sˆ, w) - value(S, w)
		# Update every other time so that the parameters are stable for one round
		if player=2 or S is terminal:
			# Update parameters
			w <- w + alpha_w*delta[1]*z[1][w]
			w <- w + alpha_w*delta[2]*z[2][w]
			theta <- theta + alpha_theta*delta[1]*z[1][theta]
			theta <- theta + alpha_theta*delta[2]*z[2][theta]
		I <- gamma*I
		S <- Sˆ
		player <- 3 - player # For hand and end board. Only 2 players.
		invert_board() # Erum við búin að hætta við þetta?
```

# Value function

The value() function will be a simple pytorch neural network object, with an input layer for a vector of the board and hand, two fully connected hidden layers and an output layer of a single value. The parameter update will therefore be stochastic gradient decent that takes elegibility traces into account. 
The policy function will be a supplementary last layer of the same NN that constitutes the value function. It will be linear weights from the second to last layer with an added softmax function.

The attributes will be represented by a one-dimensional array consisting of the following:
- The board, represented with a one-hot encoding of each tile
- The cards in hand, represented by a 50-element vector whose elements in turn represent the number of a given card in the current hand
- The trash pile (used cards), represented in the same way as the current hand

The reward function will return 0 for entering any state other than a terminal state. The reward for entering a terminal state will be 1 for a win and 0.5 for a draw. Note that entering a terminal state cannot result in a loss. The value for a terminal state will be around 0 (imperfections may result due to the approximate nature of the value function), and no reward will be obtained after entering a terminal state.
