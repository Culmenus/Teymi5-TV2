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
		Sˆ, R <- play(S, action)
		# Decay elegibility trace and update for 
			previous state
		z[player] <- gamma*lamda*z[player] + value_gradient
		# Create target
		delta[player] <- R + gamma*value(Sˆ, w) - value(S, w)
		if player=2:
			# Update parameters
			w <- w + alpha*delta[1]*z[1]
			w <- w + alpha*delta[2]*z[2]
		S <- Sˆ
		player <- 3 - player # For hand and end board
		invert_board()
		If S is terminal:
			break
```

The value() function will be a pytorch neural network object. The parameter update will therefore be stochastic gradient decent that takes elegibility traces into account.