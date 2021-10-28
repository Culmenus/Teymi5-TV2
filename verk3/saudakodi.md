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
