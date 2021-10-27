# Hugmynd:

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
