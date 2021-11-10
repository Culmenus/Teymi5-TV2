import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from numpy.core.fromnumeric import size

class SequenceEnv:

    def __init__(self, num_players = 2):
        # some global variables used by the games, what we get in the box!
        self.num_players = num_players
        self.no_feasible_move = 0  # counts how many player in a row say pass! FINNST ÞETTA FURÐULEG BREYTA.
        # There are two decks of cards each with 48 unique cards if we remove the Jacks lets label them 0,...,47
        # Let card 48 be one-eyed Jack and card 49 be two-eyed jack; there are 4 each of these
        self.cards = np.hstack((np.arange(48), np.arange(48), 48, 48, 48, 48, 49, 49, 49, 49))
        # now lets deal out the hand, each player gets m[n] cards
        self.m = (None, None, 7, 6, 6)

        self.cards_on_board = np.matrix([[-1, 0, 11, 10, 9, 8, 7, 6, 5, -1],
                                    [24, 18, 19, 20, 21, 22, 23, 12, 4, 13],
                                    [35, 17, 9, 8, 7, 6, 5, 25, 3, 14],
                                    [34, 16, 10, 43, 42, 41, 4, 26, 2, 15],
                                    [33, 15, 11, 44, 37, 40, 3, 27, 1, 16],
                                    [32, 14, 0, 45, 38, 39, 2, 28, 36, 17],
                                    [31, 13, 24, 46, 47, 36, 1, 29, 47, 18],
                                    [30, 37, 35, 34, 33, 32, 31, 30, 46, 19],
                                    [29, 38, 39, 40, 41, 42, 43, 44, 45, 20],
                                    [-1, 28, 27, 26, 25, 12, 23, 22, 21, -1]])
        self.the_cards = ['AC', '2C', '3C', '4C', '5C', '6C', '7C', '8C', '9C', '1C', 'QC', 'KC',
                     'AS', '2S', '3S', '4S', '5S', '6S', '7S', '8S', '9S', '1S', 'QS', 'KS',
                     'AD', '2D', '3D', '4D', '5D', '6D', '7D', '8D', '9D', '1D', 'QD', 'KD',
                     'AH', '2H', '3H', '4H', '5H', '6H', '7H', '8H', '9H', '1H', 'QH', 'KH',
                     '1J', '1J', '1J', '1J', '2J', '2J', '2J', '2J']

        self.attributes = []
        self.heuristic_1_table = np.zeros((num_players, 10, 10))

        # Lookup table fyrir spilin
        self.card_positions = {}
        for i in range(48):
            self.card_positions[i] = []
        for i in range(10):
            for j in range(10):
                if self.cards_on_board[i,j] != -1:
                    self.card_positions[self.cards_on_board[i,j]].append((i, j))
        for i in range(12):
            self.card_positions[i] = tuple(self.card_positions[i])

        # Some linear function approximators
        # Can be changed to neural networks
        self.value_weights = np.zeros(self.attributes[0].size)
        self.policy_weights = np.random.normal(size=self.attributes[0].size)

    def initialize_game(self):
        self.gameover = False
        self.player = 1
        self.discs_on_board = np.zeros((10, 10), dtype='int8')
        self.sequences = [0]*self.num_players
        self.discs_on_board[np.ix_([0, 0, 9, 9], [0, 9, 0, 9])] = -1
        self.deck = self.cards[np.argsort(np.random.rand(104))]
        self.hand = []
        for i in range(self.num_players):
            self.hand.append(self.deck[:self.m[self.num_players]])  # deal player i m[n] cards
            self.deck = self.deck[self.m[self.num_players]:]  # remove cards from deck
        
        self.set_attributes()

    # Function to check if the "reitur" where the one eyed jack is being played is a part of 5 in a row
    # If yes the return true ("not allowed")
    # Else return false ("allowed")
    # Takes in where the jack is being played ,
    # state of the board (disc on board),
    # which player is playing it and nr of players (n).
    def fiveInRow(self, row, col):
        tempWin = [[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]]
        other_player = self.player % self.num_players + 1
        players_discs = self.discs_on_board.copy()
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
        if len(temp_bdiag2) < 10:
            temp_bdiag2 = temp_bdiag2 + ([False]*(10 - len(temp_bdiag2)))
        if len(temp_fdiag2) < 10:
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

    def isTerminal(self, pos):
        i, j = pos
        p = self.player - 1

        temp_board = self.discs_on_board.copy()
        temp_board[0,0], temp_board[0,9], temp_board[9,0], temp_board[9,9] = [self.player]*4
        
        lengths = []
        # Horizontal
        t = 1
        while j + t < 10 and temp_board[i,j+t] == self.player:
            t += 1
        lengths.append(t - 1)
        t = 1
        while j - t > 0 and temp_board[i,j-t] == self.player:
            t += 1
        lengths.append(t - 1)

        # Vertical
        t = 1
        while i + t < 10 and temp_board[i+t,j] == self.player:
            t += 1
        lengths.append(t - 1)
        t = 1
        while i - t > 0 and temp_board[i-t,j] == self.player:
            t += 1
        lengths.append(t - 1)

        # Main diagonal
        t = 1
        while i + t < 10 and j + t < 10 and temp_board[i+t,j+t] == self.player:
            t += 1
        lengths.append(t - 1)
        t = 1
        while i - t > 0 and j - t > 0 and temp_board[i-t,j-t] == self.player:
            t += 1
        lengths.append(t - 1)

        # Antidiagonal
        t = 1
        while i + t < 10 and j - t > 0 and temp_board[i+t,j-t] == self.player:
            t += 1
        lengths.append(t - 1)
        t = 1
        while i - t > 0 and j + t < 10 and temp_board[i-t,j+t] == self.player:
            t += 1
        lengths.append(t - 1)

        for k in range(4):
            if lengths[2*k] + lengths[2*k+1] >= 8:
                return True

        for k in range(4):
            if (lengths[2*k] + lengths[2*k+1] >= 4) and not (lengths[2*k] >= 5 or lengths[2*k+1] >= 5):
                self.sequences[p] += 1

        if self.num_players == 2:
            return self.sequences[p] > 1
        else:
            return self.sequences[p] > 0

    # (tpr@hi.is)
    def drawCard(self, card_played, debug=False):
        player_hand = self.hand[self.player-1]
        # remove card played from hand
        if len(self.deck) > 0:
            new_card = self.deck[0]  # take top card from the deck
            self.deck = self.deck[1:]  # remove the card from the deck
        else:
            new_card = -1  # A nonexistent card; represented as an empty slot
        i = np.where(player_hand == card_played)  # find location of card played in hand
        if debug:
            print("Hand before change", self.hand)
        if len(i) > 0:
            self.hand[self.player-1][i[0][0]] = new_card  # replace the card played with a new one
        else:
            print("drawCard, could not find this cards in the current hand?!")
            raise
        if debug:
            print("Hand after change", self.hand)
        return new_card

    def sample_card(self):
        # Samples a card draw
        # Adds some noise to the afterstate calculations,
        # but can utilize card counting
        # TODO: Laga
        if self.deck.size == 0:
            return -1
        else:
            return np.random.choice(self.deck)

    def lookahead(self, pos, card, disc):
        # One-step lookahead; finds afterstate value
        p = self.player - 1
        i, j = pos

        # Cache current state
        old_disc = self.discs_on_board[i,j]
        old_hand = self.hand[p].copy()
        old_attributes = self.attributes.copy()

        # Update state and find value
        self.discs_on_board[i,j] = disc
        card_index = np.where(self.hand[p] == card)[0][0]
        self.hand[p][card_index] = self.sample_card()
        self.set_attributes(pos=pos, old_card=card, new_card=self.hand[p][card_index])
        policy = np.dot(self.attributes[p], self.policy_weights)
        value = self.get_value(p)

        # Reset state
        self.hand[p] = old_hand
        self.discs_on_board[i,j] = old_disc
        self.attributes = old_attributes
        return policy, value

    def getMoves(self, debug=False):
        # legal moves for normal playing cards
        iH = np.in1d(self.cards_on_board, self.hand[self.player - 1]).reshape(10, 10)  # check for cards in hand
        iA = (self.discs_on_board == 0) # there is no disc blocking
        legal_moves = np.argwhere(iH & iA)
        # legal moves for one-eyed Jacks (they remove)
        if 48 in self.hand[self.player-1]:
            legal_moves_1J = np.argwhere((self.discs_on_board != -1) & (self.discs_on_board != 0) & (self.discs_on_board != self.player))
        else:
            legal_moves_1J = np.array([]).reshape(0, 2)
        # legal moves for two-eyed Jacks (they are wild)
        if 49 in self.hand[self.player-1]:
            legal_moves_2J = np.argwhere(self.discs_on_board == 0)
        else:
            legal_moves_2J = np.array([]).reshape(0, 2)
        if debug:
            print("legal_moves for player ", self.player)
            for i, j in legal_moves:
                print(self.the_cards[self.cards_on_board[i, j]], end=" ")
            print("")
        return legal_moves, legal_moves_1J, legal_moves_2J
    
    def makeMove(self, policy="random", epsilon=0.1, debug=False):
        legal_moves, legal_moves_1J, legal_moves_2J = self.getMoves()
        len1 = len(legal_moves)
        len2 = len1 + len(legal_moves_1J)
        all_moves = np.concatenate((legal_moves, legal_moves_1J, legal_moves_2J)).astype(np.int8)
        played_card = 0
        disc = -1
        i, j = 0, 0
        if len(all_moves) > 0:
            k = 0
            randomMove = False
            if policy == "epsilon_greedy":
                cmp = np.random.rand()
                if cmp < epsilon:
                    randomMove = True
            if policy == "random" or randomMove:
                k = np.random.choice(np.arange(len(all_moves)), 1)
            elif policy == "parametrized" or policy == "epsilon_greedy":
                # Find afterstate values
                policy_estimates = []
                values = []
                for i in range(len(legal_moves)):
                    x, y = legal_moves[i]
                    pol, val = self.lookahead(legal_moves[i], self.cards_on_board[x,y], self.player)
                    policy_estimates.append(pol)
                    values.append(val)
                for i in range(len(legal_moves_1J)):
                    pol, val = self.lookahead(legal_moves_1J[i], 48, 0)
                    policy_estimates.append(pol)
                    values.append(val)
                for i in range(len(legal_moves_2J)):
                    pol, val = self.lookahead(legal_moves_2J[i], 49, self.player)
                    policy_estimates.append(pol)
                    values.append(val)
                policy_estimates = np.array(policy_estimates)
                if policy == "epsilon_greedy":
                    k = np.argmax(policy_estimates)
                elif policy == "parametrized":
                    # Linear softmax policy
                    exp = np.exp(policy_estimates)
                    probabilities = exp / np.sum(exp)
                    k = np.random.choice(np.arange(len(probabilities)), p=probabilities)
            i, j = all_moves[k]
            if k < len1 or k >= len2:
                disc = self.player
            else:
                disc = 0
            if k < len1:
                played_card = self.cards_on_board[i,j]
            elif k < len2:
                played_card = 48
            else:
                played_card = 49
        else:
            disc = -1
            self.no_feasible_move += 1
        if disc >= 0:
            self.no_feasible_move = 0

            # Update board, hand, and attributes
            self.discs_on_board[i,j] = disc
            new_card = self.drawCard(played_card)
            self.set_attributes(pos=(i,j), old_card=played_card, new_card=new_card)
        if self.no_feasible_move == self.num_players or self.isTerminal(pos=(i,j)):
            if debug:
                # Bætti við að það prentar út hnitin á síðasta spili sem var spilað. Léttara að finna hvar leikmaðurinn vann.
                print("no_feasible_move =", self.no_feasible_move, " player =", self.player, " cards in deck =", len(self.deck),
                      " last played card at coords: (", i, j, ")", "sequences:", self.sequences)
            self.gameover = True

        current_player = self.player
        self.player = current_player % self.num_players + 1

    def heuristic_1(self, temp_board, pos):
        # Namminamm
        # temp_board: discs_on_board, nema player í stað -1 í hornunum

        s = 0
        i = pos[0]
        j = pos[1]
        board_rows = 10
        board_cols = 10

        # Horizontal
        mint = max(-4, -j)
        maxt = min(4, board_cols-1-j)
        t = mint
        while t < 0:
            if temp_board[i][j+t] != self.player and temp_board[i][j+t] != 0:
                mint = t+1
        t = maxt
        while t > 0:
            if temp_board[i][j+t] != self.player and temp_board[i][j+t] != 0:
                maxt = t-1

        range = maxt - mint + 1
        if range >= 5:
            t = mint
            while t <= maxt:
                if temp_board[i][j+t] == self.player:
                    s += min(min(t - mint, maxt - t), range - 5) + 1
            t -= 1

        # Vertical
        mint = max(-4, -i)
        maxt = min(4, board_rows-1-i)
        t = mint
        while t < 0:
            if temp_board[i+t][j] != self.player and temp_board[i+t][j] != 0:
                mint = t+1
        t = maxt
        while t > 0:
            if temp_board[i+t][j] != self.player and temp_board[i+t][j] != 0:
                maxt = t-1

        range = maxt - mint + 1
        if range >= 5:
            t = mint
            while t <= maxt:
                if temp_board[i+t][j] == self.player:
                    s += min(min(t - mint, maxt - t), range - 5) + 1
                    t -= 1
        
        # Main diagonal
        mint = max(-4, max(-i, -j))
        maxt = min(4, min(board_rows-1-i, board_rows-1-j))
        t = mint
        while t < 0:
            if temp_board[i+t][j+t] != self.player and temp_board[i+t][j+t] != 0:
                mint = t+1
        t = maxt
        while t > 0:
            if temp_board[i+t][j+t] != self.player and temp_board[i+t][j+t] != 0:
                maxt = t-1

        range = maxt - mint + 1
        if range >= 5:
            t = mint
            while t <= maxt:
                if temp_board[i+t][j+t] == self.player:
                    s += min(min(t - mint, maxt - t), range - 5) + 1
                    t -= 1
        
        # Antidiagonal
        mint = max(-4, max(-i, board_rows-1-j))
        maxt = min(4, min(board_rows-1-i, -j))
        t = mint
        while t < 0:
            if temp_board[i+t][j-t] != self.player and temp_board[i+t][j-t] != 0:
                mint = t+1
        t = maxt
        while t > 0:
            if temp_board[i+t][j-t] != self.player and temp_board[i+t][j-t] != 0:
                maxt = t-1

        range = maxt - mint + 1
        if range >= 5:
            t = mint
            while t <= maxt:
                if temp_board[i+t][j-t] == self.player:
                    s += min(min(t - mint, maxt - t), range - 5) + 1
                    t -= 1

    def update_heuristic_1(self, pos):
        # Uppfæra Heuristic 1 eftir leik

        i, j = pos

        temp_board = self.discs_on_board.copy()
        temp_board[temp_board == -1] = self.player

        for p in range(self.num_players):
            if p+1 == self.player:
                self.heuristic_1_table[p][i][j] = self.heuristic_1(temp_board, i, j)
            else:
                self.heuristic_1_table[p][i][j] = 0

        # Horizontal
        for p in range(self.num_players):
            for t in range(1, 5):
                if temp_board[i+t][j] != 0 and temp_board[i+t][j] != p:
                    break
                self.heuristic_1_table[self.player-1][i+t][j] = self.heuristic_1(temp_board, i+t, j)
            for t in range(1, 5):
                if temp_board[i-t][j] != 0 and temp_board[i-t][j] != p:
                    break
                self.heuristic_1_table[self.player-1][i-t][j] = self.heuristic_1(temp_board, i-t, j)
        
        # TODO: Vertical & diagonal
    
    def set_attributes(self, pos=None, old_card=None, new_card=None):
        p = self.player - 1
        n = self.num_players + 1

        if pos is not None:
            # Uppfæra eftir leik; forðast óþarfa útreikninga
            # Mjög „optimised“; ekki þægilegt að vinna með
            # Pos: tuple (i, j); þar sem síðasti leikmaður lék
            i, j = pos

            # Staður í eigindavigri; nauðsynlegt að taka tillit til hornanna
            attr_pos = n * (10*i + j - 1)
            if i > 8:
                attr_pos -= 2 * n
            elif i > 0:
                attr_pos -= n

            # Uppfæra þann stað
            disc = self.discs_on_board[i,j]
            if disc == 0:
                for k in range(self.num_players):
                    new_attr = np.zeros(n)
                    self.attributes[k][attr_pos:attr_pos+n] = new_attr
            else:
                for k in range(self.num_players):
                    new_attr = np.zeros(n)
                    new_attr[disc] = 1
                    self.attributes[k][attr_pos:attr_pos+n] = new_attr
                    disc -= 1
                    if disc == 0:
                        disc = self.num_players

            # Uppfæra hönd
            self.attributes[p][n*96+old_card] -= 1
            if new_card is not None:
                self.attributes[p][n*96+new_card] += 1

        else:
            temp_board = self.discs_on_board.copy().flatten()
            temp_board = temp_board[temp_board != -1]

            attributes = []
            for i in range(self.num_players):
                # One hot encoding á borði
                one_hot_board = np.zeros((temp_board.size, n))
                one_hot_board[np.arange(temp_board.size),temp_board] = 1
                one_hot_board = one_hot_board.flatten()
            
                # Hönd
                hond = np.zeros(50, dtype=np.uint8)
                np.add.at(hond, self.hand[i], 1)
                attributes.append(np.concatenate((one_hot_board, hond)))

                # Sjónarhorn næsta leikmanns
                temp_board[temp_board != 0] += 1
                temp_board[temp_board == n] = 1
        
            self.attributes = np.array(attributes)

        """
        # ELÍAS
        temp_board = self.discs_on_board.copy().flatten()
        # ATH. Segi að það sé enginn leikmaður á hornunum. Þetta þarf að endurskoða, kannski sleppa.
        temp_board = temp_board[temp_board != -1]
        # One hot encoding á borði
        one_hot = torch.nn.functional.one_hot(torch.tensor(temp_board).long(), num_classes = n + 1)
        # Flet borðið út. Hér væri líka hægt að halda forminu og breyta kóðun á hönd
        one_hot = one_hot.flatten()
        # Kóða höndina eins og talað var um
        hond = np.zeros((50), dtype = np.int32)
        np.add.at(hond, hand, 1)
        # Skeyti hönd aftan við borð, fjöldi spila á indexi. Ef sett er hand í stað hönd skeytist nr. spila við, mögulega jafngóð kóðun
        return torch.cat((one_hot, torch.tensor(hond)))
        """

    def get_value(self, p):
        # State-value function
        # Currently linear, can be changed to a neural network
        #print(self.attributes[p])
        #print(self.value_weights)
        return np.dot(self.attributes[p], self.value_weights)

    # printing the board is useful for debugging code...
    def pretty_print(self):
        color = ["", "*", "*", "*", "*"]
        for i in range(10):
            for j in range(10):
                if (self.discs_on_board[i,j] <= 0):
                    if self.cards_on_board[i, j] >= 0:
                        print(self.the_cards[self.cards_on_board[i, j]], end=" ")
                    else:
                        print("-1", end=" ")
                else:
                    print(color[self.discs_on_board[i,j]] + str(self.discs_on_board[i,j]), end=" ")
            print("")
        for i in range(len(self.hand)):
            print("player ", i + 1, "'s hand: ", [self.the_cards[j] for j in self.hand[i]], sep="")

    #naive test
    def play_full_game(self):
        self.initialize_game()
        while not self.gameover:
            self.makeMove(debug=True)
            print(self.discs_on_board[0])
            plt.imshow(self.discs_on_board[0])
            plt.colorbar()
            plt.show()
    
    def learn(self, policy="parametrized", episodes=1000, verbose=True):
        self.initialize_game()

        # Implements One-step Actor-Critic

        # Learning rates for value function and policy, respectively
        alpha_w = 0.01
        alpha_theta = 0.01

        if verbose:
            indices = [(i * episodes) //  20 for i in range(20)]
            print('[', end='')
        for i in range(episodes):
            self.initialize_game()
            while not self.gameover:
                p = self.player-1
                old_value = self.get_value(p)
                self.makeMove(policy=policy, debug=False)
                new_value = self.get_value(p)
                delta = 0
                if self.gameover:
                    if self.no_feasible_move:
                        delta = 0.5 - old_value
                    else:
                        delta = 1 - old_value
                else:
                    delta = new_value - old_value
                self.value_weights += alpha_w * delta * self.attributes[p]
                self.policy_weights += alpha_theta * delta * self.attributes[p]
            if verbose:
                while len(indices) > 0 and i == indices[0]:
                    print('=', end='')
                    del(indices[0])
        if verbose:
            print('] Finished {} episodes'.format(episodes))