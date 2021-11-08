import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from numpy.core.fromnumeric import size

class SequenceEnv:

    def __init__(self, num_players = 2):
        # some global variables used by the games, what we get in the box!
        self.num_players = num_players
        self.player = 1
        self.discs_on_board = np.zeros((num_players, 10, 10), dtype='int8')  # empty!
        for i in range(num_players):
            # Set corners to -1
            self.discs_on_board[i][np.ix_([0, 0, 9, 9], [0, 9, 0, 9])] = -1
        self.no_feasible_move = 0  # counts how many player in a row say pass! FINNST ÞETTA FURÐULEG BREYTA.
        # There are two decks of cards each with 48 unique cards if we remove the Jacks lets label them 0,...,47
        # Let card 48 be one-eyed Jack and card 49 be two-eyed jack; there are 4 each of these
        self.cards = tuple(np.hstack((np.arange(48), np.arange(48), 48, 48, 48, 48, 49, 49, 49, 49)))
        self.deck = self.cards[np.argsort(np.random.rand(104))]  # here we shuffle the cards, note we cannot use shuffle (we have non-unique cards)
        # now lets deal out the hand, each player gets m[n] cards
        self.m = (None, None, 7, 6, 6)
        self.hand = []
        for i in range(num_players):
            self.hand.append(self.deck[:self.m[num_players]])  # deal player i m[n] cards
            self.deck = self.deck[self.m[num_players]:]  # remove cards from deck

        self.attributes = []

        # Some linear function approximators
        # Can be changed to neural networks
        self.value_weights = []
        self.policy_weights = []

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
        
        # Lookup table fyrir spilin
        # Confirmed bug free
        self.card_positions = {}
        for i in range(48):
            self.card_positions[i] = []
        for i in range(10):
            for j in range(10):
                if self.cards_on_board[i,j] != -1:
                    self.card_positions[self.cards_on_board[i,j]].append((i, j))
        for i in range(12):
            self.card_positions[i] = tuple(self.card_positions[i])
        
        self.gameover = False

        self.heuristic_1_table = np.zeros((num_players, 10, 10))

    def initialize_game(self):
        self.player = 1
        self.discs_on_board = np.zeros((self.num_players, 10, 10), dtype='int8')
        for i in range(self.num_players):
            # Set corners to -1
            self.discs_on_board[i][np.ix_([0, 0, 9, 9], [0, 9, 0, 9])] = -1
        self.deck = self.cards[np.argsort(np.random.rand(104))]
        self.hand = []
        for i in range(self.num_players):
            self.hand.append(self.deck[:self.m[self.num_players]])  # deal player i m[n] cards
            self.deck = self.deck[self.m[self.num_players]:]  # remove cards from deck

    # (floki@hi.is) #moddað í hlutbundið af oat
    def isTerminal(self):
        tempWin = [[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                   [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                   [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                   [0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
                   [0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
                   [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]]
        extraWin = [[0,1,1,1,1,1,1,1,1,1],
                    [1,1,1,1,1,1,1,1,1,0],
                    [1,1,1,1,1,1,1,1,1,1]]
        temp_board = self.discs_on_board[0].copy()
        temp_board[temp_board == -1] = self.player
        test = temp_board == self.player
        # test[row][col] = 1
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
                fdiag[x + y].append(test[y][x])
                bdiag[x - y - min_bdiag].append(test[y][x])
        lists = cols + rows + bdiag + fdiag

        np_lists = np.array(lists, dtype=object)

        filt = []
        for i in range(0, len(lists)):
            filt_i = len(np_lists[i]) >= 5 and sum(np_lists[i]) >= 5
            filt.append(filt_i)

        new_list = list(np_lists[filt])
        temp = []
        for i in new_list:
            if (len(i) < 10):
                i = i + ([0.0] * (10 - len(i)))
            temp.append(i)
        if (len(temp) == 0):
            return False
        player_sum = 0
        for i in temp:
            temp_sum = 0
            for j in tempWin:
                temp_sum = sum(np.multiply(i, j))
                if temp_sum >= 5:
                    if self.num_players == 2:
                        player_sum = player_sum + temp_sum
                        extra_sum0 = sum(np.multiply(extraWin[0], i))
                        extra_sum1 = sum(np.multiply(extraWin[1], i))
                        extra_sum2 = sum(np.multiply(extraWin[2], i))
                        if player_sum >= 10 or extra_sum0 >= 9 or extra_sum1 >= 9 or extra_sum2 >= 10:
                            return True
                        break
                    else:
                        return True          
        return False

    # (tpr@hi.is)
    def drawCard(self, card_played, debug=False):
        player_hand = self.hand[self.player-1]
        # remove card player from hand
        if len(self.deck) > 0:
            new_card = self.deck[0]  # take top card from the deck
            self.deck = self.deck[1:]  # remove the card from the deck
            #print("drawCard played card",card_played)  
            #print("playerhand",player_hand)
            i = np.where(player_hand == card_played)  # find location of card played in hand
            #print(i)
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
        else:
            i = np.where(player_hand == card_played)
            
            if debug:
                print("Hand before change", self.hand)
            if len(i) > 0:
                self.hand = np.delete(self.hand[self.player-1], i[0][0])
            else:
                print("drawCard, could not find this cards in the current hand?!")
                raise
            if debug:    
                print("Hand after change", self.hand)
            return None

    def sample_card(self):
        # Samples a card draw
        # Adds some noise to the afterstate calculations,
        # but can utilize card counting
        return np.random.choice(self.deck)

    def lookahead(self, pos, card, disc):
        # One-step lookahead; finds afterstate value
        i, j = pos

        # Cache current state
        old_disc = self.discs_on_board[i,j]
        old_hand = self.hand.copy()
        old_attributes = self.attributes.copy()

        # Update state and find value
        self.discs_on_board[i,j] = disc
        card_index = self.hand.index(card)
        self.hand[card_index] = self.sample_card()
        self.set_attributes(pos=pos, card_index=card_index)
        value = self.get_value()

        # Reset state
        self.hand = old_hand
        self.discs_on_board[i,j] = old_disc
        self.attributes = old_attributes
        return value

    def getMoves(self, debug=False):
        #-------------------
        # Hvað er þetta? Virðist vera óþarfi
        # if card == 48:
        #     pass
        # if card == 49:
        #     pass
        # else:
        #     legal_moves = self.card_positions[card]
        #-------------------
        # legal moves for normal playing cards
        iH = np.in1d(self.cards_on_board, self.hand[self.player - 1]).reshape(10, 10)  # check for cards in hand
        iA = (self.discs_on_board[0] == 0) # there is no disc blocking
        legal_moves = np.argwhere(iH & iA)
        # legal moves for one-eyed Jacks (they remove)
        if 48 in self.hand[self.player-1]:
            legal_moves_1J = np.argwhere((self.discs_on_board[0] != -1) & (self.discs_on_board[0] != 0) & (self.discs_on_board[0] != self.player))
        else:
            legal_moves_1J = np.array([]).reshape(0, 2)
        # legal moves for two-eyed Jacks (they are wild)
        if 49 in self.hand[self.player-1]:
            legal_moves_2J = np.argwhere(self.discs_on_board[0] == 0)
        else:
            legal_moves_2J = np.array([]).reshape(0, 2)
        if debug:
            print("legal_moves for player ", self.player)
            for i, j in legal_moves:
                print(self.the_cards[self.cards_on_board[i, j]], end=" ")
            print("")
        return legal_moves, legal_moves_1J, legal_moves_2J
    
    def makeMove(self, policy="random", epsilon=0.1):
        # Þetta þarf að bæta, taka inn stefnu og spila eftir henni
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
                else:
                    pass # TODO: Find afterstate values
            if policy == "random" or randomMove:
                k = np.random.choice(np.arange(len(all_moves)), 1)
            else:
                # Find afterstate values
                values = []
                for i in range(len(legal_moves)):
                    values.append(self.lookahead(legal_moves[i], self.cards_on_board[legal_moves[i]], self.player))
                for i in range(len(legal_moves_1J)):
                    values.append(self.lookahead(legal_moves_1J[i], 48, 0))
                for i in range(len(legal_moves_2J)):
                    values.append(self.lookahead(legal_moves_2J[i], 49, self.player))
                values = np.array(values)
                if policy == "epsilon_greedy":
                    k = np.argmax(values)
                elif policy == "parametrized":
                    # Linear softmax policy
                    exp = np.exp(values)
                    probabilities = exp / np.sum(exp)
                    k = np.random.choice(np.arange(len(probabilities)), p=probabilities)
            i, j = all_moves[k][0] # The [0] is there to unpack a nested list [[i, j]]
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
            print("Don't have a legal move for player (can this really happen?): ", self.player)
            disc = -1
            self.no_feasible_move += 1
        if disc >= 0:
            self.no_feasible_move = 0

            # Update board, hand, and attributes
            self.discs_on_board[0,i,j] = disc
            new_card = self.drawCard(played_card)
            self.set_attributes(pos=(i,j), old_card=played_card, new_card=new_card)
        if (self.no_feasible_move == self.num_players) | (len(self.deck) == 0) | (True == self.isTerminal()):
            # Bætti við að það prentar út hnitin á síðasta spili sem var spilað. Léttara að finna hvar leikmaðurinn vann.
            print("no_feasible_move = ", self.no_feasible_move, " player = ", self.player, " cards in deck = ", len(self.deck),
                  " last played card at coords: (", i, j, ")")
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

        temp_board = self.discs_on_board[0].copy()
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
        temp_board = self.discs_on_board[0].copy().flatten()
        temp_board = temp_board[temp_board != -1] # Athuga hvort þetta sé skynsamlegt. Mikilvægt að tekið sé tillit til hornanna í is Terminal

        if pos is not None:
            # Uppfæra eftir leik; forðast óþarfa útreikninga
            # Mjög „optimised“; ekki þægilegt að vinna með
            # Pos: tuple (i, j); þar sem síðasti leikmaður lék
            i, j = pos

            # Staður í eigindavigri; nauðsynlegt að taka tillit til hornanna
            c = self.num_players + 1
            attr_pos = c * (10*i + j - 1)
            attr_pos += self.discs_on_board[0,i,j]
            if i > 8:
                attr_pos -= 2 * c
            elif i > 0:
                attr_pos -= c
            
            # Uppfæra þann stað
            new_attr = np.zeros(c)
            new_attr[self.discs_on_board[0][i,j]] = 1
            self.attributes[attr_pos:attr_pos+c] = new_attr

            # Uppfæra hönd
            self.attributes[c*96+old_card] -= 1
            self.attributes[c*96+new_card] += 1

        # One hot encoding á borði
        one_hot_board = np.zeros((temp_board.size, self.num_players+1))
        one_hot_board[np.arange(temp_board.size),temp_board] = 1
        one_hot_board = one_hot_board.flatten()
        
        # Hönd
        hond = np.zeros(50, dtype=np.uint8)
        np.add.at(hond, self.hand[self.player-1], 1)
        
        self.attributes = np.concatenate((one_hot_board, hond))

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

    def get_value(self):
        # State-value function
        # Currently linear, can be changed to a neural network
        return np.dot(self.attributes, self.value_weights)

    # printing the board is useful for debugging code...
    def pretty_print(self):
        color = ["", "*", "*", "*", "*"]
        for i in range(10):
            for j in range(10):
                if (self.discs_on_board[0,i,j] <= 0):
                    if self.cards_on_board[i, j] >= 0:
                        print(self.the_cards[self.cards_on_board[i, j]], end=" ")
                    else:
                        print("-1", end=" ")
                else:
                    print(color[self.discs_on_board[0,i,j]] + str(self.discs_on_board[0,i,j]), end=" ")
            print("")
        for i in range(len(self.hand)):
            print("player ", i + 1, "'s hand: ", [self.the_cards[j] for j in self.hand[i]], sep="")

    #naive test
    def play_full_game(self):
        self.gameover = False
        self.set_attributes()
        while not self.gameover:
            self.makeMove()
            print(self.discs_on_board[0])
            plt.imshow(self.discs_on_board[0])
            plt.colorbar()
            plt.show()
    
    def learn(self, policy="parametrized", epsilon=0.1):
        self.gameover = False
        self.set_attributes()
        self.value_weights = np.zeros(self.attributes.size)
        self.policy_weights = np.random.normal(size=self.attributes.size)
        while not self.gameover:
            self.makeMove(policy=policy, epsilon=epsilon)
