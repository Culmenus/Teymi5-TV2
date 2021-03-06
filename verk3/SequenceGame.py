from PIL.Image import new
import numpy as np
import matplotlib.pyplot as plt

class SequenceEnv:

    def __init__(self, num_players = 2):
        # some global variables used by the games, what we get in the box!
        self.num_players = num_players
        self.no_feasible_move = 0  # counts how many player in a row say pass!
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
                     '1J', '2J']

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
        
        self.initialize_game() # So that all variables get initialized
        self.set_attributes()


        # Some linear function approximators
        # Can be changed to neural networks
        self.value_weights = np.zeros(self.attributes[0].size)
        self.policy_weights = np.zeros(self.attributes[0].size)

    def initialize_game(self):
        self.gameover = False
        self.player = 1
        self.discs_on_board = np.zeros((10, 10), dtype='int8')
        self.sequences = [0]*self.num_players
        self.sequence_discs = set()
        self.discs_on_board[np.ix_([0, 0, 9, 9], [0, 9, 0, 9])] = -1
        self.deck = self.cards[np.argsort(np.random.rand(104))]
        self.hand = []
        for i in range(self.num_players):
            self.hand.append(self.deck[:self.m[self.num_players]])  # deal player i m[n] cards
            self.deck = self.deck[self.m[self.num_players]:]  # remove cards from deck
        
        self.set_attributes()

    def is_terminal(self, i, j):
        # Checks whether the board is in a terminal state, given that
        # the last move was played in position (i, j)
        # Additionally updates a set containing the positions of all
        # discs which are part of a sequence.
        p = self.player - 1

        temp_board = self.discs_on_board.copy()
        temp_board[0,0], temp_board[0,9], temp_board[9,0], temp_board[9,9] = [self.player]*4
        
        # Horizontal
        t1 = 1
        while j + t1 < 10 and temp_board[i,j+t1] == self.player:
            t1 += 1
        t1 -= 1
        t2 = 1
        while j - t2 >= 0 and temp_board[i,j-t2] == self.player:
            t2 += 1
        t2 -= 1
        t = t1 + t2
        if t >= 4:
            self.sequence_discs.add((i, j))
            if t >= 8:
                return True
            if t1 < 5 and t2 < 5:
                self.sequences[p] += 1
            if t2 < 5:
                for k in range(1, t1+1):
                    self.sequence_discs.add((i, j+k))
            if t1 < 5:
                for k in range(1, t2+1):
                    self.sequence_discs.add((i, j-k))

        # Vertical
        t1 = 1
        while i + t1 < 10 and temp_board[i+t1,j] == self.player:
            t1 += 1
        t1 -= 1
        t2 = 1
        while i - t2 >= 0 and temp_board[i-t2,j] == self.player:
            t2 += 1
        t2 -= 1
        t = t1 + t2
        if t >= 4:
            self.sequence_discs.add((i, j))
            if t >= 8:
                return True
            if t1 < 5 and t2 < 5:
                self.sequences[p] += 1
            if t2 < 5:
                for k in range(1, t1+1):
                    self.sequence_discs.add((i+k, j))
            if t1 < 5:
                for k in range(1, t2+1):
                    self.sequence_discs.add((i-k, j))

        # Main diagonal
        t1 = 1
        while i + t1 < 10 and j + t1 < 10 and temp_board[i+t1,j+t1] == self.player:
            t1 += 1
        t1 -= 1
        t2 = 1
        while i - t2 >= 0 and j - t2 >= 0 and temp_board[i-t2,j-t2] == self.player:
            t2 += 1
        t2 -= 1
        t = t1 + t2
        if t >= 4:
            self.sequence_discs.add((i, j))
            if t >= 8:
                return True
            if t1 < 5 and t2 < 5:
                self.sequences[p] += 1
            if t2 < 5:
                for k in range(1, t1+1):
                    self.sequence_discs.add((i+k, j+k))
            if t1 < 5:
                for k in range(1, t2+1):
                    self.sequence_discs.add((i-k, j-k))

        # Antidiagonal
        t1 = 1
        while i + t1 < 10 and j - t1 >= 0 and temp_board[i+t1,j-t1] == self.player:
            t1 += 1
        t1 -= 1
        t2 = 1
        while i - t2 >= 0 and j + t2 < 10 and temp_board[i-t2,j+t2] == self.player:
            t2 += 1
        t2 -= 1
        t = t1 + t2
        if t >= 4:
            self.sequence_discs.add((i, j))
            if t >= 8:
                return True
            if t1 < 5 and t2 < 5:
                self.sequences[p] += 1
            if t2 < 5:
                for k in range(1, t1+1):
                    self.sequence_discs.add((i+k, j-k))
            if t1 < 5:
                for k in range(1, t2+1):
                    self.sequence_discs.add((i-k, j+k))

        if self.num_players == 2:
            return self.sequences[p] > 1
        else:
            return self.sequences[p] > 0

    # (tpr@hi.is)
    def draw_card(self, card_played, debug=False):
        player_hand = self.hand[self.player-1]
        # remove card played from hand
        if len(self.deck) > 0:
            new_card = self.deck[0]  # take top card from the deck
            self.deck = self.deck[1:]  # remove the card from the deck
        else:
            new_card = -1  # A nonexistent card; represented as an empty slot
        index = np.where(player_hand == card_played)[0][0]  # find location of card played in hand
        self.hand[self.player-1][index] = new_card  # replace the card played with a new one
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
        self.update_attributes(pos, old_card=card, new_card=self.hand[p][card_index])
        policy = np.dot(self.attributes[p], self.policy_weights)
        value = self.get_value(p)

        # Reset state
        self.hand[p] = old_hand
        self.discs_on_board[i,j] = old_disc
        self.attributes = old_attributes
        return policy, value

    def get_moves(self, debug=False):
        # legal moves for normal playing cards
        iH = np.in1d(self.cards_on_board, self.hand[self.player - 1]).reshape(10, 10)  # check for cards in hand
        iA = (self.discs_on_board == 0) # there is no disc blocking
        legal_moves = np.argwhere(iH & iA)
        # legal moves for one-eyed Jacks (they remove)
        # if 48 in self.hand[self.player-1]:
        #     legal_moves_1J = np.argwhere((self.discs_on_board != -1) & (self.discs_on_board != 0) & (self.discs_on_board != self.player))
        if 48 in self.hand[self.player-1]:
            legal_moves_1J = []
            temp_legal_moves_1J = np.argwhere((self.discs_on_board != -1) & (self.discs_on_board != 0) & (self.discs_on_board != self.player))
            if self.num_players == 2:
                for i in temp_legal_moves_1J:
                    if tuple(i) not in self.sequence_discs:
                        legal_moves_1J.append(i)
            legal_moves_1J = np.array(legal_moves_1J)
            if legal_moves_1J.size == 0:
                legal_moves_1J = np.array([]).reshape(0, 2)
        else:
            legal_moves_1J = np.array([]).reshape(0, 2)
        # legal moves for two-eyed Jacks (they are wild)
        if 49 in self.hand[self.player-1]:
            legal_moves_2J = np.argwhere(self.discs_on_board == 0)
        else:
            legal_moves_2J = np.array([]).reshape(0, 2)
        if debug:
            print("legal_moves for player", self.player)
            for i, j in legal_moves:
                print(self.the_cards[self.cards_on_board[i, j]], end=" ")
            print()
        return legal_moves, legal_moves_1J, legal_moves_2J
    
    def make_move(self, policy, epsilon=0.1, debug=False):
        legal_moves, legal_moves_1J, legal_moves_2J = self.get_moves()
        len1 = len(legal_moves)
        len2 = len1 + len(legal_moves_1J)
        all_moves = np.concatenate((legal_moves, legal_moves_1J, legal_moves_2J)).astype(np.int8)
        played_card = 0
        disc = -1
        i, j = 0, 0
        p = self.player - 1
        if len(all_moves) > 0:
            self.no_feasible_move = 0
            k = 0
            randomMove = False
            if policy[p] == "epsilon_greedy":
                cmp = np.random.rand()
                if cmp < epsilon:
                    randomMove = True
            if policy[p] == "random" or randomMove:
                k = np.random.choice(np.arange(len(all_moves)), 1)[0]
            elif policy[p] == "parametrized" or policy[p] == "epsilon_greedy":
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
                if policy[p] == "epsilon_greedy":
                    k = np.argmax(policy_estimates)
                elif policy[p] == "parametrized":
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

            # Update board, hand, and attributes
            self.discs_on_board[i,j] = disc
            new_card = self.draw_card(played_card)

            self.update_attributes((i,j), old_card=played_card, new_card=new_card)
        else:
            self.no_feasible_move += 1
            if self.no_feasible_move == self.num_players:
                self.gameover = True
        if disc > 0 and self.is_terminal(i, j):
            if debug:
                # B??tti vi?? a?? ??a?? prentar ??t hnitin ?? s????asta spili sem var spila??. L??ttara a?? finna hvar leikma??urinn vann.
                print("no_feasible_move =", self.no_feasible_move, " player =", self.player, " cards in deck =", len(self.deck),
                      " last played card at coords: (", i, j, ")", "sequences:", self.sequences, "sequence discs:", self.sequence_discs)
            self.gameover = True
            return

        current_player = self.player
        self.player = current_player % self.num_players + 1

    def set_attributes(self):
        temp_board = np.zeros(96, dtype=np.int8)

        attributes = []
        for i in range(self.num_players):
            # One hot encoding ?? bor??i
            one_hot_board = np.zeros((temp_board.size, self.num_players + 1))
            one_hot_board[np.arange(temp_board.size), temp_board] = 1
            one_hot_board = one_hot_board.flatten()
        
            # H??nd
            hond = np.zeros(50, dtype=np.uint8)
            np.add.at(hond, self.hand[i], 1)
            attributes.append(np.concatenate((one_hot_board, hond)))
    
        self.attributes = np.array(attributes)

    def update_attributes(self, pos, old_card, new_card=None):
        p = self.player - 1
        n = self.num_players + 1

        # Uppf??ra eftir leik; for??ast ????arfa ??treikninga
        # Mj??g ???optimised???; ekki ????gilegt a?? vinna me??
        # Pos: tuple (i, j); ??ar sem s????asti leikma??ur l??k
        i, j = pos

        # Sta??ur ?? eigindavigri; nau??synlegt a?? taka tillit til hornanna
        attr_pos = n * (10*i + j - 1)
        if i > 8:
            attr_pos -= 2 * n
        elif i > 0:
            attr_pos -= n

        # Uppf??ra ??ann sta??
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

        # Uppf??ra h??nd
        self.attributes[p][n*96+old_card] -= 1
        if new_card != -1:
            self.attributes[p][n*96+new_card] += 1

    def get_value(self, p):
        # State-value function
        # Currently linear, can be changed to a neural network
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
            
    def show_discs(self):
        plt.imshow(self.discs_on_board)
        plt.colorbar()
        plt.show()

    #naive test
    def play_full_game(self, policy="random", verbose=True):
        if isinstance(policy, str):
            policy = tuple([policy]*self.num_players)
        self.initialize_game()
        while not self.gameover:
            self.make_move(policy=policy, debug=verbose)
            if verbose:
                self.show_discs()
    
    def learn(self, policy="parametrized", alpha_w=0.001, alpha_theta=0.001, episodes=1000, verbose=True):
        # Implements One-step Actor-Critic

        if isinstance(policy, str):
            policy = tuple([policy]*self.num_players)
        wins = [0]*self.num_players
        if verbose:
            indices = [(i * episodes) //  20 for i in range(20)]
            print("[", end="")
        for i in range(episodes):
            self.initialize_game()
            
            self.set_attributes()
            while not self.gameover:
                p = self.player-1
                old_value = self.get_value(p)
                self.make_move(policy=policy, debug=False)
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
                ss = np.sum(np.square(self.value_weights))
                if ss > 1:
                    self.value_weights /= ss
                ss = np.sum(np.square(self.policy_weights))
                if ss > 1:
                    self.policy_weights /= ss
            wins[p] += 1
            if verbose:
                while len(indices) > 0 and i == indices[0]:
                    print('=', end='')
                    del(indices[0])
        if verbose:
            print("] Finished {} episodes".format(episodes))
        return wins