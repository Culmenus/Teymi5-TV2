# Leikmaðurinn hans Tómasar
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable

class SequenceEnv:
    def __init__(self):
        self.no_feasible_move = 0
        self.cards = np.hstack((np.arange(48), np.arange(48), 48, 48, 48, 48, 49, 49, 49, 49))

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
        self.card_positions = {}
        for i in range(48):
            self.card_positions[i] = []
        for i in range(10):
            for j in range(10):
                if self.cards_on_board[i,j] != -1:
                    self.card_positions[self.cards_on_board[i,j]].append((i, j))
        for i in range(12):
            self.card_positions[i] = tuple(self.card_positions[i])

        self.heuristic_1_table = np.zeros((2, 10, 10))
        self.heuristic_2_table = np.zeros((2, 10, 10))

        self.initialize_game(init=True)

        # NN
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.gamma = 1
        self.alpha = [0.1, 0.001, 0.001]  # step size for PG and then each layer of the neural network
        self.lmbda = 0.7
        self.nx = self.attributes[0].size
        self.nh = self.nx * 2

        self.model = [None]*4
        self.model[0] = Variable(torch.zeros((self.nh,1), device = self.device, dtype=torch.float), requires_grad = True)
        self.model[1] = Variable(0.1*torch.randn(self.nh,self.nx, device = self.device, dtype=torch.float), requires_grad = True)
        self.model[2] = Variable(torch.zeros((1,1), device = self.device, dtype=torch.float), requires_grad = True)
        self.model[3] = Variable(0.1*torch.randn(1,self.nh, device = self.device, dtype=torch.float), requires_grad = True)
        # self.model[4] = Variable(0.1*torch.randn(1,self.nh, device = self.device, dtype=torch.float), requires_grad = True)

    def load_model(self):
        return

    def initialize_game(self, init=False):
        self.player = 1
        self.discs_on_board = np.zeros((10, 10), dtype='int8')
        self.sequences = [0]*2
        self.sequence_discs = set()
        self.discs_on_board[np.ix_([0, 0, 9, 9], [0, 9, 0, 9])] = -1
        self.deck = self.cards[np.argsort(np.random.rand(104))]
        self.hand = []
        for i in range(2):
            self.hand.append(self.deck[:7])  # deal player i m[n] cards
            self.deck = self.deck[7:]  # remove cards from deck

        self.set_attributes()
        self.old_value = [None, None]

        if not init:
            self.Z_b1 = [torch.zeros(self.model[0].size(), device=self.device, dtype = torch.float) for i in range(2)]
            self.Z_w1 = [torch.zeros(self.model[1].size(), device=self.device, dtype = torch.float) for i in range(2)]
            self.Z_b2 = [torch.zeros(self.model[2].size(), device=self.device, dtype = torch.float) for i in range(2)]
            self.Z_w2 = [torch.zeros(self.model[3].size(), device=self.device, dtype = torch.float) for i in range(2)]

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

        return self.sequences[p] > 1

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

    def lookahead_attributes(self, pos, card, disc):
        # One-step lookahead; finds afterstate value
        p = self.player - 1
        i, j = pos

        # Cache current state
        old_disc = self.discs_on_board[i,j]
        old_hand = self.hand[p].copy()
        old_attributes = self.attributes.copy()

        # Update state
        self.discs_on_board[i,j] = disc
        card_index = np.where(self.hand[p] == card)[0][0]
        self.hand[p][card_index] = -1
        self.update_attributes(pos=pos, old_card=card, new_card=self.hand[p][card_index])
        attr = self.attributes[p].copy()

        # Reset state
        self.hand[p] = old_hand
        self.discs_on_board[i,j] = old_disc
        self.attributes = old_attributes
        return attr

    def learn_step(self, p, value, reward):
        self.old_value[p].backward()
        # update the eligibility traces using the gradients
        self.Z_b1[p] = self.gamma * self.lmbda * self.Z_b1[p] + self.model[0].grad.data
        self.Z_w1[p] = self.gamma * self.lmbda * self.Z_w1[p] + self.model[1].grad.data
        self.Z_b2[p] = self.gamma * self.lmbda * self.Z_b2[p] + self.model[2].grad.data
        self.Z_w2[p] = self.gamma * self.lmbda * self.Z_w2[p] + self.model[3].grad.data
        # zero the gradients
        self.model[0].grad.data.zero_()
        self.model[1].grad.data.zero_()
        self.model[2].grad.data.zero_()
        self.model[3].grad.data.zero_()
        # perform now the update for the weights
        delta = reward + self.gamma * value - self.old_value[p].detach()
        delta = torch.tensor(delta, dtype = torch.float, device=self.device)
        self.model[0].data += self.alpha[1] * delta * self.Z_b1[p]
        self.model[1].data += self.alpha[1] * delta * self.Z_w1[p]
        self.model[2].data += self.alpha[2] * delta * self.Z_b2[p]
        self.model[3].data += self.alpha[2] * delta * self.Z_w2[p]

    def get_moves(self):
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
        return legal_moves, legal_moves_1J, legal_moves_2J

    def make_random_move(self):
        legal_moves, legal_moves_1J, legal_moves_2J = self.get_moves()
        len1 = len(legal_moves)
        len2 = len1 + len(legal_moves_1J)
        all_moves = np.concatenate((legal_moves, legal_moves_1J, legal_moves_2J)).astype(np.int8)
        played_card = 0
        disc = -1
        i, j = 0, 0
        if len(all_moves) > 0:
            self.no_feasible_move = 0
            k = np.random.choice(np.arange(len(all_moves)), 1)[0]
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

            self.update_attributes((i,j), played_card, new_card)
            if disc > 0 and self.is_terminal(i, j):
                return self.player
        else:
            self.no_feasible_move += 1
            if self.no_feasible_move == 2:
                return 0

        self.player = 3 - self.player
        return -1

    def make_move(self, policy):
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
            # Find afterstate values
            x = np.zeros((self.nx, len(all_moves)))
            t = 0
            for i in range(len(legal_moves)):
                x1, x2 = legal_moves[i]
                x[:,t] = self.lookahead_attributes(legal_moves[i], self.cards_on_board[x1,x2], self.player)
                t += 1
            for i in range(len(legal_moves_1J)):
                x[:,t] = self.lookahead_attributes(legal_moves_1J[i], 48, 0)
                t += 1
            for i in range(len(legal_moves_2J)):
                x[:,t] = self.lookahead_attributes(legal_moves_2J[i], 49, self.player)
                t += 1

            nx, na = x.shape
            x = Variable(torch.tensor(x, dtype=torch.float, device=self.device))
            h = torch.mm(self.model[1], x) + self.model[0] @ torch.ones((1,na), device=self.device)
            h_tanh = h.tanh()
            y = torch.mm(self.model[3], h_tanh) + self.model[2]
            values = y.sigmoid().detach()

            k = int(torch.argmax(values))

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
            self.update_attributes((i,j), played_card, new_card)

            if self.is_terminal(i, j):
                return self.player
        else:
            self.no_feasible_move += 1
            if self.no_feasible_move == 2:
                return 0

        self.player = 3 - self.player
        return -1

    def heuristic_1(self, temp_board, pos, p):
        # Namminamm

        s = 0
        i, j = pos
        board_rows = 10
        board_cols = 10

        # Horizontal
        mint = max(-4, -j)
        maxt = min(4, board_cols-1-j)
        t = mint
        while t < 0:
            if temp_board[i][j+t] != p and temp_board[i][j+t] != 0:
                mint = t+1
            t += 1
        t = maxt
        while t > 0:
            if temp_board[i][j+t] != p and temp_board[i][j+t] != 0:
                maxt = t-1
            t -= 1

        range = maxt - mint + 1
        if range >= 5:
            t = mint
            while t <= maxt:
                if temp_board[i][j+t] == p:
                    s += min(min(t - mint, maxt - t), range - 5) + 1
                t += 1

        # Vertical
        mint = max(-4, -i)
        maxt = min(4, board_rows-1-i)
        t = mint
        while t < 0:
            if temp_board[i+t][j] != p and temp_board[i+t][j] != 0:
                mint = t+1
            t += 1
        t = maxt
        while t > 0:
            if temp_board[i+t][j] != p and temp_board[i+t][j] != 0:
                maxt = t-1
            t -= 1

        range = maxt - mint + 1
        if range >= 5:
            t = mint
            while t <= maxt:
                if temp_board[i+t][j] == p:
                    s += min(min(t - mint, maxt - t), range - 5) + 1
                t += 1

        # Main diagonal
        mint = max(-4, max(-i, -j))
        maxt = min(4, min(board_cols-1-i, board_rows-1-j))
        t = mint
        while t < 0:
            if temp_board[i+t][j+t] != p and temp_board[i+t][j+t] != 0:
                mint = t+1
            t += 1
        t = maxt
        while t > 0:
            if temp_board[i+t][j+t] != p and temp_board[i+t][j+t] != 0:
                maxt = t-1
            t -= 1

        range = maxt - mint + 1
        if range >= 5:
            t = mint
            while t <= maxt:
                if temp_board[i+t][j+t] == p:
                    s += min(min(t - mint, maxt - t), range - 5) + 1
                t += 1

        # Antidiagonal
        mint = max(-4, max(-i, j-9))
        maxt = min(4, min(9-i, j))
        t = mint
        while t < 0:
            if temp_board[i+t][j-t] != p and temp_board[i+t][j-t] != 0:
                mint = t+1
            t += 1
        t = maxt
        while t > 0:
            if temp_board[i+t][j-t] != p and temp_board[i+t][j-t] != 0:
                maxt = t-1
            t -= 1

        range = maxt - mint + 1
        if range >= 5:
            t = mint
            while t <= maxt:
                if temp_board[i+t][j-t] == p:
                    s += min(min(t - mint, maxt - t), range - 5) + 1
                t += 1

        return s

    def heuristic_2(self, pos, p):
        # Namminamminamm
        pos = tuple(pos)

        if self.discs_on_board[pos] != 0:
            return 0
        elif self.cards_on_board[pos] in self.hand[p]:
            return 1
        else:
            return 0.1

    def update_heuristic_1(self, pos):
        # Uppfæra Heuristic 1 eftir leik

        i, j = pos
        temp_board = self.discs_on_board.copy()

        if temp_board[i,j] != 0:
            self.heuristic_1_table[0][i,j] = 0
            self.heuristic_1_table[1][i,j] = 0
        else:
            self.heuristic_1_table[0][i,j] = self.heuristic_1(temp_board, (i, j), 1)
            self.heuristic_1_table[1][i,j] = self.heuristic_1(temp_board, (i, j), 2)

        for p in range(2):
            temp_board[0,0], temp_board[0,9], temp_board[9,0], temp_board[9,9] = [p+1]*4
            # Horizontal
            for t in range(1, min(5, 10 - j)):
                if temp_board[i][j+t] != 0 and temp_board[i][j+t] != p+1:
                    break
                self.heuristic_1_table[p][i][j+t] = self.heuristic_1(temp_board, (i, j+t), p+1)
            for t in range(1, min(5, j + 1)):
                if temp_board[i][j-t] != 0 and temp_board[i][j-t] != p+1:
                    break
                self.heuristic_1_table[p][i][j-t] = self.heuristic_1(temp_board, (i, j-t), p+1)

            # Vertical
            for t in range(1, min(5, 10 - i)):
                if temp_board[i+t][j] != 0 and temp_board[i+t][j] != p+1:
                    break
                self.heuristic_1_table[p][i+t][j] = self.heuristic_1(temp_board, (i+t, j), p+1)
            for t in range(1, min(5, i + 1)):
                if temp_board[i-t][j] != 0 and temp_board[i-t][j] != p+1:
                    break
                self.heuristic_1_table[p][i-t][j] = self.heuristic_1(temp_board, (i-t, j), p+1)

            # Main diagonal
            for t in range(1, min(5, min(10 - i, 10 - j))):
                if temp_board[i+t][j+t] != 0 and temp_board[i+t][j+t] != p+1:
                    break
                self.heuristic_1_table[p][i+t][j+t] = self.heuristic_1(temp_board, (i+t, j+t), p+1)
            for t in range(1, min(5, min(i + 1, j + 1))):
                if temp_board[i-t][j-t] != 0 and temp_board[i-t][j-t] != p+1:
                    break
                self.heuristic_1_table[p][i-t][j-t] = self.heuristic_1(temp_board, (i-t, j-t), p+1)

            # Antidiagonal
            for t in range(1, min(5, min(10 - i, j + 1))):
                if temp_board[i+t][j-t] != 0 and temp_board[i+t][j-t] != p+1:
                    break
                self.heuristic_1_table[p][i+t][j-t] = self.heuristic_1(temp_board, (i+t, j-t), p+1)
            for t in range(1, min(5, min(i + 1, 10 - j))):
                if temp_board[i-t][j+t] != 0 and temp_board[i-t][j+t] != p+1:
                    break
                self.heuristic_1_table[p][i-t][j+t] = self.heuristic_1(temp_board, (i-t, j+t), p+1)

    def set_attributes(self):
        temp_board = np.zeros(96, dtype=np.int8)

        # Heuristic 1
        self.heuristic_1_table = np.zeros((2, 10, 10))

        for i in range(2):
            for k in range(0, 9):
                self.heuristic_1_table[i][0,k] = 1
                self.heuristic_1_table[i][9,k] = 1

                self.heuristic_1_table[i][k,0] = 1
                self.heuristic_1_table[i][k,9] = 1

                self.heuristic_1_table[i][k,k] = 1
                self.heuristic_1_table[i][k,9-k] = 1

        # Heuristic 2
        self.heuristic_2_table = 0.1 + np.zeros((2, 10, 10))

        for i in range(2):
            for card in self.hand[i]:
                if card < 48:
                    self.heuristic_2_table[i][tuple(self.card_positions[card])] = 1

        attributes = []
        hf = np.multiply(self.heuristic_1_table, self.heuristic_2_table)
        for i in range(2):
            # One hot encoding á borði
            one_hot_board = np.zeros((temp_board.size, 3))
            one_hot_board[np.arange(temp_board.size),temp_board] = 1
            one_hot_board = one_hot_board.flatten()

            # Hönd
            hond = np.zeros(50, dtype=np.uint8)
            np.add.at(hond, self.hand[i], 1)

            mh1 = np.sum(self.heuristic_1_table[i]) / 100
            mh2 = np.sum(np.square(self.heuristic_1_table[i])) / 100
            mh3 = np.max(self.heuristic_1_table[i])
            mh4 = np.sum(hf[i]) / 100
            mh5 = np.sum(np.square(hf[i])) / 100
            mh6 = np.max(hf[i])
            meta_heuristics = np.array([mh1, mh2, mh3, mh4, mh5, mh6])

            attributes.append(np.concatenate((one_hot_board, hond, self.heuristic_1_table[i].flatten(), self.heuristic_2_table[i].flatten(), meta_heuristics, [self.sequences[i]])))
            # attributes.append(np.concatenate((one_hot_board, hond)))

        self.attributes = np.array(attributes)

    def update_attributes(self, pos, old_card, new_card):
        p = self.player - 1
        i, j = pos

        # Staður í eigindavigri; nauðsynlegt að taka tillit til hornanna
        attr_pos = 3 * (10*i + j - 1)
        if i > 8:
            attr_pos -= 6
        elif i > 0:
            attr_pos -= 3

        # Uppfæra þann stað
        disc = self.discs_on_board[i,j]
        if disc == 0:
            for k in range(2):
                new_attr = np.zeros(3)
                self.attributes[k][attr_pos:attr_pos+3] = new_attr
        else:
            for k in range(2):
                new_attr = np.zeros(3)
                new_attr[disc] = 1
                self.attributes[k][attr_pos:attr_pos+3] = new_attr
                disc = 3 - disc

        # Uppfæra hönd
        self.attributes[p][288+old_card] -= 1
        if new_card != -1:
            self.attributes[p][288+new_card] += 1

        # Heuristic 1
        self.update_heuristic_1(pos)

        # Heuristic 2
        for k in range(2):
            self.heuristic_2_table[k][i,j] = self.heuristic_2(pos, k)

        hf = np.multiply(self.heuristic_1_table, self.heuristic_2_table)
        t = 338
        for k in range(2):
            self.attributes[k][t:t+100] = self.heuristic_1_table[k].flatten()
            self.attributes[k][t+100+10*i+j] = self.heuristic_2_table[k][i,j]

            mh1 = np.sum(self.heuristic_1_table[k]) / 100
            mh2 = np.sum(np.square(self.heuristic_1_table[k])) / 100
            mh3 = np.max(self.heuristic_1_table[k])
            mh4 = np.sum(hf[k]) / 100
            mh5 = np.sum(np.square(hf[k])) / 100
            mh6 = np.max(hf[k])
            meta_heuristics = np.array([mh1, mh2, mh3, mh4, mh5, mh6])

            self.attributes[k][t+200:t+206] = meta_heuristics
            self.attributes[k][t+206] = self.sequences[k]

    def play_full_game(self, policy='random', verbose=True):
        if isinstance(policy, str):
            policy = tuple([policy]*2)
        self.initialize_game()
        w = -1
        while w == -1:
            if policy[self.player-1] == 'random':
                w = self.make_random_move()
            else:
                w = self.make_move(policy)
            if verbose:
                fig, axes = plt.subplots(1, 3)
                fig.set_size_inches(20, 6)
                im = axes[0].imshow(self.discs_on_board)
                fig.colorbar(im, ax=axes[0])
                im = axes[1].imshow(self.heuristic_1_table[0])
                fig.colorbar(im, ax=axes[1])
                im = axes[2].imshow(self.heuristic_1_table[1])
                fig.colorbar(im, ax=axes[2])
                plt.show()

    def learn2(self, policy=('greedy', 'random'), episodes=1000, verbose=True):
        wins = [0, 0]
        if verbose:
            indices = [(i * episodes) //  20 for i in range(20)]
            print('[', end='')

        for i in range(episodes):
            self.initialize_game()

            w = -1
            while w == -1:
                if policy[self.player-1] == 'random':
                    w = self.make_random_move()
                else:
                    w = self.make_move(policy)

            if w > 0:
                wins[w-1] += 1

            if verbose:
                while len(indices) > 0 and i == indices[0]:
                    print('=', end='')
                    del(indices[0])
        if verbose:
            print(f'] Finished {episodes} episodes')
        return wins

    def learn(self, policy=('greedy', 'greedy'), episodes=1000, verbose=True):
        # Implements Lambda Actor-Critic

        wins = [0, 0]
        if verbose:
            indices = [(i * episodes) //  20 for i in range(20)]
            print('[', end='')

        for i in range(episodes):
            self.initialize_game()

            x = Variable(torch.tensor(np.array([self.attributes[0]]).T, dtype=torch.float, device=self.device))
            h = torch.mm(self.model[1], x) + self.model[0]
            h_tanh = h.tanh()
            y = torch.mm(self.model[3], h_tanh) + self.model[2]
            y_sigmoid = y.sigmoid()
            self.old_value[0] = y_sigmoid

            self.make_move(policy)

            x = Variable(torch.tensor(np.array([self.attributes[1]]).T, dtype=torch.float, device=self.device))
            h = torch.mm(self.model[1], x) + self.model[0]
            h_tanh = h.tanh()
            y = torch.mm(self.model[3], h_tanh) + self.model[2]
            y_sigmoid = y.sigmoid()
            self.old_value[1] = y_sigmoid

            w = self.make_move(policy)

            while w == -1:
                p = self.player - 1
                x = Variable(torch.tensor(np.array([self.attributes[p]]).T, dtype=torch.float, device=self.device))
                h = torch.mm(self.model[1], x) + self.model[0]
                h_tanh = h.tanh()
                y = torch.mm(self.model[3], h_tanh) + self.model[2]
                y_sigmoid = y.sigmoid()
                new_value = y_sigmoid.detach()

                self.learn_step(p, new_value, 0)

                self.old_value[p] = y_sigmoid

                w = self.make_move(policy)

            if w > 0:
                self.learn_step(w - 1, 0, -1)
                self.learn_step(2 - w, 0, 1)
                wins[w-1] += 1
            else:
                self.learn_step(self.player - 1, 0, 0)
                self.learn_step(2 - self.player, 0, 0)

            if verbose:
                while len(indices) > 0 and i == indices[0]:
                    print('=', end='')
                    del(indices[0])
        if verbose:
            print(f'] Finished {episodes} episodes')
        return wins