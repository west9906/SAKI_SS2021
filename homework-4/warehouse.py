import numpy as np
import itertools
import pandas as pd
import copy


class Warehouse:
    def __init__(self):
        self.items = np.asarray(['empty', 'red', 'blue', 'white'])
        self.actions = ['restore_red', 'restore_blue', 'restore_white', 'store_red', 'store_blue', 'store_white']

        # -> [0, 1]
        #    [2, 3]
        self.positions = [0, 1, 2, 3]

        # -> [3, 2]
        #    [2, 1]
        self.reward = np.asarray([3, 2, 2, 1])

        self.df_action_dist = pd.read_csv("data/processed/action_distribution_2x2", index_col=0)

        self.states = np.asarray(list(itertools.product(self.items, self.items, self.items, self.items, self.actions)))

        self.N = len(self.states)

        self.test_set = pd.read_csv("data/processed/test_set_2x2", index_col=0)

    def check_action(self, p, s, s_p, a, c):
        if a == 'store':
            if s[p] == 'empty' and s_p[p] == c:
                if np.count_nonzero(s == s_p) == 3:
                    return True

        if a == 'restore':
            if s[p] == c and s_p[p] == 'empty':
                if np.count_nonzero(s == s_p) == 3:
                    return True

        return False

    def save_tpm(self):
        all_tpm = []
        for p in self.positions:
            tpm = np.zeros((self.N, self.N))
            for i, s in enumerate(self.states, start=0):  # s = current state
                for j, s_p in enumerate(self.states, start=0):  # s_p = next state
                    action = s[-1].split(sep='_')[0]
                    color = s[-1].split(sep='_')[1]
                    if self.check_action(p, s[:-1], s_p[:-1], action, color):
                        tpm[i, j] = self.df_action_dist['count'][self.df_action_dist['action'] == action][
                            self.df_action_dist['color'] == color].values[0]

            all_tpm.append(tpm)

        all_tpm = np.asarray(all_tpm)
        for i in range(len(all_tpm)):
            row_sums = all_tpm[i].sum(axis=1)
            all_tpm[i] = np.nan_to_num(all_tpm[i] / row_sums[:, np.newaxis], nan=0)

        np.save("data/processed/tpm_2x2.npy", np.asarray(all_tpm))

    def rewards_matrix(self):
        rm = np.zeros((self.N, len(self.positions)))

        for p in self.positions:
            for i, s in enumerate(self.states, start=0):
                action = s[-1].split(sep='_')[0]
                color = s[-1].split(sep='_')[1]

                if action == 'store' and s[p] == 'empty':
                    rm[i, p] = self.reward[p]

                elif action == 'restore' and s[p] == color:
                    rm[i, p] = self.reward[p]
        return rm

    def get_tpm(self):
        tpm = np.load("data/processed/tpm_2x2.npy")
        for i in range(len(tpm)):
            row_sums = tpm[i].sum(axis=1)
            tpm[i][np.diag_indices_from(tpm[i])] = 1 - row_sums

        return tpm

    def new_warehouse(self, wh, a_p, a, c):
        warehouse = copy.deepcopy(wh)
        if a == 'store' and warehouse[a_p] == 'empty':
            warehouse[a_p] = c
        elif a == 'restore' and warehouse[a_p] != 'empty':
            warehouse[a_p] = 'empty'
        return warehouse

    def get_current_state(self, wh, a, c):
        warehouse = copy.deepcopy(wh)
        action_color = a + "_" + c

        if len(warehouse) == len(self.positions):
            warehouse.append(action_color)
        elif len(warehouse) == len(self.positions) + 1:
            warehouse[len(self.positions)] = action_color
        return warehouse

    def test_rl_policy(self, policy: tuple):
        warehouse = ['empty'] * len(self.positions)

        traveled_distance = 0
        columns = ['0', '1', '2', '3', 'transaction']
        df_wh_s = pd.DataFrame(columns=columns)

        for i, row in self.test_set.iterrows():
            current_state = self.get_current_state(warehouse, row.action, row.color)

            idx_cs = np.where(np.all(self.states == current_state, axis=1))[0][0]

            action_position = policy[idx_cs]
            traveled_distance += (1 + self.reward[0] - self.reward[action_position]) * 2
            warehouse = self.new_warehouse(warehouse, action_position, row.action, row.color)

            new_row = {
                '0': str(warehouse[0]),
                '1': str(warehouse[1]),
                '2': str(warehouse[2]),
                '3': str(warehouse[3]),
                'transaction': row.action+"_"+row.color}

            df_wh_s = df_wh_s.append(new_row, ignore_index=True)

        result = [traveled_distance, df_wh_s]

        return result

    def get_state_policy(self, state, policy):
        idx_cs = np.where(np.all(self.states == state, axis=1))[0][0]
        return policy[idx_cs]

    def store_greedy(self, wh, c):
        for index, pos in enumerate(wh, start=0):
            if pos == 'empty':
                wh[index] = c
                result = [index, wh]
                return result

    def restore_greedy(self, warehouse, color):
        for index, pos in enumerate(warehouse, start=0):
            if pos == color:
                warehouse[index] = 'empty'
                result = [index, warehouse]
                return result

    def test_greedy(self):
        warehouse = ['empty'] * len(self.positions)

        traveled_distance = 0
        columns = ['0', '1', '2', '3', 'transaction']
        df_wh_s = pd.DataFrame(columns=columns)

        for index, row in self.test_set.iterrows():
            current_state = ''
            if row.action == 'store':
                current_state = self.store_greedy(warehouse, row.color)
                traveled_distance += (1 + self.reward[0] - self.reward[current_state[0]]) * 2

            elif row.action == 'restore':
                current_state = self.restore_greedy(warehouse, row.color)
                traveled_distance += (1 + self.reward[0] - self.reward[current_state[0]]) * 2

            new_row = {
                '0': str(current_state[1][0]),
                '1': str(current_state[1][1]),
                '2': str(current_state[1][2]),
                '3': str(current_state[1][3]),
                'transaction': row.action + "_" + row.color}

            df_wh_s = df_wh_s.append(new_row, ignore_index=True)

        result = [traveled_distance, df_wh_s]

        return result