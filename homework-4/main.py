import preprocessing
from warehouse import Warehouse
import numpy as np
import mdptoolbox
import pandas as pd


#Step 1
# run only one time to create transition probability matrix
preprocessing.run()
warehouse = Warehouse()
warehouse.save_tpm()


#Step2
warehouse = Warehouse()
tpm = warehouse.get_tpm()
rewards_matrix = warehouse.rewards_matrix()

mdp_p = mdptoolbox.mdp.PolicyIteration(tpm, rewards_matrix, 0.9, max_iter=100)
mdp_v = mdptoolbox.mdp.ValueIteration(tpm, rewards_matrix, 0.9, max_iter=100)

mdp_p.run()
mdp_v.run()

result_p = warehouse.test_rl_policy(mdp_p.policy)
result_v = warehouse.test_rl_policy(mdp_v.policy)

print("ValueIteration Robot traveled: ", result_v[0])
value_iter_states = result_v[1]

print("PolicyIteration Robot traveled: ", result_p[0])
policy_iter_states = result_p[1]
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(policy_iter_states)

traveled_fields_greedy = warehouse.test_greedy()
print("Greedy Robot traveled: ", traveled_fields_greedy[0])


greedy_states = traveled_fields_greedy[1]
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(greedy_states)