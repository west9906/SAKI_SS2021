import pandas as pd

def run():
    train_set = pd.read_csv("data/external/warehousetraining_2x2.txt", delimiter='\t',
                                    names=["action", "color"])
    test_set = pd.read_csv("data/external/warehouseorder_2x2.txt", delimiter='\t',
                                   names=["action", "color"])

    action_distribution = train_set.copy()
    action_distribution = action_distribution.groupby(['action', 'color']).size().reset_index(name='count')
    action_distribution['count'] = action_distribution['count'].div(len(train_set))

    action_distribution.to_csv("data/processed/action_distribution_2x2")
    test_set.to_csv("data/processed/test_set_2x2")
