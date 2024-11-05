# Install LightFM if you haven't already
# !pip install lightfm

import pandas as pd
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k, auc_score
import numpy as np

user_data = pd.read_csv('data/FS_user_id_operator_user_id.csv')
game_data = pd.read_csv('data/FS_game_id_names.csv')
user_game_data = pd.read_csv('data/FS_user_id_game_id.csv')

user_game_data = user_game_data.merge(user_data, on="user_id", how="left")
user_game_data = user_game_data.merge(game_data, on="game_id", how="left")

dataset = Dataset()
dataset.fit(
    (x for x in user_game_data['user_id'].unique()),
    (x for x in user_game_data['game_id'].unique())
)

(interactions, weights) = dataset.build_interactions(
    ((x['user_id'], x['game_id']) for _, x in user_game_data.iterrows())
)

model = LightFM(loss='warp')

model.fit(interactions, epochs=10, num_threads=4)

precision = precision_at_k(model, interactions, k=10).mean()
auc = auc_score(model, interactions).mean()

print("Precision@k (k=10): {:.4f}".format(precision))
print("AUC Score: {:.4f}".format(auc))

user_activity = np.array(interactions.sum(axis=1)).flatten()
print("User activity (number of interactions):", user_activity[:10])  # Пример для первых 10 пользователей

user_id = 1562727  # пример user_id
user_internal_id = dataset.mapping()[0].get(user_id)  # Преобразуем user_id в внутренний id LightFM

n_items = interactions.shape[1]  # Общее количество элементов
scores = model.predict(user_internal_id, list(range(n_items)))

top_items = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:10]
print("Top recommended items for user {}: {}".format(user_id, top_items))

