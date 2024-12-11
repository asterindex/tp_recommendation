# Install LightFM if you haven't already
# !pip install lightfm

import pickle
import pandas as pd
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k, auc_score
import numpy as np

## Data loading from csv
user_data = pd.read_csv('data/FS_user_id_operator_user_id.csv')
game_data = pd.read_csv('data/FS_game_id_names.csv')
user_game_data = pd.read_csv('data/FS_user_id_game_id.csv')

# data merge
user_game_data = user_game_data.merge(user_data, on="user_id", how="left")
user_game_data = user_game_data.merge(game_data, on="game_id", how="left")

# Init Dataset для LightFM
dataset = Dataset()
dataset.fit(
    (x for x in user_game_data['user_id'].unique()),
    (x for x in user_game_data['game_id'].unique())
)

# Matrix cretion
(interactions, weights) = dataset.build_interactions(
    ((x['user_id'], x['game_id']) for _, x in user_game_data.iterrows())
)

# Model init ???
model = LightFM(loss='warp')

# Train
model.fit(interactions, epochs=60, num_threads=1)

# precision_at_k и auc_score
precision = precision_at_k(model, interactions, k=10).mean()
auc = auc_score(model, interactions).mean()

print("Precision@k (k=10): {:.4f}".format(precision))
print("AUC Score: {:.4f}".format(auc))

# saving *******

# Save the model to a file
model_file = 'lightfm_model.pkl'
with open(model_file, 'wb') as f:
    pickle.dump(model, f)
print(f"Model saved to {model_file}")

# Load the model back from the file
with open(model_file, 'rb') as f:
    loaded_model = pickle.load(f)
print("Model loaded successfully")



# Obtain user activities
user_activity = np.array(interactions.sum(axis=1)).flatten()
print("User activity (number of interactions):", user_activity[:10])  # The first 10 users

# Recommendation for UserId
user_id = 1562727  # пример user_id
user_internal_id = dataset.mapping()[0].get(user_id)  # Преобразуем user_id в внутренний id LightFM

n_items = interactions.shape[1]  # Общее количество элементов
scores = loaded_model.predict(user_internal_id, list(range(n_items)))


# Sorting
top_items = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:10]
print("Top recommended items for user {}: {}".format(user_id, top_items))

