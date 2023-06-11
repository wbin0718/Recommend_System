from collections import defaultdict

import numpy as np

from util.dataloader import DataLoader
from util.metric_calculator import MetricCalculator


class Recommender:
    def __init__(self, metric):
        self.metric = metric()
        self.dataset = DataLoader(num_users=1000, num_test_items=5, data_path="./data/ml-10M100K/").load()

    def random_recommend(self, **kwargs):
        unique_user_ids = sorted(self.dataset.train["user_id"].unique())
        unique_movie_ids = sorted(self.dataset.train["movie_id"].unique())
        user_id2index = dict(zip(unique_user_ids, range(len(unique_user_ids))))
        movie_id2index = dict(zip(unique_movie_ids, range(len(unique_movie_ids))))

        pred_matrix = np.random.uniform(0.5, 5.0, (len(unique_user_ids), len(unique_movie_ids)))

        movie_rating_predict = self.dataset.test.copy()
        pred_results = []
        for i, row in self.dataset.test.iterrows():
            user_id = row["user_id"]
            if row["movie_id"] not in movie_id2index:
                pred_results.append(np.random.uniform(0.5, 5.0))
                continue

            user_index = user_id2index[row["user_id"]]
            movie_index = movie_id2index[row["movie_id"]]
            pred_score = pred_matrix[user_index, movie_index]
            pred_results.append(pred_score)
        movie_rating_predict["rating_pred"] = pred_results

        pred_user2items = defaultdict(list)
        user_evaluated_movies = self.dataset.train.groupby("user_id").agg({"movie_id": list})["movie_id"].to_dict()

        for user_id in unique_user_ids:
            user_index = user_id2index[user_id]
            movie_indexes = np.argsort(-pred_matrix[user_index, :])
            for movie_index in movie_indexes:
                movie_id = unique_movie_ids[movie_index]
                if movie_id not in user_evaluated_movies[user_id]:
                    pred_user2items[user_id].append(movie_id)
                if len(pred_user2items[user_id]) == 10:
                    break
        metrics = self.metric.calc(
            self.dataset.test["rating"].tolist(),
            movie_rating_predict["rating_pred"],
            self.dataset.test_user2items,
            pred_user2items,
            10,
        )
        print(metrics)


if __name__ == "__main__":
    recommender = Recommender(MetricCalculator)
    recommender.random_recommend()
