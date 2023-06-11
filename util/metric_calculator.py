import numpy as np
from sklearn.metrics import mean_squared_error

from util.models import Metrics


class MetricCalculator:
    def calc(self, true_rating, pred_rating, true_user2items, pred_user2items, k=10):
        rmse = self._calc_rmse(true_rating, pred_rating)
        precision_at_k = self._calc_precision_at_k(true_user2items, pred_user2items, k)
        recall_at_k = self._calc_recall_at_k(true_user2items, pred_user2items, k)

        return Metrics(rmse, precision_at_k, recall_at_k)

    def _precision_at_k(self, true_items, pred_items, k=10):
        if k == 0:
            return 0.0
        p_at_k = (len(set(true_items) & set(pred_items[:k]))) / k
        return p_at_k

    def _calc_precision_at_k(self, true_user2items, pred_user2items, k):
        scores = []

        for user_id in true_user2items.keys():
            p_at_k = self._precision_at_k(true_user2items[user_id], pred_user2items[user_id], k)
            scores.append(p_at_k)
        return np.mean(scores)

    def _calc_recall_at_k(self, true_user2items, pred_user2items, k):
        scores = []
        for user_id in true_user2items.keys():
            r_at_k = self._recall_at_k(true_user2items[user_id], pred_user2items[user_id], k)
            scores.append(r_at_k)
        return np.mean(scores)

    def _recall_at_k(self, true_items, pred_items, k=10):
        if len(true_items) == 0 or k == 0:
            return 0.0
        r_at_k = (len(set(true_items) & set(pred_items[:k]))) / len(true_items)
        return r_at_k

    def _calc_rmse(self, true_rating, pred_rating):
        return np.sqrt(mean_squared_error(true_rating, pred_rating))
