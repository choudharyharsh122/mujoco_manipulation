"""Mission-aligned models: execution cost upward, terminal feedback once."""

import numpy as np
from .debug import traced, log


class Models:
    def __init__(self):
        self.theta_C = {
            t: 3.0 for t in ["MoveEmpty", "PickLift", "MoveHolding", "PlaceRelease"]
        }
        self.count_C = {t: 0 for t in self.theta_C}
        self.version = 0
        self.color = np.full((3, 5), 0.5)
        self.color_n = np.zeros((3, 5), int)
        self.letter_data = []
        self.intercept = 0.5
        self.letter = np.zeros((5, 5))

    @traced("LEARNING: realized action cost R_o; reward=-cost; update symbolic action-type cost estimate")
    def execution(self, kind, observed_cost):
        predicted = self.theta_C[kind]
        self.count_C[kind] += 1
        self.theta_C[kind] += (observed_cost - predicted) / self.count_C[kind]
        self.version += 1
        return dict(
            C_bar=predicted,
            R_o=observed_cost,
            R_e=None,
            R=-observed_cost,
            C_hat=observed_cost,
            E_C=observed_cost - predicted,
            theta_C=self.theta_C.copy(),
            version=self.version,
        )

    @traced("LEARNING: completed-round color rating updates platform/leader preference")
    def rate_color(self, platform, leader, rating):
        if rating is None:
            return
        self.color_n[platform, leader] += 1
        self.color[platform, leader] += (
            rating - self.color[platform, leader]
        ) / self.color_n[platform, leader]
        self.version += 1

    @staticmethod
    def features(order):
        x = np.zeros((5, 5))
        x[np.arange(5), order] = 1
        return x

    @traced("LEARNING: completed-round rating fits identity-position preferences")
    def rate_letter(self, order, rating, regularization=0.1):
        if rating is None:
            return
        self.letter_data.append((tuple(order), float(rating)))
        P = np.eye(5) - np.ones((5, 5)) / 5
        X = np.array([(P @ self.features(o) @ P).ravel() for o, r in self.letter_data])
        y = np.array([r for o, r in self.letter_data])
        self.intercept = float(y.mean())
        Xc = X - X.mean(axis=0)
        theta = np.linalg.solve(
            Xc.T @ Xc + regularization * np.eye(25), Xc.T @ (y - y.mean())
        ).reshape(5, 5)
        self.letter = P @ theta @ P
        self.intercept = float(y.mean() - np.mean(X @ self.letter.ravel()))
        self.version += 1

    def letter_utility(self, order):
        return self.intercept + float(np.sum(self.letter * self.features(order)))


class HiddenEvaluator:
    """Separate synthetic evaluator; its parameters are never passed to planner."""

    def __init__(self):
        self._preferred = (2, 3, 0)

    @traced("REWARD: synthetic terminal color rating; not an MPC reward")
    def color(self, platform, leader):
        return float(leader == self._preferred[platform])

    @traced("REWARD: synthetic terminal letter-order rating; not an MPC reward")
    def letter(self, order):
        return 0.5 * sum(order[i] in (0, 4) for i in (1, 3))
