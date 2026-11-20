"""Finite best-first search over the four legal typed actions and terminal candidates.

The executor replans at every typed boundary, including while holding.
Supports, payload, arm anchors and frozen round memory remain distinct. Search has a declared budget and is
not an optimality certificate. No future MPC solve enters its objective.
"""

from dataclasses import dataclass
from itertools import permutations, combinations, count
import heapq, time
import numpy as np
from .model import cells
from .debug import traced


@dataclass(frozen=True)
class Transfer:
    block: int
    destination: str | int


def cell_order(c):
    p = cells()[c]
    return (p[0], p[1], c)


def color_goals(previous, destination, models):
    sites = sorted(
        [c for c in cells() if c.startswith(f"p{destination}")], key=cell_order
    )
    goals = []
    for order in permutations(range(5)):
        if any(a == b for a, b in zip(order, previous)):
            continue
        for selected in combinations(sites, 5):
            supports = [None] * 5
            for b, c in zip(order, selected):
                supports[b] = c
            xy = np.array([cells()[c][:2] for c in selected])
            D = (
                sum(np.sum((xy[i] - xy[j]) ** 2) for i in range(5) for j in range(i))
                / 10
                / (0.47**2)
            )
            cost = -8.0 * models.color[destination, order[0]] - 2.0 * D
            goals.append((tuple(supports), float(cost), tuple(order), float(D)))
    return goals


def letter_goals(previous, destination, models):
    goals = []
    for order in permutations(range(5)):
        if order[0] in (previous[0], previous[-1]):
            continue
        supports = [None] * 5
        supports[order[0]] = destination
        for lower, upper in zip(order, order[1:]):
            supports[upper] = lower
        goals.append(
            (tuple(supports), -8.0 * models.letter_utility(order), tuple(order), 0.0)
        )
    return goals


def clear(s):
    return [b for b in range(len(s)) if b not in s]


class Planner:
    def __init__(self, models, budget=100000, epsilon=0.1, seed=0, geometry=None):
        self.geometry = geometry
        self.models = models
        self.budget = budget
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.records = []

    # Q : How does it select the best legal action to take next?
    @traced("SYMBOLIC search: supports, goals, sites, arm anchor and held block; returns plan and terminal supports")
    def solve(self, supports, goals, sites, anchor=("home", None), held=None):
        start = time.perf_counter()
        n = len(goals[0][0])
        initial = tuple(supports.get(i) for i in range(n))
        targetcost = {g[0]: g[1] for g in goals}
        unit = sum(self.models.theta_C.values()) + 0.4
        targets = [g[0] for g in goals]
        terminal = np.array([g[1] for g in goals])

        def estimate(s, h, a):
            mismatch = np.array([sum(x != y for x, y in zip(s, g)) for g in targets])
            credit = 0.0
            if h is not None:
                credit = (
                    self.models.theta_C["MoveEmpty"]
                    + self.models.theta_C["PickLift"]
                    + 0.2
                )
            elif a[0] == "pre":
                credit = self.models.theta_C["MoveEmpty"] + 0.1
            return float(
                np.min(2.0 * np.maximum(0.0, unit * mismatch - credit) + terminal)
            )

        def successors(s, h, a):
            free = [b for b in range(n) if s[b] is not None and b not in s]
            if h is None:
                for b in free:
                    if a == ("pre", b):
                        ns = list(s)
                        ns[b] = None
                        yield "PickLift", b, s[b], tuple(ns), b, ("lift", b)
                    else:
                        yield "MoveEmpty", b, None, s, None, ("pre", b)
            else:
                for d in [c for c in sites if c not in s] + free:
                    if self.geometry is not None and not self.geometry(s, h, d):
                        continue
                    if a == ("above", d):
                        ns = list(s)
                        ns[h] = d
                        yield "PlaceRelease", h, d, tuple(ns), None, ("low", d)
                    else:
                        yield "MoveHolding", h, d, s, h, ("above", d)

        queue = []
        serial = count()
        initialkey = (initial, held, anchor)
        heapq.heappush(
            queue, (estimate(*initialkey), next(serial), 0.0, initialkey, ())
        )
        best = {initialkey: 0.0}
        expanded = 0
        while queue and expanded < self.budget:
            _, _, cost, key, path = heapq.heappop(queue)
            s, h, at = key
            if cost > best.get(key, float("inf")) + 1e-9:
                continue
            if h is None and s in targetcost:
                self.records.append(
                    dict(
                        expanded=expanded,
                        seconds=time.perf_counter() - start,
                        estimated_cost=cost + targetcost[s],
                        typed_actions=len(path),
                        model_version=self.models.version,
                    )
                )
                return list(path), s
            expanded += 1
            for kind, b, d, ns, nh, na in successors(s, h, at):
                nc = cost + self.models.theta_C[kind] + 0.1
                newkey = (ns, nh, na)
                if nc + 1e-9 >= best.get(newkey, float("inf")):
                    continue
                best[newkey] = nc
                heapq.heappush(
                    queue,
                    (
                        nc + estimate(ns, nh, na),
                        next(serial),
                        nc,
                        newkey,
                        path + ((kind, b, d),),
                    ),
                )
        raise RuntimeError(
            f"planning budget exhausted: {expanded} expansions; no impossibility claim"
        )


def order_from_supports(supports, kind):
    if kind == "color":
        return tuple(sorted(supports, key=lambda b: cell_order(supports[b])))
    base = next(b for b, s in supports.items() if isinstance(s, str))
    order = [base]
    while len(order) < len(supports):
        order.append(next(b for b, s in supports.items() if s == order[-1]))
    return tuple(order)
