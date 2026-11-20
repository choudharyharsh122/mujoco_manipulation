"""Declared finite-cell physical adaptations of B0 and the eight-block B1."""

from itertools import permutations


def supports_from_towers(towers, sites):
    s = {}
    for tower, site in zip(towers, sites):
        for k, b in enumerate(tower):
            s[b] = site if k == 0 else tower[k - 1]
    return s


def rule_towers(order):
    colors = "RBGYRBGY"
    towers = [order[:4], order[4:]]
    for tower in towers:
        if sum(colors[b] == "R" for b in tower) > 1:
            return False
        for j, b in enumerate(tower):
            if colors[b] == "R" and (j == 0 or colors[tower[j - 1]] != "B"):
                return False
            if colors[b] == "G" and not any(colors[c] == "Y" for c in tower[j + 1 :]):
                return False
    for a, b in [(0, 2), (4, 6)]:
        tower = next(t for t in towers if a in t)
        if b not in tower or tower.index(a) <= tower.index(b):
            return False
    return True


def fixture(kind):
    if kind == "b0":
        initial = supports_from_towers([(0, 1, 2)], ["p0c0"])
        target = supports_from_towers([(1, 0, 2)], ["p1c0"])
        return (
            initial,
            [(tuple(target[b] for b in range(3)), 0.0, (1, 0, 2), 0.0)],
            ["p0c0", "p1c0", "p0c2", "p0c5"],
        )
    if kind != "b1":
        raise ValueError(kind)
    initial = supports_from_towers([(0, 2, 3, 1), (4, 6, 7, 5)], ["p0c0", "p0c2"])
    goals = []
    for order in permutations(range(8)):
        if not rule_towers(order):
            continue
        target = supports_from_towers([order[:4], order[4:]], ["p1c0", "p1c2"])
        goals.append((tuple(target[b] for b in range(8)), 0.0, order, 0.0))
    return (
        initial,
        goals,
        ["p0c0", "p0c2", "p1c0", "p1c2", "p0c3", "p0c5", "p2c0", "p2c2"],
    )


class Milestone:
    """Optional B3 monitor: initial satisfaction counts; history persists."""

    def __init__(self, initial):
        self.value = bool(initial)

    def observe(self, current):
        self.value = self.value or bool(current)
        return self.value
