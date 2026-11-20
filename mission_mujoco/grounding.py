"""Geometric fuzzy degrees, globally consistent support forests, and guards.

All coordinates are metres. These are satisfaction degrees, not probabilities.
No intended action effect is used to force the observed support assignment.
"""

from .debug import traced
from dataclasses import dataclass
from itertools import product
import numpy as np
from scipy.spatial import ConvexHull
from .model import cells, SIDE, TOP


def taper(d, core, cutoff):
    return float(np.clip((cutoff - d) / (cutoff - core), 0, 1))


def polygon_area(p):
    p = np.asarray(p)
    return (
        abs(
            float(
                np.dot(p[:, 0], np.roll(p[:, 1], -1))
                - np.dot(p[:, 1], np.roll(p[:, 0], -1))
            )
        )
        / 2
        if len(p) > 2
        else 0.0
    )


def intersect(subject, clip):
    """Sutherland--Hodgman, convex counter-clockwise polygons."""
    out = list(np.asarray(subject))
    for a, b in zip(clip, np.roll(clip, -1, axis=0)):
        inp = out
        out = []
        if not inp:
            break
        side = lambda p: float(
            (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        )
        prev = inp[-1]
        dp = side(prev)
        for cur in inp:
            dc = side(cur)
            if (dc >= -1e-12) != (dp >= -1e-12):
                out.append(prev + (cur - prev) * dp / (dp - dc))
            if dc >= -1e-12:
                out.append(cur)
            prev, dp = cur, dc
    return np.asarray(out)


def square(center, half):
    return np.asarray(center)[:2] + np.array(
        [[-half, -half], [half, -half], [half, half], [-half, half]]
    )


def footprint(p, R):
    vertices = np.array(
        [
            [x, y, -SIDE / 2]
            for x, y in [
                (-SIDE / 2, -SIDE / 2),
                (SIDE / 2, -SIDE / 2),
                (SIDE / 2, SIDE / 2),
                (-SIDE / 2, SIDE / 2),
            ]
        ]
    )
    return (vertices @ R.T + p)[:, :2]


def valid_forest(supports):
    if len(set(supports.values())) != len(supports):
        return False
    for b in supports:
        seen = {b}
        s = supports[b]
        while isinstance(s, (int, np.integer)):
            if s in seen or s not in supports:
                return False
            seen.add(s)
            s = supports[s]
    return True


@dataclass
class Grounding:
    supports: dict | None
    held: int | None
    memberships: dict
    gap: float
    accepted: bool
    reason: str


def support_components(positions, rotations, held=None):
    """Pure geometric membership inputs and factors for every support candidate."""
    footprints = [footprint(p, R) for p, R in zip(positions, rotations)]
    topfaces = [footprint(p + SIDE * R[:, 2], R)
                for p, R in zip(positions, rotations)]
    result = {}
    for b, p in enumerate(positions):
        if b == held:
            continue
        angle = float(np.arccos(np.clip(rotations[b][2, 2], -1, 1)))
        orientation = taper(angle, 0.035, 0.17)
        result[b] = {}
        for s in list(cells()) + [i for i in range(len(positions)) if i != b and i != held]:
            if isinstance(s, str):
                poly, height = square(cells()[s], 0.040), TOP
            else:
                poly = topfaces[s]
                normal = rotations[s][:, 2]
                top = positions[s] + SIDE / 2 * normal
                bottom = p - SIDE / 2 * rotations[b][:, 2]
                height = top[2] - float(normal[:2] @ (bottom[:2] - top[:2])) / max(normal[2], 1e-6)
            area = polygon_area(intersect(footprints[b], poly)) / max(polygon_area(footprints[b]), 1e-12)
            height_error = float(p[2] - SIDE / 2 * rotations[b][2, 2] - height)
            height_degree = taper(abs(height_error), 0.002, 0.009)
            result[b][s] = dict(area_ratio=float(area), height_error=height_error,
                                orientation_error=angle, area=float(area),
                                height=height_degree, orientation=orientation,
                                combined=float(area * height_degree * orientation))
    return result


class Grounder:
    def __init__(self, eta_on=0.80, eta_off=0.70, gap=0.10, diagnostics=None):
        self.eta_on = eta_on
        self.eta_off = eta_off
        self.gap = gap
        self.previous = None
        self.diagnostics = diagnostics

    @traced("CONTINUOUS-to-SYMBOLIC grounding: physical World plus held identity; returns acceptance, support relations and fuzzy memberships")
    def measure(self, w, held=None, require_settled=False, settled_blocks=None, plot_final=False):
        positions = w.observe()
        rotations = [w.data.xmat[x].reshape(3, 3).copy() for x in w.blocks]
        components = support_components(positions, rotations, held)
        result = self._measure(w, positions, rotations, components, held,
                               require_settled, settled_blocks)
        if self.diagnostics is not None and (self.diagnostics.mode == 'all' or plot_final):
            truth = np.array([w.blockpos(b) for b in range(w.nblock)])
            self.diagnostics.grounding(w, result, components,
                                      support_components(truth, rotations, held),
                                      positions, truth, self, require_settled, plot_final)
        return result

    def _measure(self, w, positions, rotations, components, held,
                 require_settled, settled_blocks):
        # Free-joint velocities give a direct released-motion check, independent
        # of geometric membership and intended action effects.
        for b, body in enumerate(w.blocks):
            if (
                b == held
                or not require_settled
                or (settled_blocks is not None and b not in settled_blocks)
            ):
                continue
            dof = w.model.jnt_dofadr[w.model.body(body).jntadr[0]]
            velocity = w.data.qvel[dof : dof + 6]
            if (
                np.linalg.norm(velocity[:3]) > 0.01
                or np.linalg.norm(velocity[3:]) > 0.10
            ):
                return Grounding(
                    None, held, {}, 0.0, False, "supported-body velocity threshold"
                )
        footprints = [footprint(p, R) for p, R in zip(positions, rotations)]
        topfaces = [footprint(p + SIDE * R[:, 2], R)
                    for p, R in zip(positions, rotations)]
        membership = {b: {s: v["combined"] for s, v in options.items()}
                      for b, options in components.items()}
        candidates = {b: [s for s, v in options.items() if v >= self.eta_off]
                      for b, options in membership.items()}
        if any(not c for c in candidates.values()):
            return Grounding(None, held, membership, 0.0, False, "unassigned support")
        keys = list(candidates)
        forests = []
        for values in product(*(candidates[b] for b in keys)):
            graph = dict(zip(keys, values))
            if valid_forest(graph):
                forests.append((sum(membership[b][s] for b, s in graph.items()), graph))
        forests.sort(key=lambda x: x[0], reverse=True)
        if not forests:
            return Grounding(
                None, held, membership, 0.0, False, "support forest inconsistent"
            )
        gap = forests[0][0] - forests[1][0] if len(forests) > 1 else float(len(keys))
        best = forests[0][1]
        prior = {b: s for b, s in (self.previous or {}).items() if b != held}
        if (
            prior
            and set(prior) == set(keys)
            and valid_forest(prior)
            and all(membership[b].get(s, 0) >= self.eta_off for b, s in prior.items())
        ):
            best = prior.copy()
        elif gap < self.gap or any(
            membership[b][s] < (self.eta_off if prior.get(b) == s else self.eta_on)
            for b, s in best.items()
        ):
            return Grounding(
                None, held, membership, gap, False, "threshold or assignment gap"
            )
        # Contact corroboration is independent of geometric overlap.
        for b, s in best.items():
            support_geom = (
                w.model.geom("platform" + s[1]).id
                if isinstance(s, str)
                else w.blockgeoms[s]
            )
            if not any(
                c["other"] == support_geom and c["normal_force"] > 0.005
                for c in w.contacts(b)
            ):
                return Grounding(
                    None, held, membership, gap, False, "support contact absent"
                )
        if not stable_com(best, positions, footprints, topfaces=topfaces):
            return Grounding(None, held, membership, gap, False, "substack COM margin")
        self.previous = best.copy()
        return Grounding(best, held, membership, gap, True, "accepted")


def stable_com(supports, positions, footprints, margin=0.003, topfaces=None):
    """Equal-mass cubes; test COM of the entire substack at every interface."""
    topfaces = footprints if topfaces is None else topfaces
    for b, s in supports.items():
        above = [b]
        current = b
        while True:
            child = next((i for i, d in supports.items() if d == current), None)
            if child is None:
                break
            above.append(child)
            current = child
        support_polygon = (
            square(cells()[s], 0.04) if isinstance(s, str) else topfaces[s]
        )
        poly = intersect(footprints[b], support_polygon)
        if len(poly) < 3:
            return False
        com = np.mean(positions[above, :2], axis=0)
        hull = ConvexHull(poly)
        if np.any(hull.equations[:, :2] @ com + hull.equations[:, 2] > -margin):
            return False
    return True


def anchor_membership(position, target, rotation_error=0.0):
    d = float(
        np.sum(((np.asarray(position) - target) / 0.004) ** 2)
        + (rotation_error / 0.04) ** 2
    )
    return 1.0 if d <= 1 else (float(np.exp(-(d - 1))) if d < 9 else 0.0)


def conjunction(memberships, supports):
    result = 1.0
    for b, options in memberships.items():
        for s, v in options.items():
            result *= v if supports.get(b) == s else 1 - v
    return float(result)
