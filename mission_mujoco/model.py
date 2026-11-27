"""Vendored UR10e kinematics/inertias; explicitly idealized two-slider gripper."""

from pathlib import Path
import xml.etree.ElementTree as E
import numpy as np

ROOT = Path(__file__).resolve().parent
SIDE = 0.05
TOP = 0.32
PLATFORMS = np.array([[0.50, -0.36], [0.80, 0], [0.50, 0.36]])
COLORS = [
    (0.8, 0.1, 0.1, 1),
    (0.9, 0.7, 0.1, 1),
    (0.1, 0.2, 0.8, 1),
    (0.1, 0.7, 0.2, 1),
    (0.6, 0.1, 0.7, 1),
]


def cells():
    return {
        f"p{p}c{i}": np.array([xy[0] + dx, xy[1] + dy, TOP + SIDE / 2])
        for p, xy in enumerate(PLATFORMS)
        for i, (dx, dy) in enumerate(
            (
                (-0.11, -0.06),
                (0, -0.06),
                (0.11, -0.06),
                (-0.11, 0.06),
                (0, 0.06),
                (0.11, 0.06),
            )
        )
    }


def scene(
    supports=None,
    mass_scale=1.0,
    friction=3.0,
    timestep=0.002,
    impratio=1.0,
    support_solref=0.02,
):
    root = E.parse(ROOT / "assets/upstream_ur10e.xml").getroot()
    root.find("compiler").attrib.pop("meshdir", None)
    if impratio != 1.0:
        root.find("option").set("impratio", str(impratio))
    E.SubElement(root.find("option"), "flag", multiccd="enable")
    root.find("option").attrib.update(
        timestep=str(timestep),
        gravity="0 0 -9.81",
        cone="elliptic",
        iterations="60",
        tolerance="1e-9",
    )
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "mesh" or (
                child.tag == "geom" and child.get("class") == "visual"
            ):
                parent.remove(child)
    for tag in ["actuator", "keyframe"]:
        root.remove(root.find(tag))
    for i, g in enumerate(root.findall(".//worldbody//geom")):
        g.set("name", f"arm_geom_{i}")
        g.set("rgba", ".55 .67 .72 1")
    vertices = " ".join(
        str(v)
        for x in [-SIDE / 2, SIDE / 2]
        for y in [-SIDE / 2, SIDE / 2]
        for z in [-SIDE / 2, SIDE / 2]
        for v in (x, y, z)
    )
    E.SubElement(root.find("asset"), "mesh", name="cube_convex", vertex=vertices)
    world = root.find("worldbody")
    E.SubElement(
        world,
        "geom",
        name="floor",
        type="plane",
        size="2 2 .1",
        rgba=".22 .24 .26 1",
        friction=".8 .005 .0001",
    )
    E.SubElement(world, "light", pos="1 -1 3", dir="-1 1 -3", diffuse=".7 .7 .7")
    for i, xy in enumerate(PLATFORMS):
        E.SubElement(
            world,
            "geom",
            name=f"platform{i}",
            type="box",
            pos=f"{xy[0]} {xy[1]} {TOP - 0.04}",
            size=".19 .135 .04",
            rgba=".55 .45 .32 1",
            friction="1 .005 .0001",
        )
    wrist = root.find(".//body[@name='wrist_3_link']")
    grip = E.SubElement(wrist, "body", name="gripper", pos="0 .1 0", quat="-1 1 0 0")
    E.SubElement(
        grip,
        "geom",
        name="palm",
        type="box",
        pos="0 0 .023",
        size=".06 .028 .02",
        mass=".35",
        rgba=".15 .16 .18 1",
    )
    E.SubElement(grip, "site", name="tcp", pos="0 0 .105", size=".005", rgba="1 0 0 1")
    for s, sign in [("left", 1), ("right", -1)]:
        body = E.SubElement(grip, "body", name=s + "_finger", pos=f"{sign * 0.060} 0 0")
        E.SubElement(
            body,
            "joint",
            name=s + "_slide",
            type="slide",
            axis=f"{-sign} 0 0",
            range="0 .049",
            damping="2",
            armature=".002",
        )
        E.SubElement(
            body,
            "geom",
            name=s + "_stem",
            type="box",
            pos=f"{sign * 0.006} 0 .065",
            size=".003 .016 .025",
            mass=".05",
            rgba=".25 .25 .27 1",
        )
        E.SubElement(
            body,
            "geom",
            name=s + "_pad",
            priority="1",
            type="box",
            pos="0 0 .105",
            size=".007 .018 .020",
            mass=".03",
            rgba=".08 .09 .1 1",
            friction=f"{friction} .002 .0001",
            condim="3",
            solref=".006 1",
            solimp=".95 .99 .001",
        )
    act = E.SubElement(root, "actuator")
    joints = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]
    for j, limit in zip(joints, [330, 330, 150, 56, 56, 56]):
        E.SubElement(
            act,
            "motor",
            name=j + "_motor",
            joint=j,
            ctrlrange=f"-{limit} {limit}",
            forcerange=f"-{limit} {limit}",
            gear="1",
        )
    for s in ["left", "right"]:
        E.SubElement(
            act,
            "position",
            name=s + "_servo",
            joint=s + "_slide",
            kp="2000",
            kv="20",
            ctrlrange="0 .049",
            forcerange="-25 25",
        )
    supports = supports or {i: f"p0c{i}" for i in range(5)}
    positions = {}

    def pos(b):
        if b not in positions:
            sup = supports[b]
            positions[b] = (
                cells()[sup].copy() if isinstance(sup, str) else pos(sup) + [0, 0, SIDE]
            )
        return positions[b]

    for i in range(len(supports)):
        xyz = pos(i)
        body = E.SubElement(
            world, "body", name=f"block{i}", pos=" ".join(map(str, xyz))
        )
        E.SubElement(body, "freejoint", name=f"free{i}")
        E.SubElement(
            body,
            "geom",
            name=f"cube{i}",
            type="mesh",
            mesh="cube_convex",
            mass=str(0.08 * mass_scale),
            rgba=" ".join(
                map(
                    str,
                    COLORS[
                        ([0, 2, 3, 1, 0, 2, 3, 1][i] if len(supports) == 8 else i % 5)
                    ],
                )
            ),
            friction=f"{friction} .002 .0001",
            condim="3",
            solref=".02 1" if support_solref == 0.02 else f"{support_solref} 1",
            solimp=".9 .95 .001",
        )
    E.SubElement(root, "visual")
    return E.tostring(root, encoding="unicode")
