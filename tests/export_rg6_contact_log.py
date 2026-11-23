#!/usr/bin/env python3
"""Export recorded RG6 forces and moments about the cube center from a ROS bag."""

import argparse
from bisect import bisect_left
import csv
from pathlib import Path

import numpy as np


CUBE_TOPIC = "/other_objects"
CONTACT_TOPIC = "/bullet_ros/mission_contacts"
ARROW_METRES_PER_NEWTON = 0.004
MATCH_WINDOW = 0.05  # seconds
MIN_RESULTANT_FORCE = 0.1  # N, below which a line of action is poorly conditioned


def xyz(value):
    return np.array([value.x, value.y, value.z])


def vector_columns(name, unit):
    return [f"{name}_{axis}_{unit}" for axis in "xyz"]


def nearest_sample(stamps, stamp):
    index = bisect_left(stamps, stamp)
    candidates = range(max(0, index - 1), min(len(stamps), index + 1))
    return min(candidates, key=lambda i: abs(stamps[i] - stamp))


def line_of_action(force, moment):
    force_squared = force @ force
    if force_squared < MIN_RESULTANT_FORCE ** 2:
        return None, None
    closest_point = np.cross(force, moment) / force_squared
    axial_moment = force @ moment / np.sqrt(force_squared)
    return 1000 * np.linalg.norm(closest_point), axial_moment


def export(bag_path, prefix, mass, legacy_friction_sign):
    import rosbag

    point_path = Path(f"{prefix}_points.csv")
    wrench_path = Path(f"{prefix}_wrench.csv")
    with rosbag.Bag(str(bag_path)) as bag:
        start = bag.get_start_time()
        cube_stamps, cube_poses = [], []
        for _, message, stamp in bag.read_messages(topics=[CUBE_TOPIC]):
            for marker in message.markers:
                if (marker.header.frame_id == "base_link"
                        and marker.mesh_resource.endswith("/mission_block.obj")):
                    pose = marker.pose
                    q = pose.orientation
                    cube_stamps.append(stamp.to_sec())
                    cube_poses.append((xyz(pose.position), (q.x, q.y, q.z, q.w)))
                    break
        topics = bag.get_type_and_topic_info().topics
        if not cube_stamps or CONTACT_TOPIC not in topics:
            raise ValueError(f"The bag must contain the mission block on {CUBE_TOPIC} and {CONTACT_TOPIC}")

        with point_path.open("w", newline="") as point_file, wrench_path.open("w", newline="") as wrench_file:
            points = csv.writer(point_file)
            wrench = csv.writer(wrench_file)
            points.writerow(["bag_seconds", "stamp_s", "finger", "contact_id", "component",
                             *vector_columns("point", "m"), *vector_columns("force", "N"),
                             *vector_columns("moment", "Nm")])
            columns = ["bag_seconds", "stamp_s", "cube_pose_match_ms", "left_contacts", "right_contacts",
                       *vector_columns("cube_center", "m"), "cube_qx", "cube_qy", "cube_qz", "cube_qw"]
            for component in ("normal", "friction", "finger_total"):
                columns.extend(vector_columns(f"{component}_force", "N"))
                columns.extend(vector_columns(f"{component}_moment", "Nm"))
                columns.extend([f"{component}_line_offset_mm", f"{component}_axial_moment_Nm"])
            columns.extend(["gravity_force_z_N", "finger_plus_gravity_force_z_N"])
            wrench.writerow(columns)
            samples = skipped = 0
            for _, message, recorded in bag.read_messages(topics=[CONTACT_TOPIC]):
                stamp = recorded.to_sec()
                index = nearest_sample(cube_stamps, stamp)
                match_error = abs(cube_stamps[index] - stamp)
                if match_error > MATCH_WINDOW:
                    skipped += 1
                    continue
                center, orientation = cube_poses[index]
                forces = {component: np.zeros(3) for component in ("normal", "friction")}
                moments = {component: np.zeros(3) for component in forces}
                counts = dict(left=0, right=0)
                for marker in message.markers:
                    if marker.header.frame_id != "base_link" or len(marker.points) != 2:
                        continue
                    finger, _, component = marker.ns.partition("_")
                    if finger not in counts or component not in forces:
                        continue
                    origin, endpoint = (xyz(point) for point in marker.points)
                    force = (endpoint - origin) / ARROW_METRES_PER_NEWTON
                    if legacy_friction_sign and component == "friction":
                        force = -force
                    moment = np.cross(origin - center, force)
                    forces[component] += force
                    moments[component] += moment
                    counts[finger] += component == "normal"
                    points.writerow([stamp - start, stamp, finger, marker.id, component,
                                     *origin, *force, *moment])
                row = [stamp - start, stamp, match_error * 1000, counts["left"], counts["right"],
                       *center, *orientation]
                total_force = forces["normal"] + forces["friction"]
                total_moment = moments["normal"] + moments["friction"]
                for force, moment in ((forces["normal"], moments["normal"]),
                                      (forces["friction"], moments["friction"]),
                                      (total_force, total_moment)):
                    row.extend([*force, *moment, *line_of_action(force, moment)])
                gravity = -9.81 * mass
                wrench.writerow([*row, gravity, total_force[2] + gravity])
                samples += 1
    print(f"Exported {samples} samples, skipped {skipped} without a nearby cube pose.")
    print(point_path)
    print(wrench_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--mass-kg", type=float, default=0.08)
    parser.add_argument("--legacy-friction-sign", action="store_true",
                        help="Correct orange arrows recorded before the friction direction was fixed")
    args = parser.parse_args()
    if not np.isfinite(args.mass_kg) or args.mass_kg <= 0:
        parser.error("--mass-kg must be finite and positive")
    export(args.bag, args.output_prefix, args.mass_kg, args.legacy_friction_sign)


if __name__ == "__main__":
    main()
