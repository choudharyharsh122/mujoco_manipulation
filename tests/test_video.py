import numpy as np
from mission_mujoco.physics import Config, World


def test_video_captures_after_initialization_at_five_fps():
    for timestep in (0.001, 0.002):
        world = World(config=Config(timestep=timestep, dt=0.08))
        times = []
        world.capture = lambda: times.append(float(world.data.time))
        try:
            for _ in range(25):
                world.step(np.zeros(6), 0.0, render=True)
            assert 10 <= len(times) <= 11
            assert all(0.07 < b - a < 0.25 for a, b in zip(times, times[1:]))
        finally:
            world.close()
