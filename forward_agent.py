
from carla.agent.agent import Agent
from carla.client import VehicleControl
import numpy as np
from xdsnewDQN import DeepQNetwork
from carla.image_converter import depth_to_local_point_cloud, to_rgb_array
class ForwardAgent(Agent):
    """
    Simple derivation of Agent Class,
    A trivial agent agent that goes straight
    """

    def run_step(self, measurements, sensor_data, directions, target, frame):
        actions = [-1.0, -0.5, 0, 0.5, 1, 0, 0.5, 1.0]
        control = VehicleControl()
        if frame < 30:
            control.throttle = 0
        else:




            control.throttle = 0.6

        return control
