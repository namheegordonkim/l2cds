from abc import abstractmethod

import pygame
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch import nn

DT = 0.033

np.set_printoptions(formatter={'float': lambda x: "{0:0.1f}".format(x)})

import os

from gym.envs.dart import dart_env

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
view_scale = np.asarray([60, 20])


class DummyParameter:

    def __init__(self):
        self.requires_grad = False


class Pendulum2DState:
    def __init__(self, angle, angular_velocity, phase):
        self.angle = angle
        self.angular_velocity = angular_velocity
        self.phase = phase

    @staticmethod
    def from_numpy(x):
        angle, angular_velocity, phase = x
        return Pendulum2DState(angle, angular_velocity, phase)

    @staticmethod
    def get_num_features():
        return 3


class Pendulum2D(dart_env.DartEnv):

    def __init__(self, skel_name: str, disable_viewer=True, output_phase=True, render_color=BLACK):
        self.render_color = render_color
        self.output_phase = output_phase
        self.time = 0
        self.mass = 10
        self.string_length = 0.6
        self.scale = 100
        self.skel_name = skel_name
        self.disable_viewer = disable_viewer

        # target angle cycle
        self.max_phase = 20
        self.k = 1 / 3 * np.pi  # amplitude
        # self.omega = np.sqrt(9.81 / self.string_length)  # frequency coefficient
        self.omega = 2 * np.pi / self.max_phase  # frequency coefficient
        # self.omega = 1  # frequency coefficient
        self.phase = 0
        self.target_angle = 0
        self.target_angular_velocity = 0
        self.set_phase(0)

        asset_fullpath = os.path.abspath(skel_name)
        control_bounds = np.array([[1.0] * 1, [-1.0] * 1])
        dart_env.DartEnv.__init__(self, asset_fullpath, 1, 4, control_bounds, dt=DT, disableViewer=disable_viewer)

        self.reset()

    def reset_model(self):
        # self.set_state(np.random.uniform(0, 2 * np.pi, 1), np.array([0]))
        # preserve phase
        self.set_state(np.array([self.target_angle]), np.array([self.target_angular_velocity]))
        return self._get_obs()

    def set_phase(self, phase):
        self.phase = phase
        self.phase %= self.max_phase
        self.target_angle = self.get_target_angle()
        self.target_angular_velocity = self.get_target_angular_velocity()

    def get_target_angular_velocity(self):
        return self.compute_target_angular_velocity(self.phase)

    def compute_target_angular_velocity(self, phase):
        return -self.k * self.omega * np.sin(self.omega * phase) / DT

    def get_target_angle(self):
        return self.compute_target_angle(self.phase)

    def compute_target_angle(self, phase):
        return self.k * np.cos(self.omega * phase)

    def step(self, action):
        """
        Action is a numpy array of shape (3,), encoding the state and the torque
        """
        # angle, angular_velocity, torque = action
        torque = action
        self.do_simulation(torque, 1)
        # self.set_state(np.array([self.target_angle]), np.array([self.target_angular_velocity]))
        self.set_phase(self.phase + 1)

        angle_error = (self.get_target_angle() - self.get_angle()) ** 2
        velocity_error = (self.get_target_angular_velocity() - self.get_angular_velocity()) ** 2
        done = False
        # if angle_error > 1.0 or velocity_error > 1.0:
        #     done = True

        angle_reward = np.exp(-angle_error)
        velocity_reward = np.exp(-velocity_error)

        total_reward = angle_reward + velocity_reward

        return self._get_obs(), total_reward, done, {}

    def get_angle(self):
        return self.robot_skeleton.q[0]

    def get_angular_velocity(self):
        return self.robot_skeleton.dq[0]

    def get_phase(self):
        return self.phase

    def _get_obs(self):
        if self.output_phase:
            # return np.asarray([self.get_angle(), self.get_angular_velocity(),
            #                    self.get_target_angle(), self.get_target_angular_velocity()])
            return np.asarray([self.get_angle(), self.get_angular_velocity(), self.get_phase()])

        else:
            return np.asarray([self.get_angle(), self.get_angular_velocity()])

    def viewer_setup(self):
        self._get_viewer().scene.tb._set_orientation(0, 0)
        self._get_viewer().scene.tb.trans[2] = -7.5
        self.track_skeleton_id = 0

    def render(self, mode='human', close=False):
        if not self.disableViewer:
            self._get_viewer().scene.tb.trans[0] = -self.dart_world.skeletons[self.track_skeleton_id].com()[0]*1
        if close:
            if self.viewer is not None:
                self._get_viewer().close()
                self.viewer = None
            return

        if mode == 'rgb_array':
            data = self._get_viewer().getFrame()
            return data
        elif mode == 'human':
            print(self._get_viewer().scene.tb.trans[0])
            print(self.dart_world.skeletons[self.track_skeleton_id].com()[0])
            self._get_viewer().runSingleStep()
        elif mode == 'pygame':
            self._render_pygame()

    def reset_for_normalization(self):
        return self.reset()

    def set_state_vector(self, state):
        angle, angular_velocity = state
        self.set_state(np.array([angle]), np.array([angular_velocity]))

    def get_state_vector(self):
        return self._get_obs()

    def render_pygame(self, screen):
        state = self._get_obs()
        angle = state[0]
        angular_velocity = state[1]

        l = self.string_length * 200
        start_pos = (320, 240)
        dx = l * np.cos(angle + np.pi / 2)
        dy = l * np.sin(angle + np.pi / 2)
        end_pos = (start_pos[0] + dx, start_pos[1] + dy)

        pygame.draw.line(screen, self.render_color, start_pos, end_pos, 5)


class Pendulum2DSolver(nn.Module):

    def __init__(self, pendulum: Pendulum2D, output_state=False):
        super().__init__()
        self.output_state = output_state
        self.mass = pendulum.mass
        self.string_length = pendulum.string_length
        self.input_dim = 4  # angle, angular velocity, target_angle, target_velocity
        self.pendulum = pendulum
        if output_state:
            self.output_dim = 3  # torque
        else:
            self.output_dim = 1

    def forward(self, x):
        """
        Assume that x is a tensor [theta, theta_dot]. Generate torque by using PD controls.
        """
        n, _ = x.shape
        angle = x[:, 0]
        angular_velocity = x[:, 1]
        phase = x[:, 2]

        target_angle = self.pendulum.compute_target_angle(phase)
        target_angular_velocity = self.pendulum.compute_target_angular_velocity(phase)

        # element-wise operations
        a = 50 * (target_angle - angle) + \
            100 * (target_angular_velocity - angular_velocity)
        a /= (self.mass * self.string_length ** 2)
        a = a.reshape(n, 1)
        if self.output_state:
            mu = torch.cat([x[:, :2], a], dim=1)
        else:
            mu = a
        log_std = -2.0 * torch.ones(self.output_dim).unsqueeze(0).expand_as(mu)

        return mu, log_std


class PhasedActor:
    def __init__(self):
        pass

    @abstractmethod
    def get_action(self, state, phase):
        raise NotImplementedError


class TorchRLActor(PhasedActor):
    """
    Unified actor class
    """

    def __init__(self, model, state_scaler: StandardScaler = None):
        super().__init__()
        self.model = model
        self.state_scaler = state_scaler

    def get_action(self, state, phase=None):
        if self.state_scaler is None:
            state = state.reshape(1, -1)
        else:
            state = self.state_scaler.transform(state.reshape(1, -1))
        state_tensor = torch.as_tensor(state).float()
        if phase is not None:
            phase_tensor = torch.as_tensor([phase]).reshape(-1, 1).float() / 20.
        else:
            phase_tensor = None
        mu, _ = self.model(state_tensor, phase_tensor)
        action = mu.cpu().detach().reshape(-1, ).numpy()
        return action


class SKLRLActor(PhasedActor):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def get_action(self, state, phase):
        action = self.model.predict(state[np.newaxis, ...])
        return action
