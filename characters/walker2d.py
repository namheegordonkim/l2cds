from collections import deque

import os
import torch

import torch.utils.data
import pickle
import random
import numpy as np

from gym import utils
from gym.envs.dart import dart_env
from torch.autograd import Variable
from models import ActorCriticNet
from utils.skel_generator import generate_skel_with_params
from utils.utils import reflect_control_vector
from utils.factories import expand_reference_motion


class Walker2D(dart_env.DartEnv, utils.EzPickle):
    def __init__(self, skel_name='./custom_dart_assets/walker2d_mini.skel', obs_dim=17, disable_viewer=True):
        self.control_bounds = np.array([[1.0] * 6, [-1.0] * 6])
        self.action_scale = np.array([100, 100, 100, 100, 100, 100])
        self.time_limit = 1000
        self.time = 0
        self.energy = 0
        self.sim_freq = 1

        asset_fullpath = os.path.abspath(skel_name)
        dart_env.DartEnv.__init__(self, asset_fullpath, 4, obs_dim, self.control_bounds, disableViewer=disable_viewer)
        self.track_skeleton_id = 1  # usually 0 is ground and 1 is walker

        try:
            self.dart_world.set_collision_detector(3)
        except Exception as e:
            print('Does not have ODE collision detector, reverted to bullet collision detector')
            self.dart_world.set_collision_detector(2)

        utils.EzPickle.__init__(self)

    def compute_reward(self):
        alive_bonus = 4.0
        vel = (self.posafter - self.posbefore) / self.dt
        # energy_reward = -np.square(a).sum()

        # reward = np.exp(-(self.robot_skeleton.dq[0] - 4.0) **2)# * 0.5 + energy_reward * 0.5
        vel_reward = -np.abs(vel - 2.0)
        reward = vel_reward * 2

        return reward

    def clamp_control(self, a):
        clamped_control = np.array(a)
        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]
            if clamped_control[i] < self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]
        return clamped_control

    def step(self, a):

        self.time += 1
        # print(self.time)
        self.posbefore = self.robot_skeleton.q[0]

        clamped_control = self.clamp_control(a)

        # propagate simulation clock
        tau = np.zeros(self.robot_skeleton.ndofs)
        tau[3:] = clamped_control * self.action_scale
        self.do_simulation(tau, self.frame_skip)

        ang = self.robot_skeleton.q[2]
        self.posafter = self.robot_skeleton.q[0]

        height = self.robot_skeleton.bodynodes[2].com()[1]

        contacts = self.dart_world.collision_result.contacts
        total_force_mag = 0
        for contact in contacts:
            total_force_mag += np.square(contact.force).sum()
        #
        # alive_bonus = 4.0
        # vel = (posafter - posbefore) / self.dt
        # energy_reward = -np.square(a).sum()
        #
        # # reward = np.exp(-(self.robot_skeleton.dq[0] - 4.0) **2)# * 0.5 + energy_reward * 0.5
        # vel_reward = -np.abs(vel - 2.0)
        # reward = (vel_reward * 2 + alive_bonus + energy_reward * 0 * 1e-3) / 4.0
        # print(reward * 16)
        # reward += alive_bonus
        # reward /= 2.0
        # print(self.time, reward)
        # reward -= 1e-3 * np.square(a).sum()

        # vel_reward = np.exp(-(self.robot_skeleton.dq[0] - 1.5)**2)
        # reward = 0.5 * vel_reward + 0.5 * energy_reward
        # print("vel", vel)
        reward = self.compute_reward()
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > .8) and (height < 2.0) and (abs(ang) < 1.0))

        if self.time_limit <= self.time:
            print("time reached")
            done = True

        ob = self._get_obs()
        self.energy += np.square(a).sum()

        return ob, reward, done, {}

    def _get_obs(self):
        state = np.concatenate([
            self.robot_skeleton.q[1:],
            np.clip(self.robot_skeleton.dq, -10, 10)
        ])
        # state[0] = self.robot_skeleton.bodynodes[2].com()[1]

        return state

    def reset_model(self):
        self.dart_world.reset()
        self.time = 0
        self.reinitialize()

        return self._get_obs()

    def reset_for_normalization(self):
        return self.reset()

    def reset_for_test(self):
        return self.reset()

    def reinitialize(self):
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        self.set_com_velocity(2)
        self.set_state(qpos, qvel)

    def viewer_setup(self):
        self._get_viewer().scene.tb._set_orientation(0, 0)
        self._get_viewer().scene.tb.trans[2] = -5.5

    def get_com_velocity(self):
        # vel = (self.posafter - self.posbefore) / (self.sim_freq * 0.002)
        # return vel
        return self.robot_skeleton.dq[0]

    def get_joint_angles(self):
        return self.robot_skeleton.q

    def get_joint_velocities(self):
        return self.robot_skeleton.dq

    def set_joint_velocities(self, dq):
        qpos = self.robot_skeleton.q
        qvel = dq
        self.set_state(qpos, qvel)

    def set_com_velocity(self, com_vel):
        qpos = self.robot_skeleton.q
        qvel = self.robot_skeleton.dq
        qvel[0] = com_vel
        self.set_state(qpos, qvel)


class Walker2DPhase(Walker2D):
    """
    Indirect sample-based environment where reward is specified in terms of similarity between generated movement and
    reference motion in each phase.

    Use samples from trained encoders with labeled discrete phases.
    Consume a trainable encoder which computes canonical state internally.
    """

    def __init__(self, skel_name='./custom_dart_assets/walker2d_reg.skel', volume_scaling=1.0, max_phase=20,
                 init_phase=None,
                 target_velocity=1.0, velocity_reward_weight=0.05, joint_reward_weight=0.90,
                 orientation_reward_weight=0.05, disable_viewer=True, mirror=True, obs_dim=18, create_box=False):

        super().__init__(skel_name=skel_name, disable_viewer=disable_viewer, obs_dim=obs_dim)
        self.mirror = mirror
        self.disable_viewer = disable_viewer

        # kinetics variables
        self.volume_scaling = volume_scaling
        self.max_phase = max_phase
        self.init_phase = init_phase
        self.phase = init_phase
        self.target_velocity = target_velocity
        self.speed = 1.0
        self.counter = 0
        self.sim_freq = 25
        self.P = np.array([75, 75, 75, 75, 75, 75]) * self.volume_scaling ** 3
        self.D = self.P / 10.0
        self.time_limit = 400
        self.velocity_buffer = deque(maxlen=100)
        self.reward_buffer = deque(maxlen=100)
        self.observation_space = np.concatenate(
            [self.robot_skeleton.q[1:], self.robot_skeleton.dq, np.array([self.phase])])
        # reinforcement learning variables
        self.total_reward = 0
        self.velocity_reward_weight = velocity_reward_weight
        self.joint_reward_weight = joint_reward_weight
        self.orientation_reward_weight = orientation_reward_weight
        self.reward_weights = np.array([velocity_reward_weight, joint_reward_weight, orientation_reward_weight])
        # normalize since we can apply decays
        self.reward_weights /= np.sum(self.reward_weights)

        # load reference kinetic data
        with open('./data/walker2d_ref.pkl', 'rb') as f:
            kin_data = pickle.load(f)
        kin_data[:, 0] *= self.volume_scaling
        kin_data[:, 1] *= 0
        self.reference_position, self.reference_velocity = \
            expand_reference_motion(kin_data, sim_freq=self.sim_freq,
                                    max_phase=self.max_phase,
                                    speed=self.speed)

        self.box_skeleton = None
        if create_box:
            self.box_skeleton = self.dart_world.add_skeleton("./custom_dart_assets/box.sdf")

        # initial state
        self.set_initial_state(init_phase)
        self.posbefore = self.robot_skeleton.q[0]
        self.posafter = self.robot_skeleton.q[0]

        self.tau_history = []
        self.joint_history = []

    def set_box_position(self, x, y):
        if self.box_skeleton is None:
            raise RuntimeError("Box hasn't been created")

        q = self.box_skeleton.q
        q[0] = 0
        q[1] = 0
        q[2] = 0
        q[3] = x
        q[4] = y
        q[5] = 0
        dq = self.box_skeleton.dq
        dq[0] = 0
        dq[1] = 0
        dq[2] = 0
        dq[3] = 0
        dq[4] = 0
        dq[5] = 0
        self.box_skeleton.set_positions(q)
        self.box_skeleton.set_velocities(dq)

    def set_box_velocity(self, dx, dy):
        if self.box_skeleton is None:
            raise RuntimeError("Box hasn't been created")
        dq = np.asarray(self.box_skeleton.dq)
        dq[0] = 0
        dq[1] = 0
        dq[2] = np.random.normal(0, 1)
        dq[3] = dx
        dq[4] = dy
        dq[5] = 0
        self.box_skeleton.set_velocities(dq)

    def throw_box_from(self, x, y):
        if self.box_skeleton is None:
            raise RuntimeError("Box hasn't been created")
        self.set_box_position(x, y)
        robot_x = self.robot_skeleton.q[0]
        robot_y = self.robot_skeleton.bodynodes[2].com()[1]
        robot_dx = self.robot_skeleton.dq[0]

        delta_x = robot_x - x
        delta_y = robot_y - y
        t = np.sqrt(delta_y * 2 / (-9.81))
        vx = (delta_x + robot_dx * t) / t

        self.set_box_velocity(vx, 0)

    def set_initial_state(self, phase):
        if self.box_skeleton is not None:
            self.set_box_position(1, 1)
            self.set_box_velocity(0, 0)

        if phase is None:
            phase = np.random.randint(self.max_phase)

        self.set_phase(phase)
        qpos, qvel = self.get_kin_state()
        qpos[1] = self.robot_skeleton.q[1]
        qvel[0] = self.target_velocity
        self.set_state(qpos, qvel)

    def set_phase(self, phase):
        self.phase = phase
        self.phase %= self.max_phase
        self.counter += int(self.phase == 0)

    def get_kin_state_for_phase(self, phase):
        n_ref_frames, _ = self.reference_position.shape
        index = int(phase * n_ref_frames / self.max_phase)
        ref_pos = self.reference_position[index, :]
        ref_vel = self.reference_velocity[index, :]
        return ref_pos, ref_vel

    def get_kin_state(self):
        return self.get_kin_state_for_phase(self.phase)

    def get_kin_next_state(self):
        return self.get_kin_state_for_phase((self.phase + 1) % self.max_phase)

    def compute_joint_reward(self):
        joint_angles = self.robot_skeleton.q[3:]
        ref_pos, ref_vel = self.get_kin_state()
        ref_joint_angles = ref_pos[3:]
        ref_pos_penalty_weights = np.asarray([0.3, 0.1, 0.1, 0.3, 0.1, 0.1])
        ref_pos_errs = (joint_angles - ref_joint_angles) ** 2
        joint_penalty = ref_pos_penalty_weights @ ref_pos_errs
        joint_reward = np.exp(-joint_penalty * 10)
        return joint_reward

    def compute_orientation_reward(self):
        x, y, rotation = self.robot_skeleton.q[:3]
        orientation_penalty = rotation ** 2  # any deviation from 0
        orientation_reward = np.exp(-orientation_penalty)
        return orientation_reward

    def compute_velocity_reward(self):
        vel = self.get_com_velocity()
        velocity_reward = np.exp(-(self.target_velocity - vel) ** 2)
        return velocity_reward

    def compute_reward(self):
        velocity_reward = self.compute_velocity_reward()
        joint_reward = self.compute_joint_reward()
        orientation_reward = self.compute_orientation_reward()

        # compute final reward term
        reward_terms = np.array([velocity_reward, joint_reward, orientation_reward])
        # reward_terms = np.array([velocity_reward, 0, orientation_reward])
        # reward_terms /= np.sum(reward_terms)
        reward = reward_terms @ self.reward_weights

        return reward

    def get_joint_angle_and_velocity(self, idx):
        joint_angles = self.robot_skeleton.q
        joint_torques = self.robot_skeleton.dq
        return joint_angles[idx], joint_torques[idx]

    def step_simulation(self, action):
        """
        Do simulation for one simulation clock, using PD control and target reference position
        """
        # target = np.zeros(6)
        # a = np.copy(action)
        # for i in range(6):
        #     target[i] = a[i] + ref_pos[i + 3]

        target = action * 1.5
        # target = action + ref_pos[3:9]

        joint_angle_4, joint_velocity_4 = self.get_joint_angle_and_velocity(4)
        joint_angle_7, joint_velocity_7 = self.get_joint_angle_and_velocity(7)
        self.joint_history.append(np.asarray([joint_angle_4, joint_velocity_4, joint_angle_7, joint_velocity_7]))

        joint_angles = self.robot_skeleton.q[3:]
        joint_velocities = self.robot_skeleton.dq[3:]

        tau = np.zeros(self.robot_skeleton.ndofs)  # torque to apply at each simulation clock
        tau[3:] = self.P * (target - joint_angles) - self.D * joint_velocities
        tau = np.clip(tau, -150 * self.volume_scaling, 150 * self.volume_scaling)
        self.tau_history.append(tau)
        # print(tau)
        self.do_simulation(tau, 1)

    def step(self, a):
        """
        Action 'a' comes from the agent, who is agnostic to directions
        """
        if self.mirror and self.phase >= self.max_phase / 2:
            a = self.reflect_action(a)
        self.time += 1

        self.posbefore = self.robot_skeleton.q[0]

        self.do_dart_clocks(a)
        self.set_phase(self.phase + 1)

        self.posafter = self.robot_skeleton.q[0]

        # ref_pos, ref_vel = self.get_kin_state()
        # self.set_state(ref_pos, ref_vel)

        # common behavior for returning step() results
        done = self.is_done()
        ob = self._get_obs()
        reward = self.compute_reward()
        self.reward_buffer.append(reward)
        self.total_reward += reward

        self.energy += np.square(a).sum()
        return ob, reward, done, {}

    def do_dart_clocks(self, a):
        # run many simulation clocks.
        ref_pos, _ = self.get_kin_next_state()
        for i in range(self.sim_freq):
            self.step_simulation(a)  # this mutates robot_skeleton

    def is_done(self) -> bool:
        angle = self.robot_skeleton.q[2]
        height = self.robot_skeleton.bodynodes[2].com()[1]
        s = self.state_vector()

        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > .8 * self.volume_scaling) and (height < 2.0 * self.volume_scaling) and (abs(angle) < 1.0))
        if self.time_limit <= self.time:
            print("time reached")
            done = True

        # if self.compute_reward() < 0.6:
        #     done = True

        return done

    def _get_obs(self):
        state = super(Walker2DPhase, self)._get_obs()
        if self.mirror and self.phase >= self.max_phase / 2:
            state = self.reflect_state(state)
        phase = self.phase
        # if self.mirror:
        #     phase %= int(self.max_phase / 2)
        # phase /= float(self.max_phase)  # to appease neural network input
        extra = np.asarray([phase])
        return np.concatenate([state, extra])

    def reset_model(self):
        self.dart_world.reset()
        # clean-up routine for any reset
        self.counter = 0
        self.time = 0
        self.speed = 1
        self.set_initial_state(self.init_phase)
        return self._get_obs()

    def reflect_state(self, s):
        """
        Apply reflection so that both legs have same control.
        Assume state is purely control stuff.
        """
        s[2:8] = reflect_control_vector(s[2:8])
        s[11:17] = reflect_control_vector(s[11:17])
        return s

    def reflect_action(self, a):
        """
        Apply reflection so that both legs have same control.
        Assume state is purely control stuff.
        """
        return reflect_control_vector(a)

    def set_state_vector(self, state):
        qpos = np.concatenate([np.zeros(1), state[:8]])
        # self.robot_skeleton.q = qpos
        self.robot_skeleton.set_positions(qpos)
        self.robot_skeleton.dq = state[8:]
