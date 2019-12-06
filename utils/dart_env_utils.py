from typing import List

import numpy as np
from abc import abstractmethod

from gym.envs.dart import DartEnv
from gym.utils import EzPickle

from utils.utils import reflect_control_vector


class EnvStateGetter:

    @abstractmethod
    def get_state(self, env: DartEnv):
        raise NotImplementedError


class EnvStateGetterDummy(EnvStateGetter):

    def get_state(self, env: DartEnv):
        return None


class EnvStateGetterQdq(EnvStateGetter):

    def __init__(self, q_start_idx=1):
        self.q_start_idx = q_start_idx

    def get_state(self, env: DartEnv):
        state = np.concatenate([
            env.robot_skeleton.q[self.q_start_idx:],
            np.clip(env.robot_skeleton.dq, -10, 10)
        ])
        return state


class EnvStateGetterMirrored(EnvStateGetterQdq):

    def __init__(self, child_state_getter: EnvStateGetter, q_range: np.ndarray, dq_range: np.ndarray):
        self.child_state_getter = child_state_getter
        self.q_range = q_range
        self.dq_range = dq_range

    def get_state(self, env: DartEnv):
        state = self.child_state_getter.get_state(env)
        if env.phase >= env.max_phase / 2:
            state = self._mirror_state(state)
        return state

    def _mirror_state(self, state):
        state[self.q_range] = reflect_control_vector(state[self.q_range])
        state[self.dq_range] = reflect_control_vector(state[self.dq_range])
        return state


class EnvStateGetterPhase(EnvStateGetter):

    def __init__(self, child_state_getter: EnvStateGetter):
        self.child_state_getter = child_state_getter

    def get_state(self, env: DartEnv):
        state = self.child_state_getter.get_state(env)
        phase = env.phase
        return np.concatenate([state, np.asarray([phase])])


class EnvRewardGetter:

    def __init__(self, weight):
        self.weight = weight

    @abstractmethod
    def get_reward(self, env: DartEnv):
        raise NotImplementedError


class EnvRewardGetterSum(EnvRewardGetter):

    def __init__(self, reward_getters: List[EnvRewardGetter]):
        super().__init__(1.)
        self.reward_getters = reward_getters

    def get_reward(self, env: DartEnv):
        rewards = [reward_getter.get_reward(env) for reward_getter in self.reward_getters]
        return np.sum(rewards)


class EnvRewardGetterVelocity(EnvRewardGetter):

    def __init__(self, dq_idx, weight):
        super().__init__(weight)
        self.dq_idx = dq_idx

    def get_reward(self, env: DartEnv):
        dq = env.robot_skeleton.dq[self.dq_idx]
        # if dq > env.target_velocity:
        #     vel_penalty = 0
        # else:
        #     vel_penalty = (env.target_velocity - dq) ** 2
        vel_penalty = (env.target_velocity - dq) ** 2
        vel_reward = np.exp(-vel_penalty)
        return vel_reward * self.weight


class EnvRewardGetterOrientation(EnvRewardGetter):

    def __init__(self, q_idx, weight):
        super().__init__(weight)
        self.q_idx = q_idx

    def get_reward(self, env: DartEnv):
        orientation = env.robot_skeleton.q[self.q_idx]
        orientation_penalty = orientation ** 2
        orientation_reward = np.exp(-orientation_penalty)
        return orientation_reward * self.weight


class EnvRewardGetterReferenceMotion(EnvRewardGetter):

    def __init__(self, reference_poses: np.ndarray, q_idx: np.ndarray, penalty_weights: np.ndarray, weight):
        super().__init__(weight)
        self.penalty_weights = penalty_weights
        self.q_idx = q_idx
        self.reference_poses = reference_poses

    def get_reward(self, env: DartEnv):
        n_ref_frames, _ = self.reference_poses.shape
        index = int(env.phase * n_ref_frames / env.max_phase)
        ref_pose_for_phase = self.reference_poses[index, self.q_idx]
        pose = env.robot_skeleton.q[self.q_idx]
        ref_pose_penalty = self.penalty_weights @ ((ref_pose_for_phase - pose) ** 2)
        ref_pose_reward = np.exp(-10 * ref_pose_penalty)
        return ref_pose_reward * self.weight


class EnvDoneGetter:

    def __init__(self, time_limit):
        self.time_limit = time_limit

    @abstractmethod
    def get_done(self, env: DartEnv):
        raise NotImplementedError


class EnvDoneGetterFalling(EnvDoneGetter):

    def __init__(self, orientation_idx, time_limit):
        super().__init__(time_limit)
        self.orientation_idx = orientation_idx

    def get_done(self, env: DartEnv):
        s = env._get_obs()
        angle = env.robot_skeleton.q[self.orientation_idx]
        height = env.robot_skeleton.bodynodes[2].com()[1]
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > .8) and (height < 2.0) and (abs(angle) < 1.0))

        # print(height, angle)
        if self.time_limit <= env.time:
            # print("time reached")
            done = True
        return done


class EnvActionApplier:

    @abstractmethod
    def apply_action(self, env: DartEnv, action: np.ndarray):
        raise NotImplementedError


class EnvActionApplierPDControllerReferenceMotion(EnvActionApplier):

    def __init__(self, sim_freq, p_gain, d_gain, action_dim, torque_limit, reference_poses, q_idx, action_multiplier,
                 debug):
        self.debug = debug
        self.action_multiplier = action_multiplier
        self.q_idx = q_idx
        self.reference_poses = reference_poses
        self.torque_limit = torque_limit
        self.P = np.array([p_gain] * action_dim)
        self.D = np.array([d_gain] * action_dim)
        self.sim_freq = sim_freq
        self.tau_history = []
        self.target_history = []
        self.joint_angle_history = []
        self.joint_velocity_history = []
        self.ref_pose_history = []

    def apply_action(self, env: DartEnv, action: np.ndarray):
        # target = action
        # target = action * 3.0
        n_ref_frames, _ = self.reference_poses.shape
        index = int(env.phase * n_ref_frames / env.max_phase)
        ref_pose_for_phase = self.reference_poses[index, self.q_idx]
        target = action * self.action_multiplier + ref_pose_for_phase
        for _ in range(self.sim_freq):
            joint_angles = env.robot_skeleton.q[3:]
            joint_velocities = env.robot_skeleton.dq[3:]

            tau = np.zeros(env.robot_skeleton.ndofs)  # torque to apply at each simulation clock
            tau[3:] = self.P * (target - joint_angles) - self.D * joint_velocities
            tau = np.clip(tau, -self.torque_limit, self.torque_limit)
            env.do_simulation(tau, 1)

        if self.debug:
            self.tau_history.append(tau)
            self.ref_pose_history.append(ref_pose_for_phase)
            self.target_history.append(target)
            self.joint_angle_history.append(np.asarray(env.robot_skeleton.q))
            self.joint_velocity_history.append(np.asarray(env.robot_skeleton.dq))


class EnvActionApplierPDController(EnvActionApplier):

    def __init__(self, sim_freq, p_gain, d_gain, action_dim, torque_limit, q_idx, action_multiplier, debug):
        self.debug = debug
        self.action_multiplier = action_multiplier
        self.q_idx = q_idx
        self.torque_limit = torque_limit
        self.P = np.array([p_gain] * action_dim)
        self.D = np.array([d_gain] * action_dim)
        self.sim_freq = sim_freq
        self.tau_history = []
        self.target_history = []
        self.joint_angle_history = []
        self.joint_velocity_history = []
        self.ref_pose_history = []

    def apply_action(self, env: DartEnv, action: np.ndarray):
        # target = action
        target = action * self.action_multiplier
        # self.target_history.append(target)
        for _ in range(self.sim_freq):

            joint_angles = env.robot_skeleton.q[3:]
            joint_velocities = env.robot_skeleton.dq[3:]

            tau = np.zeros(env.robot_skeleton.ndofs)  # torque to apply at each simulation clock
            tau[3:] = self.P * (target - joint_angles) - self.D * joint_velocities
            tau = np.clip(tau, -self.torque_limit, self.torque_limit)
            env.do_simulation(tau, 1)

            if self.debug:
                self.target_history.append(target)
                self.joint_angle_history.append(np.asarray(env.robot_skeleton.q))
                self.joint_velocity_history.append(np.asarray(env.robot_skeleton.dq))
                self.tau_history.append(tau)


class EnvActionApplierPDControllerMirrored(EnvActionApplier):

    def __init__(self, child_action_applier):
        self.child_action_applier = child_action_applier

    def apply_action(self, env: DartEnv, action: np.ndarray):
        if env.phase >= env.max_phase / 2:
            action = reflect_control_vector(action)
        self.child_action_applier.apply_action(env, action)


# class EnvActionApplierPendulum2D(EnvActionApplier):
#
#     def __init__(self, mass, string_length, k, omega, DT):
#         self.mass = mass
#         self.string_length = string_length
#         self.k = k
#         self.omega = omega
#         self.DT = DT
#
#     def compute_target_angular_velocity(self, phase):
#         return -self.k * self.omega * np.sin(self.omega * phase) / self.DT
#
#     def compute_target_angle(self, phase):
#         return self.k * np.cos(self.omega * phase)
#
#     def apply_action(self, env: DartEnv, action: np.ndarray):
#         angle, angular_velocity, phase = env._get_obs()
#         target_angle = self.compute_target_angle(env.phase)
#         target_angular_velocity = self.compute_target_angular_velocity(env.phase)
#         a = 50 * (target_angle - angle) + \
#             100 * (target_angular_velocity - angular_velocity)
#         a /= (self.mass * self.string_length ** 2)


class EnvResetter:

    @abstractmethod
    def reset(self, env: DartEnv) -> np.ndarray:
        raise NotImplementedError


class EnvResetterZero(EnvResetter):

    def reset(self, env: DartEnv):
        env.time = 0
        qpos = np.zeros(env.robot_skeleton.ndofs)
        qvel = np.zeros(env.robot_skeleton.ndofs)
        env.set_state(qpos, qvel)
        return env._get_obs()


class EnvResetterReferenceMotion(EnvResetter):

    def __init__(self, reference_poses, reference_velocities):
        self.reference_poses = reference_poses
        self.reference_velocities = reference_velocities

    def reset(self, env: DartEnv):
        init_phase = env.init_phase
        if init_phase is None:
            init_phase = np.random.randint(0, env.max_phase)

        n_ref_frames, _ = self.reference_poses.shape
        index = int(init_phase * n_ref_frames / env.max_phase)

        env.time = 0
        env.phase = init_phase
        qpos = self.reference_poses[index, :]
        qvel = self.reference_velocities[index, :]
        qvel[0] = env.target_velocity
        env.set_state(qpos, qvel)
        return env._get_obs()


class DartEnvBasic(DartEnv, EzPickle):

    def step(self, action):
        self.action_applier.apply_action(self, action)
        self.time += 1
        # common behavior for returning step() results
        done = self.is_done()
        state = self._get_obs()
        reward = self.compute_reward()

        return state, reward, done, {}

    def reset_model(self):
        self.dart_world.reset()
        if self.box_skeleton is not None:
            self.set_box_position(-1, 1)
            self.set_box_velocity(0, 0)

        self.time = 0
        return self.resetter.reset(self)

    def __init__(self, model_paths, frame_skip, observation_size, action_bounds, disable_viewer, simulation_frequency,
                 state_getter: EnvStateGetter, reward_getter: EnvRewardGetter, done_getter: EnvDoneGetter,
                 action_applier: EnvActionApplier, resetter: EnvResetter, create_box=False):
        DartEnv.__init__(self, model_paths, frame_skip, observation_size, action_bounds, disableViewer=disable_viewer)
        try:
            self.dart_world.set_collision_detector(3)
        except Exception as e:
            print('Does not have ODE collision detector, reverted to bullet collision detector')
            self.dart_world.set_collision_detector(2)

        EzPickle.__init__(self)

        self.state_getter = state_getter
        self.reward_getter = reward_getter
        self.done_getter = done_getter
        self.action_applier = action_applier
        self.resetter = resetter

        self.simulation_frequency = simulation_frequency
        _, self.action_dim = action_bounds.shape
        self.target_velocity = 1
        self.time = 0
        self.track_skeleton_id = 1  # usually 0 is ground and 1 is walker

        self.box_skeleton = None
        if create_box:
            self.box_skeleton = self.dart_world.add_skeleton("./custom_dart_assets/box.sdf")

        self.reset()

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
        self.box_skeleton.set_positions(q)

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

    def viewer_setup(self):
        self._get_viewer().scene.tb._set_orientation(0, 0)
        self._get_viewer().scene.tb.trans[2] = -7.5

    def is_done(self):
        return self.done_getter.get_done(self)

    def _get_obs(self):
        return self.state_getter.get_state(self)

    def compute_reward(self):
        return self.reward_getter.get_reward(self)

    def set_pose(self, pose: np.ndarray):
        self.robot_skeleton.set_positions(pose)

    def set_vel(self, vel):
        self.robot_skeleton.set_velocities(vel)

    def get_ndofs(self):
        return self.robot_skeleton.ndofs

    def set_state_vector(self, state):
        self.robot_skeleton.set_positions(state[0:int(len(state) / 2)])
        self.robot_skeleton.set_velocities(state[int(len(state) / 2):])


class DartEnvPhase(DartEnvBasic):

    def __init__(self, model_paths, frame_skip, observation_size, action_bounds, disable_viewer, simulation_frequency,
                 state_getter, reward_getter, done_getter, action_applier, resetter, max_phase, init_phase, create_box):
        self.max_phase = max_phase
        self.init_phase = init_phase
        self.phase = self.init_phase

        super().__init__(model_paths, frame_skip, observation_size, action_bounds, disable_viewer, simulation_frequency,
                         state_getter, reward_getter, done_getter, action_applier, resetter, create_box)

    def step(self, action):
        self.action_applier.apply_action(self, action)
        self.time += 1
        self.phase += 1
        self.phase %= self.max_phase
        # common behavior for returning step() results
        done = self.is_done()
        state = self._get_obs()
        reward = self.compute_reward()

        return state, reward, done, {}
