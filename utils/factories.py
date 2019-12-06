import os
import pickle
from abc import abstractmethod

import numpy as np

from utils.dart_env_utils import EnvRewardGetterSum, EnvRewardGetterVelocity, EnvRewardGetterOrientation, \
    EnvRewardGetterReferenceMotion, EnvDoneGetter, EnvActionApplierPDController, \
    EnvResetterReferenceMotion, DartEnvBasic, DartEnvPhase, EnvDoneGetterFalling, EnvActionApplierPDControllerMirrored, \
    EnvStateGetterPhase, EnvActionApplierPDControllerReferenceMotion, EnvStateGetterQdq, \
    EnvStateGetterMirrored, EnvRewardGetter, EnvActionApplier, EnvResetterZero, EnvStateGetterDummy
from utils.utils import expand_reference_motion


class EnvFactory:

    @abstractmethod
    def make_env(self):
        raise NotImplementedError


class Walker2DPhaseFactory(EnvFactory):

    def __init__(self, skel, scale, i, disable_viewer, mirror, create_box, debug):
        self.create_box = create_box
        self.mirror = mirror
        self.skel = skel
        self.scale = scale
        self.init_phase = i
        self.disable_viewer = disable_viewer
        self.debug = debug

    def make_env(self):
        asset_fullpath = os.path.abspath(self.skel)
        action_bounds = np.array([[1.0] * 6, [-1.0] * 6])

        with open("./data/walker2d_ref.pkl", 'rb') as f:
            raw_reference_keyframes = pickle.load(f)

        reference_poses, reference_velocities = expand_reference_motion(raw_reference_keyframes, 25,
                                                                        20)
        _, ndofs = reference_poses.shape
        ref_pos_penalty_weights = np.asarray([0.3, 0.1, 0.1, 0.3, 0.1, 0.1])
        ref_pos_penalty_weights /= np.sum(ref_pos_penalty_weights)

        q_range = np.arange(2, 8)
        dq_range = np.arange(11, 17)
        state_getter = EnvStateGetterQdq()
        state_getter = EnvStateGetterPhase(state_getter)
        if self.mirror:
            state_getter = EnvStateGetterMirrored(state_getter, q_range, dq_range)

        pose_q_idx = np.arange(3, ndofs)
        reward_getter = EnvRewardGetterSum([
            EnvRewardGetterVelocity(0, 0.05),
            EnvRewardGetterOrientation(2, 0.05),
            EnvRewardGetterReferenceMotion(reference_poses, pose_q_idx, ref_pos_penalty_weights, 0.90)
        ])

        done_getter = EnvDoneGetterFalling(2, 400)
        action_applier = EnvActionApplierPDController(25, 75, 7.5, 6, 300,
                                                      pose_q_idx, 1.5, self.debug)
        if self.mirror:
            action_applier = EnvActionApplierPDControllerMirrored(action_applier)

        resetter = EnvResetterReferenceMotion(reference_poses, reference_velocities)

        env = DartEnvPhase(
            asset_fullpath, 1, 18, action_bounds, self.disable_viewer, 25,
            state_getter, reward_getter, done_getter, action_applier, resetter, 20, None, self.create_box
        )
        return env


class DartEnvFactory(EnvFactory):

    def __init__(self, skel_file, state_dim, action_dim, disable_viewer, create_box):
        self.disable_viewer = disable_viewer
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.skel_file = skel_file
        self.create_box = create_box

    def make_env(self):
        asset_fullpath = os.path.abspath(self.skel_file)
        control_bounds = np.array([[1.0] * self.action_dim, [-1.0] * self.action_dim])

        state_getter = EnvStateGetterDummy()
        action_applier = EnvActionApplier()
        reward_getter = EnvRewardGetter(0)
        done_getter = EnvDoneGetter(400)
        resetter = EnvResetterZero()

        env = DartEnvBasic(asset_fullpath, 1, self.state_dim, control_bounds, self.disable_viewer, 25, state_getter,
                           reward_getter, done_getter, action_applier, resetter, self.create_box)
        return env


class DartEnvFactoryOstrich2D(EnvFactory):

    def __init__(self, skel_file, state_dim, action_dim, sim_freq, max_phase, disable_viewer, create_box, mirror, debug):
        self.debug = debug
        self.max_phase = max_phase
        self.sim_freq = sim_freq
        self.disable_viewer = disable_viewer
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.skel_file = skel_file
        self.create_box = create_box
        self.mirror = mirror

    def make_env(self):
        asset_fullpath = os.path.abspath(self.skel_file)
        action_bounds = np.array([[1.0] * self.action_dim, [-1.0] * self.action_dim])

        with open("./data/ostrich2d_ref.pkl", 'rb') as f:
            raw_reference_keyframes = pickle.load(f)

        reference_poses, reference_velocities = expand_reference_motion(raw_reference_keyframes, self.sim_freq,
                                                                        self.max_phase)
        _, ndofs = reference_poses.shape
        ref_pos_penalty_weights = np.asarray([0.3, 0.1, 0.1, 0.01, 0.3, 0.1, 0.1, 0.01])
        ref_pos_penalty_weights /= np.sum(ref_pos_penalty_weights)

        q_range = np.arange(2, 10)
        dq_range = np.arange(13, 21)
        state_getter = EnvStateGetterQdq()
        state_getter = EnvStateGetterPhase(state_getter)
        if self.mirror:
            state_getter = EnvStateGetterMirrored(state_getter, q_range, dq_range)

        pose_q_idx = np.arange(3, ndofs)
        reward_getter = EnvRewardGetterSum([
            EnvRewardGetterVelocity(0, 0.05),
            EnvRewardGetterOrientation(2, 0.05),
            EnvRewardGetterReferenceMotion(reference_poses, pose_q_idx, ref_pos_penalty_weights, 0.90)
        ])

        done_getter = EnvDoneGetterFalling(2, 400)
        action_applier = EnvActionApplierPDControllerReferenceMotion(self.sim_freq, 50, 5.0, self.action_dim, 300,
                                                                     reference_poses, pose_q_idx, 1, self.debug)
        if self.mirror:
            action_applier = EnvActionApplierPDControllerMirrored(action_applier)

        resetter = EnvResetterReferenceMotion(reference_poses, reference_velocities)

        env = DartEnvPhase(
            asset_fullpath, 1, self.state_dim, action_bounds, self.disable_viewer, self.sim_freq,
            state_getter, reward_getter, done_getter, action_applier, resetter, self.max_phase, None, self.create_box
        )
        return env
