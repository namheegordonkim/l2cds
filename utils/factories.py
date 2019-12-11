import os
import pickle
from abc import abstractmethod
from xml.etree import ElementTree as ET

import numpy as np

from characters.pendulum2d import BLACK, Pendulum2D
from utils.dart_env_utils import EnvRewardGetterSum, EnvRewardGetterVelocity, EnvRewardGetterOrientation, \
    EnvRewardGetterReferenceMotion, EnvDoneGetter, EnvActionApplierPDController, \
    EnvResetterReferenceMotion, DartEnvBasic, DartEnvPhase, EnvDoneGetterFalling, EnvActionApplierPDControllerMirrored, \
    EnvStateGetterPhase, EnvActionApplierPDControllerReferenceMotion, EnvStateGetterQdq, \
    EnvStateGetterMirrored, EnvRewardGetter, EnvActionApplier, EnvResetterZero, EnvStateGetterDummy
from utils.utils import interpolate_arrays


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

        with open("./data/walker2d_reg_ref.pkl", 'rb') as f:
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

    def __init__(self, skel_file, sim_freq, max_phase, disable_viewer, create_box, mirror, debug):
        self.debug = debug
        self.max_phase = max_phase
        self.sim_freq = sim_freq
        self.disable_viewer = disable_viewer
        self.action_dim = 8
        self.state_dim = 22
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
            EnvRewardGetterOrientation(2, 0.15),
            EnvRewardGetterReferenceMotion(reference_poses, pose_q_idx, ref_pos_penalty_weights, 0.80)
        ])

        done_getter = EnvDoneGetterFalling(2, 400)
        action_applier = EnvActionApplierPDControllerReferenceMotion(self.sim_freq, 75, 7.5, self.action_dim, 300,
                                                                     reference_poses, pose_q_idx, 1, self.debug)
        if self.mirror:
            action_applier = EnvActionApplierPDControllerMirrored(action_applier)

        resetter = EnvResetterReferenceMotion(reference_poses, reference_velocities)

        env = DartEnvPhase(
            asset_fullpath, 1, self.state_dim, action_bounds, self.disable_viewer, self.sim_freq,
            state_getter, reward_getter, done_getter, action_applier, resetter, self.max_phase, None, self.create_box
        )
        return env


def expand_reference_motion(kin_data: np.ndarray, sim_freq=25, max_phase=20, speed=1.0):
    t = 0.002 * sim_freq * max_phase / 100.0

    n, d = kin_data.shape

    # get position
    interpolated_arrays = []
    for i in range(n - 1):
        current = kin_data[i, :]
        next = kin_data[i + 1, :]
        interpolated = interpolate_arrays(current, next)
        interpolated_arrays.append(interpolated)
    expanded_reference_position = np.concatenate(interpolated_arrays, axis=0)

    n_ref_frames, _ = expanded_reference_position.shape
    expanded_reference_velocity = np.zeros([n_ref_frames, d])

    # get velocity
    for i in range(n - 1):
        expanded_reference_velocity[i, :] = (expanded_reference_position[i + 1, :] -
                                             expanded_reference_position[i, :]) / t
        expanded_reference_velocity[i, 0] *= speed

    expanded_reference_velocity[-1, :] = (expanded_reference_position[0, :] - expanded_reference_position[-1, :]) / t
    expanded_reference_velocity[-1, 0] = 0.1 / 10.0 / t * speed

    return expanded_reference_position, expanded_reference_velocity


def get_env_factory(config_dict, config_name, disable_viewer, create_box):
    if config_name == "ostrich2d":
        factory = DartEnvFactoryOstrich2D(config_dict["skel_file"], 25, 20, disable_viewer=disable_viewer,
                                          create_box=create_box, mirror=True, debug=False)
    elif config_name == "walker2d_reg":
        factory = Walker2DPhaseFactory(config_dict["skel_file"], 1.0, None, disable_viewer, True, create_box, False)
    else:
        factory = Pendulum2DFactory("./custom_dart_assets/pendulum2d.skel", "./artifacts/pendulum2d_generated.skel", 1.,
                                    disable_viewer, True)
    return factory


class Pendulum2DFactory(EnvFactory):

    def __init__(self, skel_template, artifact_name, damping, disable_viewer=True, output_phase=True,
                 render_color=BLACK):
        self.render_color = render_color
        self.output_phase = output_phase
        self.skel_template = skel_template
        self.artifact_name = artifact_name
        self.damping = damping
        self.disable_viewer = disable_viewer

    def generate_pendulum2d_skel(self, skel_template='./custom_dart_assets/pendulum2d.skel', damping=1.0,
                                 save_path='./artifacts/asdf.skel'):
        etree = ET.parse(skel_template)
        root = etree.getroot()

        pole_joint_elem = root.find(".//joint[@name='j_pole']")
        damping_elem = pole_joint_elem.find(".//damping")
        damping_elem.text = str(damping)

        etree._setroot(root)
        etree.write(save_path)

    def make_env(self):
        self.generate_pendulum2d_skel(save_path=self.artifact_name, damping=self.damping)
        return Pendulum2D(skel_name=self.artifact_name, disable_viewer=self.disable_viewer,
                          output_phase=self.output_phase, render_color=self.render_color)