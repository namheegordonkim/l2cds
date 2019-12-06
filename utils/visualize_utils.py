from collections import deque

import numpy as np
import pygame

from characters.pendulum2d import PhasedActor

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
PURPLE = (255, 50, 255)
YELLOW = (252, 186, 3)
CYAN = (3, 252, 248)
DARK_BLUE = (10, 0, 41)
DARK_RED = (41, 0, 0)
view_scale = np.asarray([60, 20])


class PhaseSpaceDiagram:

    def __init__(self, center, dot_radius=3):
        self.view_scale = view_scale
        self.dot_radius = dot_radius
        self.center = center  # center
        self.buffers = []  # list of deques
        self.colors = []  # list of tuples
        self.k = 2 / 3 * np.pi  # amplitude
        self.string_length = 0.6
        self.omega = np.sqrt(9.81 / self.string_length)  # frequency coefficient
        # self.omega = 1  # frequency coefficient

        timesteps = np.linspace(0, 2 * np.pi, 100) / self.omega

        # populate reference ellipse
        self.reference_ellipse_pts = []
        thetas_ellipse = self.k * np.cos(self.omega * timesteps)
        theta_dots_ellipse = -self.k * self.omega * np.sin(self.omega * timesteps)

        for theta, theta_dot in zip(thetas_ellipse, theta_dots_ellipse):
            draw_pos = np.asarray(self.center) + np.asarray([theta, theta_dot]) * self.view_scale
            self.reference_ellipse_pts.append(draw_pos)

    def push(self, pos, buffer):
        corrected_pos = pos.copy()[:2]
        corrected_pos[1] = -corrected_pos[1]  # coordinate systems are different
        draw_pos = np.asarray(self.center) + np.asarray(corrected_pos) * self.view_scale
        buffer.append(draw_pos)

    def render(self, screen):
        # draw axes
        x, y = self.center
        pygame.draw.line(screen, BLACK, [0, y], [screen.get_width(), y])
        pygame.draw.line(screen, BLACK, [x, 0], [x, screen.get_height()])

        # draw reference limit cycle
        # for ref_pos1, ref_pos2 in zip(self.reference_ellipse_pts, self.reference_ellipse_pts[1:]):
        #     color = np.asarray(BLACK)
        #     pygame.draw.line(screen, BLACK, ref_pos1, ref_pos2, 2)

        # draw actuated limit cycle
        for buffer, color in zip(self.buffers, self.colors):
            self.draw_buffer_as_lines(screen, color, buffer)
            if len(buffer) > 0:
                pos = buffer[-1]
                pos = int(pos[0]), int(pos[1])
                pygame.draw.circle(screen, color, pos, 10)

    def draw_buffer_as_lines(self, screen, line_color, buffer):
        # intensity = 0.5
        draw_list = list(buffer)
        for draw_pos1, draw_pos2 in zip(draw_list, draw_list[1:]):
            color = line_color
            # color = mix_colors(line_color, WHITE, intensity)
            # pygame.draw.circle(screen, color, draw_pos1.astype(int), self.dot_radius)
            pygame.draw.line(screen, color, draw_pos1, draw_pos2, 1)
            x_diff, y_diff = draw_pos2 - draw_pos1
            angle = np.arctan2(y_diff, x_diff) + np.pi / 2
            self.draw_equilateral(screen, draw_pos2, color, angle, width=5, height=10)
            # intensity *= 0.5

    def make_new_buffer(self, color, maxlen=100):
        buffer = deque(maxlen=maxlen)
        self.add_buffer(buffer, color)

    def add_buffer(self, buffer, color):
        self.buffers.append(buffer)
        self.colors.append(color)

    def draw_equilateral(self, screen, pos, color, angle, width, height):
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

        left_ear = (rotation @ np.asarray([-width / 2, height])) + pos
        right_ear = (rotation @ np.asarray([width / 2, height])) + pos

        pygame.draw.polygon(screen, color, (pos, left_ear, right_ear))


class Game:

    def __init__(self, diagram: PhaseSpaceDiagram):
        self.envs = []
        self.solvers = []
        self.diagram = diagram
        self.perturb_intervals = []
        self.state_converters = []
        self.envs_to_convert = []

    def add_env_and_solver(self, env, solver: PhasedActor, color, perturb_interval=0, maxlen=100):
        self.envs.append(env)
        self.solvers.append(solver)
        self.diagram.make_new_buffer(color, maxlen)
        self.perturb_intervals.append(perturb_interval)

    def add_state_converter(self, env, state_converter, color, maxlen=100):
        self.state_converters.append(state_converter)
        self.diagram.make_new_buffer(color, maxlen)
        self.envs_to_convert.append(env)

    def reset_all(self):
        states = []
        for env in self.envs:
            state = env.reset()
            states.append(state)
        return states

    def act_all(self, states):
        actions = []
        for state, solver, env in zip(states, self.solvers, self.envs):
            phase = env.get_phase()
            action = solver.get_action(state, phase)
            actions.append(action)
        return actions

    def step_all(self, states, actions):
        states_new = []
        states_to_draw = []
        for i, (state, action, env, solver, buffer) in enumerate(
                zip(states, actions, self.envs, self.solvers, self.diagram.buffers)):
            states_to_draw.append(state)
            state_new, _, _, _ = env.step(action)
            states_new.append(state_new)

        for env, converter in zip(self.envs_to_convert, self.state_converters):
            state = env.get_state_vector()[:2]
            converted_state = converter.convert_state(state)
            states_to_draw.append(converted_state)

        for state, buffer in zip(states_to_draw, self.diagram.buffers):
            self.diagram.push(state[:2], buffer)

        return states_new

    def render_onto(self, screen):
        self.diagram.render(screen)

    def perturb_all(self, states):
        states_new = []
        for env, state, perturb_interval in zip(self.envs, states, self.perturb_intervals):
            if perturb_interval == 0:
                pass
            elif env.time % perturb_interval == 0:
                state[1] += 1 * np.random.standard_normal()
                env.set_state(np.array([state[0]]), np.array([state[1]]))
            states_new.append(state)
        return states_new


def mix_colors(color1, color2, intensity):
    color = np.asarray(color1) + np.asarray(color2) * intensity
    color = np.maximum(color, np.zeros_like(color))
    color = np.minimum(color, 255 * np.ones_like(color))
    return color