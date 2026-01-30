'''
Miscellaneous functions used in developing/testing this project
'''

import warnings
import numpy as np
from os import path, makedirs
import json
from pathlib import Path
from typing import Literal
from itertools import islice
from PIL import Image

warnings.filterwarnings('ignore')

import gymnasium as gym
from gymnasium.envs.registration import register

# project libraries
from config import (HTML_TEMPLATE, RunConfig, RunConfigSpec, validate_config)
from environment import GridWorldEnv, save_gif
from localization import get_state_probabilities, ViterbiAlgorithm, map_state_sequence

def load_env(gridworld_dimensions=20, obstacle_ratio=0.4, pe=0.4):
	# we assume inputs have been validated with the config spec

	env_config = {
	'name': 'GridWorld-v0',
	'config': (GridWorldEnv, {'N':gridworld_dimensions, 'obstacle_ratio':obstacle_ratio, 'pe':pe})
	}
	
	register(id=env_config['name'], entry_point=env_config['config'][0], kwargs=env_config['config'][1])
	env = gym.make(env_config['name'], disable_env_checker=True)
	return env

def init_probabilities(num_states: int, 
					   mode: Literal['uniform', 'your-mode'] = 'uniform') -> np.ndarray:
	'''
	supported initialization modes: 
	- uniform: uniform probability across all states

	return:
		- initial state probs, shape (num_states,), where num_states is the N*N of the gridworld

	NOTE: Implement your own initialization
	'''
	if mode == 'uniform':
		init_probs = np.full(num_states, 1.0 / num_states, dtype=float)
		assert np.allclose(np.sum(init_probs), 1), 'initial probabilities do not sum to one'
	elif mode == 'your-mode':
		raise NotImplementedError('mode {} is not implemented.'.format(mode))
	else: 
		raise NotImplementedError('mode {} is not currently supported.'.format(mode))
	
	assert np.allclose(np.sum(init_probs), 1), 'initial probabilities do not sum to one'
	return init_probs

def run_sample(experiment_name: str, run_data: RunConfig, results_dir='results/'):
	
	# validate inputs 
	try:
		validate_config(run_data, RunConfigSpec)
	except ValueError as spec_error:
		print('Failed to load config. \n{}'.format(spec_error))
		return None
	
	# unpack run data struc
	gridworld_dimensions = run_data.gridworld_dimensions
	obstacle_ratio = run_data.obstacle_ratio
	max_steps = run_data.max_steps
	robot = run_data.robot_policy
	dummy = run_data.dummy
	init_prob_mode = run_data.init_prob_mode

	if dummy:
		print('WARNING: Using dummy get_state_probabilities(). You must implement your own.')
	
	# load environment 
	env = load_env(gridworld_dimensions, obstacle_ratio, pe=run_data.pe)

	# generate a new world
	observation, priviledged_info, model_info = env.reset()
	
	# init state probabilities 
	num_states = model_info['num_states']
	init_probs = init_probabilities(num_states=num_states, mode=init_prob_mode)

	obs_history = [observation]
	probs_history = [init_probs]
	viterbi = None
	if not dummy:
		# Initialize online Viterbi decoder (uses raw observation sequence).
		viterbi = ViterbiAlgorithm(model_info=model_info, init_probs=init_probs)
		# delta_last: (num_states,) after initialization
		viterbi.initialize(observation)

	# generate initial frame
	frame = env.render(state_probabilities=probs_history)
	frames = [frame]

	# simulate multiple steps
	for t in range(1,max_steps):
		# robot observes and takes an action
		action = robot.act(observation)
		
		# step environment with action and receive new observations
		observation, _, _, _, priviledged_info = env.step(action=action)

		# Preserve the raw observation for Viterbi before dummy overwrites it.
		raw_observation = observation # (4,)
		if not dummy:
			# Online Viterbi update using the raw 4-bit observation.
			viterbi.step(raw_observation)

		observation = observation if not dummy else priviledged_info # used to allow dummy to work with the same API
		
		## ----------------------------------------
		# TODO: implement get_state_probabilities()
		# First filter step uses the initial distribution; after that, use last belief.
		prev = init_probs if t == 1 else probs_history[-1]
		probs_t = get_state_probabilities(observation=observation,
										  model_info=model_info,
										  previous_state_dist=prev,
										  observation_history=obs_history,
										  dummy=dummy)
		
		## ----------------------------------------
		
		# check outputs
		assert probs_t.shape == (num_states,), 'probs_t wrong shape, must be of shape: {}, it has shape: {}'.format(init_probs, probs_t.shape)
		assert np.all(probs_t >= 0), 'probs_t contains negative probabilities'
		assert np.isclose(probs_t.sum(), 1.0, atol=1e-5), 'probs_t does not add to one'

		obs_history.append(observation)
		probs_history.append(probs_t)
		# generate new frame
		frame = env.render(state_probabilities=probs_history)
		frames.append(frame)

	viterbi_state_ids = None
	map_state_ids = None
	if not dummy:
		# Backtrack once at the end to recover the most likely path.
		viterbi_state_ids = viterbi.backtrack() # list length T (num_steps)
		# Compute MAP-from-marginals path with adjacency constraints.
		map_state_ids = map_state_sequence(probs_history=probs_history, model_info=model_info) # list length T

	# save frames
	results_save_path = path.join(results_dir, experiment_name)
	makedirs(results_save_path, exist_ok=True)
	for frame_idx, frame in enumerate(frames):
		f_name = 'frame-{}.png'.format(frame_idx)
		f_path = path.join(results_save_path, f_name)
		try:
			frame.save(f_path, format='PNG')
		except Exception as e:
			print('Error saving frame {}: {}'.format(f_name, e))

	if not dummy:
		# Render and save a Viterbi-only map (no heatmap).
		viterbi_img = env.render_gridworld(
			state_probabilities=None,
			viterbi_state_ids=viterbi_state_ids,
			viterbi_prefix_len=None,
			draw_viterbi_labels=True
		)
		viterbi_path = path.join(results_save_path, 'viterbi_path.png')
		Image.fromarray(viterbi_img.astype(np.uint8)).save(viterbi_path)
		# Render and save a MAP-only map (no heatmap).
		map_img = env.render_gridworld(
			state_probabilities=None,
			map_state_ids=map_state_ids,
			map_prefix_len=None,
			draw_map_labels=True
		)
		map_path = path.join(results_save_path, 'map_path.png')
		Image.fromarray(map_img.astype(np.uint8)).save(map_path)

	# save animation for README
	figures_save_path = 'figures'
	makedirs(figures_save_path, exist_ok=True)
	gif_result_save_path = path.join(figures_save_path,experiment_name+'.gif')
	save_gif(frames, save_to_path=gif_result_save_path)

	if not dummy:
		# Use the base env to avoid wrapper attribute shadowing.
		base_env = env.unwrapped if hasattr(env, 'unwrapped') else env
		# base_env.agent_pos_history: list length T, each (2,)
		# Ensure stored positions align with the number of rendered frames.
		assert len(base_env.agent_pos_history) == len(obs_history), 'agent_pos_history length does not match obs_history'
		# Build a second animation with Viterbi overlay on the heatmap frames.
		frames_viterbi = []
		for k in range(len(obs_history)):
			# Restore simulator state for frame k.
			base_env.agent_pos = base_env.agent_pos_history[k].copy().astype(np.int32) # (2,)
			base_env.sensor_obs = np.array(obs_history[k], dtype=np.int8)              # (4,)
			base_env.cur_step = k + 1
			# Draw the Viterbi path growing with time.
			prefix_len = k + 1
			frame_k = base_env.render(
				state_probabilities=probs_history[:k+1], # list length k+1, each (num_states,)
				viterbi_state_ids=viterbi_state_ids,      # list length T
				viterbi_prefix_len=prefix_len,
				draw_viterbi_labels=True
			)
			frames_viterbi.append(frame_k)

		viterbi_gif_path = path.join(figures_save_path, experiment_name + '_viterbi.gif')
		save_gif(frames_viterbi, save_to_path=viterbi_gif_path)
		# Build a third animation with MAP overlay on the heatmap frames.
		frames_map = []
		for k in range(len(obs_history)):
			# Restore simulator state for frame k.
			base_env.agent_pos = base_env.agent_pos_history[k].copy().astype(np.int32) # (2,)
			base_env.sensor_obs = np.array(obs_history[k], dtype=np.int8)              # (4,)
			base_env.cur_step = k + 1
			# Draw the MAP path growing with time.
			prefix_len = k + 1
			frame_k = base_env.render(
				state_probabilities=probs_history[:k+1], # list length k+1, each (num_states,)
				map_state_ids=map_state_ids,              # list length T
				map_prefix_len=prefix_len,
				draw_map_labels=True
			)
			frames_map.append(frame_k)

		map_gif_path = path.join(figures_save_path, experiment_name + '_map.gif')
		save_gif(frames_map, save_to_path=map_gif_path)
	return gif_result_save_path


def head_dict(d, n=10):
	''' return a small dict with the first n items of d'''
	return dict(islice(d.items(), n))

def preview_dict(name, parent_dict, n=3):
	d = parent_dict[name]
	header = '{}: {} keys (showing first {})'.format(name, len(d), min(n, len(d)))
	items = list(islice(d.items(), n))
	body = '\t{' + ', '.join(['{}: {}'.format(k, v) for k, v in items])
	if len(d) > n:
		body += ', ...'
	body += '}'
	print(header)
	print(body)

'''
Visulizer HTML index generator 

I used some LLM help to generate the HTLM template/regeneration below.
The idea is to have a static HTML index file that is used by Github Pages.

Given that index.html must have static/pre-defined values for the dropdown for all
this work I resorted to dynamically populating them based on the contents of "/results".
'''

def _find_result_directories(results_dir):
	''' return sorted list of result/ subdirectories that contain at least one frame '''
	if not results_dir.exists():
		raise SystemExit('results directory not found: {0}'.format(results_dir))

	dirs = []
	for entry in sorted(results_dir.iterdir()):
		if not entry.is_dir():
			continue

		has_png = False
		for f in entry.iterdir():
			if f.is_file() and f.suffix.lower() == '.png':
				has_png = True
				break

		if has_png:
			dirs.append(entry.name)

	if not dirs:
		raise SystemExit('No result subdirectories with PNG frames found in {0}'.format(results_dir))

	return dirs

def generate_index():
	ROOT = Path(__file__).parent
	RESULTS_DIR = ROOT / 'results'
	OUTPUT_HTML = ROOT / 'index.html'

	dirs = _find_result_directories(RESULTS_DIR)
	default_dir = dirs[0]

	# build <option> list for dropdown
	options_lines = []
	for d in dirs:
		line = '            <option value="{0}">{0}</option>'.format(d)
		options_lines.append(line)
	options_html = '\n'.join(options_lines)
	html = HTML_TEMPLATE.replace('__OPTIONS__', options_html)
	html = html.replace('__DEFAULT_DIR__', default_dir)

	OUTPUT_HTML.write_text(html, encoding='utf-8')
	print('Wrote index.html with {} runs.'.format(len(dirs)))
