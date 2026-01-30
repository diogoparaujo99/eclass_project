'''
Self-Localization modeled as a Hidden Markov Model (HMM)
'''

import numpy as np
from typing import List, Optional, Tuple

def get_state_probabilities(observation: Tuple[int, int, int, int],
							model_info: dict,
							previous_state_dist: np.ndarray,
							observation_history: Optional[List[Tuple[int, int, int, int]]] = None,
							dummy: bool = True) -> np.ndarray:
	'''
	TODO: implement your own state probabilities estimation, replace the placehold return with yours

	suggested inputs, feel free to modify/ignore:
		
		observation: tuple 
			- current sensor reading (N, E, S, W) as a binary tuple, i.e. (0, 1, 0, 1)

		model_info: dictionary of model information provided by the environment. Items inside:
			- transition_matrix: numpy array of shape (N*N, N*N)
				- state transition matrix P(x_t | x_{t-1})
			- observation_matrix: numpy array of shape (N*N, Z)	
				- observation likelihoods P(o_t | x_t)
			- grid_shape: (N,N) tuple of the gridworld dimensions
			- num_states: N*N total number of possible states (includes obstacles)
			- obs_id_lookup: map of raw obs tuple (N, E, S, W) to column index in observation matrix
			- obs_id_reverse_lookup: map of column index in observation matrix to raw obs tuple (N, E, S, W)
			- state_id_to_xy: map of state row/col id (equivalent to row/col since T is symmetric) to np.array(x,y)
			- xy_to_state_id: map of tuple (x,y) state to state row/col id (equivalent to row/col since T is symmetric). Note: tuple (x,y) is used as key because numpy arrays cannot be hashed for a lookup key.
		
		observation_history: list of past observations (tuples, same as observation)
		
		previous_state_dist: numpy array of shape (num_states,)
			- previous filtered belief π_{t-1}
		
		dummy: use placeholder dummy probabilities, set to False when testing your own implementation

	required outputs:
		numpy array of shape (N*N,), probability distribution over the robot position at time t
	'''
	if dummy: # baseline, do not modify, set dummy to False
		return _get_dummy_probs(observation,  model_info, observation_history)
	
	else:
		num_states = model_info['num_states']
		T = model_info['transition_matrix']
		O = model_info['observation_matrix']

		# Basic shape checks to keep the filter well-defined.
		if previous_state_dist.shape != (num_states,):
			raise ValueError('previous_state_dist must have shape ({},), got {}'.format(num_states, previous_state_dist.shape))
		if T.shape != (num_states, num_states):
			raise ValueError('transition_matrix must have shape ({}, {}), got {}'.format(num_states, num_states, T.shape))
		if O.shape[0] != num_states:
			raise ValueError('observation_matrix must have shape ({}, Z), got {}'.format(num_states, O.shape))

		# Validate previous belief as a probability distribution.
		if np.any(previous_state_dist < 0):
			raise ValueError('previous_state_dist contains negative probabilities')
		if not np.isclose(previous_state_dist.sum(), 1.0, atol=1e-6):
			raise ValueError('previous_state_dist must sum to 1')

		# Normalize and validate the observation as a 4-bit tuple.
		if observation is None or len(observation) != 4:
			raise ValueError('observation must be a length-4 iterable of 0/1 values (N, E, S, W)')
		try:
			obs_tuple = tuple(int(v) for v in observation)
		except (TypeError, ValueError):
			raise ValueError('observation must contain values convertible to int (0 or 1)')
		if any(v not in (0, 1) for v in obs_tuple):
			raise ValueError('observation values must be 0 or 1')

		obs_id_lookup = model_info['obs_id_lookup']
		if obs_tuple not in obs_id_lookup:
			raise KeyError('observation {} not found in obs_id_lookup'.format(obs_tuple))

		z = obs_id_lookup[obs_tuple]
		emission_vec = O[:, z]
		D = np.diag(emission_vec)

		# (I) Prediction: propagate belief through the dynamics.
		state_pred = T.T @ previous_state_dist
		# (II) Update: apply emission likelihood for the current observation.
		unnormalized = D @ state_pred
		normalizer = unnormalized.sum() # compute total likelihood of observation -> same as emission_vec.T @ state_pred
		if normalizer < 1e-12:
			raise ValueError('Zero likelihood for observation under current model; cannot normalize belief.')

		# Normalize to obtain the filtered belief.
		state_update = unnormalized / normalizer
		return state_update

class ViterbiAlgorithm:
	'''
	Online Viterbi decoder (no log-probabilities).
	Stores the forward recursion state and backtracks at the end.
	'''
	def __init__(self, model_info: dict, init_probs: np.ndarray):
		# Cache model components and validate shapes.
		self.T = model_info['transition_matrix']  # (num_states, num_states)
		self.O = model_info['observation_matrix'] # (num_states, num_observations)
		self.obs_id_lookup = model_info['obs_id_lookup']
		self.state_id_to_xy = model_info['state_id_to_xy']
		self.num_states = model_info['num_states']

		# init_probs: (num_states,)
		if init_probs.shape != (self.num_states,):
			raise ValueError('init_probs must have shape ({},), got {}'.format(self.num_states, init_probs.shape))
		if self.T.shape != (self.num_states, self.num_states):
			raise ValueError('transition_matrix must have shape ({}, {}), got {}'.format(self.num_states, self.num_states, self.T.shape))
		if self.O.shape[0] != self.num_states:
			raise ValueError('observation_matrix must have shape ({}, Z), got {}'.format(self.num_states, self.O.shape))
		if np.any(init_probs < 0):
			raise ValueError('init_probs contains negative probabilities')

		self.init_probs = init_probs.astype(float, copy=True)
		self.delta_last = None          # (num_states,)
		self.psi_history = []           # list of arrays, each (num_states,)
		self.T_len = 0                  # number of processed observations

	def _obs_to_id(self, observation: Tuple[int, int, int, int]) -> int:
		# Normalize observation to a 4-bit tuple and map to obs_id.
		if observation is None or len(observation) != 4:
			raise ValueError('observation must be a length-4 iterable of 0/1 values (N, E, S, W)')
		try:
			obs_tuple = tuple(int(v) for v in observation)
		except (TypeError, ValueError):
			raise ValueError('observation must contain values convertible to int (0 or 1)')
		if any(v not in (0, 1) for v in obs_tuple):
			raise ValueError('observation values must be 0 or 1')
		if obs_tuple not in self.obs_id_lookup:
			raise KeyError('observation {} not found in obs_id_lookup'.format(obs_tuple))
		return self.obs_id_lookup[obs_tuple]

	def initialize(self, observation_0) -> None:
		# Initialize forward recursion at t=0 with the first observation.
		obs_id = self._obs_to_id(observation_0)
		emission_vec = self.O[:, obs_id]      # (num_states,)
		delta = self.init_probs * emission_vec # (num_states,)
		# Scale to reduce underflow (does not affect argmax path).
		max_val = delta.max()
		if max_val > 0:
			delta = delta / max_val
		psi_0 = np.zeros(self.num_states, dtype=int) # (num_states,)

		self.delta_last = delta               # (num_states,)
		self.psi_history = [psi_0]            # len = 1, each (num_states,)
		self.T_len = 1

	def step(self, observation_t) -> None:
		# One online Viterbi update for the next observation.
		if self.delta_last is None:
			raise ValueError('ViterbiAlgorithm must be initialized before calling step().')
		obs_id = self._obs_to_id(observation_t)
		emission_vec = self.O[:, obs_id]      # (num_states,)

		# scores[i, j] = delta_t(i) * T[i, j]
		scores = self.delta_last[:, None] * self.T  # (num_states, num_states)
		psi_t = np.argmax(scores, axis=0).astype(int) # (num_states,)
		best_prev = np.max(scores, axis=0)            # (num_states,)
		delta_new = emission_vec * best_prev          # (num_states,)

		# Scale to reduce underflow (does not affect argmax path).
		max_val = delta_new.max()
		if max_val > 0:
			delta_new = delta_new / max_val

		self.delta_last = delta_new           # (num_states,)
		self.psi_history.append(psi_t)        # append (num_states,)
		self.T_len += 1

	def backtrack(self) -> List[int]:
		# Backtrack once all observations have been processed.
		if self.delta_last is None or self.T_len == 0:
			raise ValueError('ViterbiAlgorithm has no history to backtrack.')
		T_len = self.T_len
		path = [0] * T_len                    # length T_len
		path[T_len - 1] = int(np.argmax(self.delta_last))
		for t in range(T_len - 2, -1, -1):
			path[t] = int(self.psi_history[t + 1][path[t + 1]])
		return path

def map_state_sequence(probs_history: List[np.ndarray], model_info: dict) -> List[int]:
	'''
	Greedy MAP-from-marginals path with adjacency constraint (not Viterbi).
	- Choose x_T = argmax(probs_history[T-1])
	- For t = T-1 down to 1: choose x_{t-1} from valid predecessors of x_t
	'''
	if probs_history is None or len(probs_history) == 0:
		raise ValueError('probs_history must be a non-empty list of numpy arrays.')

	num_states = model_info['num_states']
	Tmat = model_info['transition_matrix'] # (num_states, num_states)

	# Validate transition matrix shape.
	if Tmat.shape != (num_states, num_states):
		raise ValueError('transition_matrix must have shape ({}, {}), got {}'.format(num_states, num_states, Tmat.shape))

	# Validate probabilities history.
	for t, probs_t in enumerate(probs_history):
		if not isinstance(probs_t, np.ndarray):
			raise TypeError('probs_history[{}] must be a numpy array. Received: {}'.format(t, type(probs_t)))
		if probs_t.shape != (num_states,):
			raise ValueError('probs_history[{}] must have shape ({},), got {}'.format(t, num_states, probs_t.shape))
		if np.any(probs_t < 0):
			raise ValueError('probs_history[{}] contains negative probabilities'.format(t))
		if not np.isfinite(probs_t).all():
			raise ValueError('probs_history[{}] contains non-finite values'.format(t))
		if not np.isclose(probs_t.sum(), 1.0, atol=1e-5):
			raise ValueError('probs_history[{}] must sum to 1 within tolerance'.format(t))

	T_len = len(probs_history)
	path = [0] * T_len # length T

	# last_belief: (num_states,)
	last_belief = probs_history[-1]
	path[T_len - 1] = int(np.argmax(last_belief))

	# Backtrack with adjacency constraints using incoming neighbors.
	for t in range(T_len - 2, -1, -1):
		curr = path[t + 1]
		# predecessors: (K,)
		predecessors = np.where(Tmat[:, curr] > 0)[0]
		prev_belief = probs_history[t] # (num_states,)
		if predecessors.size == 0:
			# Fallback: no valid predecessor, use marginal argmax.
			path[t] = int(np.argmax(prev_belief))
			continue
		# masked: (K,)
		masked = prev_belief[predecessors]
		best_k = int(np.argmax(masked))
		path[t] = int(predecessors[best_k])

	return path

## -------------------------------------------------

def  _get_dummy_probs(priviledged_info, model_info, observation_history, 
					high_uncertainty_duration=0.5, sigma_range=(50.0,2.0), max_steps=100):
	'''
	simulate a dummy scenario with an arbitrary gaussian probability for converging to the true localization.
	dummy since it uses the "ground-truth" position. 

	dummy parameters: 
	- priviledged_info: get grount-truth agent position
	- model_info: contains information about environmental models
	- observation_history: used to compute current timestep to simulate ucnertainty decrease


	Internal parameters:
	- high_uncertainty_duration: how long to maintain a high standard deviation (std)
	- sigma_range: std bounds for scenario
	- max_steps: max number of steps in episode, used to decide how long to maintain high/low uncertainty

	'''    
	t = len(observation_history) # get a proxy for the current timestep from history length

	max_sigma, min_sigma = sigma_range
	# generate a sequence of decreasing sigmas
	sigmas = np.linspace(max_sigma, min_sigma, num=int(max_steps*high_uncertainty_duration))
	sigmas = np.concatenate([sigmas, min_sigma*np.ones(shape=(int(max_steps*(1-high_uncertainty_duration)),))])
	
	# compute new probabilities
	probs_t = _gaussian_2d_array(model_info['grid_shape'], sigmas[t], center=priviledged_info['agent_position']).flatten()

	return probs_t

def _gaussian_2d_array(shape, sigma=1.0, center=(0,0)):
	# create a meshgrid of x and y coordinates
	x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
	# calculate the distance a center provided
	dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
	# generate the Gaussian values
	gaussian = np.exp(-(dist**2) / (2 * sigma**2))
	# normalize the values to sum to 1
	gaussian /= np.sum(gaussian)
	return gaussian
