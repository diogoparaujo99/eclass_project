import random 
import numpy as np
from abc import ABC, abstractmethod
from environment import Actions, Observations

class Agent(ABC):
	''' Abstract class used to implement agents '''
	def __init__(self, 
			  	 name,
				 valid_actions, *kwargs):
		self.name = name
		self.valid_actions = valid_actions

	@abstractmethod
	def reset(self, observations=None):
		''' reset any internal state/memory '''
		return 
	
	@abstractmethod
	def act(self, observations=None):
		''' implement policy '''
		return 
	def __repr__(self):
		return 'policy.Agent()'

class Random(Agent):
	def __init__(self, sticky_actions=False):
		''' 
		pseudo-random action policy from a set of fixed actions. 
		only valid actions (free-cells) given the current observation are considered
		(optional) enable sticky actions, change `num_repeat` to configure stickyness
		'''

		self.valid_actions = {action.name for action in Actions}
		self.sticky_actions = sticky_actions # repeat actions
		self.prev_action = random.choice(list(self.valid_actions))
		self.num_repeat = 5
		self.repeat_count = 0
		self.name = 'random' # metadata

		super().__init__(self.name, self.valid_actions)

	def reset(self):
		self.repeat_count = 0
		self.prev_action = random.choice(list(self.valid_actions))
	
	def act(self, observation):
		
		# get the set of actions unavailable given the observation
		actions_not_available = {
			Actions(Observations.get_direction_by_index(obs_idx)).name 
			for obs_idx, obs in enumerate(observation)
			if obs==1}

		# get the names of the actions available by doing set subtraction
		actions_available = self.valid_actions - actions_not_available

		# only keep the actions available in Action(Enum) type
		actions_available = [Actions[name] for name in actions_available]
		if (self.sticky_actions) and (self.repeat_count % self.num_repeat != 0) and (self.prev_action is not None):
			action =  self.prev_action
		else: 
			action = random.choice(actions_available).name
		
		self.prev_action = action
		self.repeat_count += 1
		return  action
	
	def __repr__(self):
		cls = self.__class__
		return '{}.{}'.format(cls.__module__, cls.__name__)