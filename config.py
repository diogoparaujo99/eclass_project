from dataclasses import dataclass, fields, field
from typing import Set, Literal, Optional, Type

from policy import Agent, Random

'''
config for running simulation
'''
@dataclass
class RunConfig:
	gridworld_dimensions: int = 20
	obstacle_ratio: float = 0.
	max_steps: int = 100
	robot_policy: Agent = Random()
	dummy: bool = True # set False when using your method
	init_prob_mode: Literal['uniform', 'your-mode'] = 'uniform'
	pe: float = 0.05 # sensor error probability, must be one of {0.0, 0.05, 0.4}

'''
config Spec for running simulation
'''
# config specification used for validation
@dataclass
class RunConfigSpec:
	# numeric ranges and type
	gridworld_dimensions: tuple[int, int, Type] = (5, 100, int)
	obstacle_ratio: tuple[float, float, Type] = (0.0, 1.0, float)
	max_steps: tuple[int, int, Type] = (1, 100, int)
	dummy: Type = bool
	robot_policy: Type = Agent
	init_prob_mode: Set[str] = field(default_factory=set)
	# init_prob_mode: Set[str] = {'uniform'}
	pe: Set[float] = field(default_factory=lambda: {0.0, 0.05, 0.4})

'''
config validation
'''
def validate_config(config: RunConfig, spec: RunConfigSpec):
	''' config spec validation '''
	# Allow passing the spec class; instantiate to resolve default_factory fields.
	if isinstance(spec, type):
		spec = spec()
	params_to_check_group_0 = ['gridworld_dimensions', 'obstacle_ratio', 'max_steps']
	for p in params_to_check_group_0:
		_type_validator(config, spec, p)
		_range_validator(config, spec, p)
	
	params_to_check_group_1 = ['dummy', 'robot_policy', 'pe']
	for p in params_to_check_group_1:
		_type_validator(config, spec, p)
		if p == 'pe':
			_allowed_set_validator(config, spec, p)

	return True

def _range_validator(config: RunConfig, spec: RunConfigSpec, param: str):
	''' validate input ranges '''
	spec_tuple = getattr(spec, param)
	config_value = getattr(config, param)
	
	min_val, max_val = spec_tuple[0], spec_tuple[1]
	
	# Range validation logic
	if not min_val <= config_value <= max_val:
		raise ValueError(
			'Config Error: {} value of {} is out of range. Must be between {} and {}.'.format(param, config_value, min_val, max_val)
		)

def _type_validator(config: RunConfig, spec: RunConfigSpec, param: str):
	''' validate the datatype '''
	config_value = getattr(config, param)
	expected_type = getattr(spec, param)
	
	if isinstance(expected_type, tuple) and len(expected_type) == 3:
		expected_type = expected_type[2]
	elif isinstance(expected_type, set):
		# infer element type for allowed-set specs
		if expected_type:
			expected_type = type(next(iter(expected_type)))
		else:
			return True

	if not isinstance(config_value, expected_type):
		raise TypeError(
			'Config Error: {} value must be of type {}, but got {}.'.format(param, expected_type.__name__, type(config_value).__name__ ) 
			)

def _allowed_set_validator(config: RunConfig, spec: RunConfigSpec, param: str):
	''' validate value is in allowed set '''
	spec_set = getattr(spec, param)
	config_value = getattr(config, param)
	if not isinstance(spec_set, set):
		raise TypeError('Config Error: {} spec must be a set of allowed values.'.format(param))
	if spec_set and config_value not in spec_set:
		raise ValueError(
			'Config Error: {} value of {} is not in allowed set {}.'.format(param, config_value, spec_set)
		)

def make_config_markdown(config: RunConfig, spec: RunConfigSpec) -> str:
	''' automatically generate a markdown table for documentation '''
	headers = ['variable name', 'data-type/format', 'input value/range', 'default value']
	lines = [
			'| ' + ' | '.join(headers) + ' |',
			'| ' + ' | '.join(['---'] * len(headers)) + ' |',
			]
	
	for f in fields(RunConfig):
		name = f.name
		value = getattr(config, name)
		spec_val = getattr(spec, name, None)

		# when spec is a (min, max, type) tuple
		if isinstance(spec_val, tuple) and len(spec_val) == 3:
			min_val, max_val, t = spec_val
			dtype = t.__name__
			value_range = '{}-{}'.format(min_val, max_val)

		# when spec is a set of allowed values
		elif isinstance(spec_val, set):
			if spec_val and all(isinstance(v, float) for v in spec_val):
				dtype = 'float'
			elif spec_val and all(isinstance(v, int) for v in spec_val):
				dtype = 'int'
			else:
				dtype = 'str'
			if spec_val:
				# show allowed options, e.g. {'uniform'}
				value_range = '{{{}}}'.format(', '.join(sorted(repr(v) for v in spec_val)))
			else:
				value_range = '{}' # empty set when no fixed list / extensible
				
		# when spec is a generic type
		elif isinstance(spec_val, type):
			dtype = spec_val.__name__
			if spec_val is bool:
				value_range = '[True, False]'
			else:
				value_range = '{}'.format(dtype) # generic description

		else:
			# fallback, use current value
			dtype = type(value).__name__
			value_range = repr(spec_val)

		default_str = repr(value)

		line = '| `{}` | `{}` | {} | {} |'.format(name, dtype, value_range, default_str)
		lines.append(line)
	
	return '\n'.join(lines)

markdown_config_table = make_config_markdown(RunConfig(), RunConfigSpec())

## --------------------------------
## An LLM helped with the template below :) 

HTML_TEMPLATE = """<!DOCTYPE html>
<html> 
<head>
	<title>Image Sequence</title>
	<style>
		#container {                 /* container for dropdown and controls */
			display: flex;           /* use flexbox for horizontal layout */
			align-items: center;     /* vertically align items */
			justify-content: left;   /* centers the content horizontally*/
			margin-bottom: 10px;     /* add some space below */
		}
		#sequence {
			max-width: 100%;
			display: block;
			margin: 0 auto;
		}
		.controls {
			margin-left: 50px; 
			display: flex;  /* use flexbox for horizontal layout of buttons */
			gap: 5px;       /* add some space between buttons */
		}
		.reset {
			margin-left: 50px; 
			display: flex; 
		}
	</style>
</head>
<body>
	<div id="container">
		<select id="directorySelect" onchange="changeDirectory()">
__OPTIONS__
		</select>
		<div class="controls">
			<button onclick="prevFrame()">Previous</button>
			<button onclick="togglePlay()">Play/Pause</button>
			<button onclick="nextFrame()">Next</button>
		</div>
		<div class="reset">
			<button onclick="toggleReset()">Reset</button>
		</div>
	</div>
	<img id="sequence" src="results/__DEFAULT_DIR__/frame-0.png">

	<script>
		let currentDirectory = "__DEFAULT_DIR__"; // initial directory
		let numFrames = 0;
		let currentFrame = 0;
		let isPlaying = false;
		let intervalId;
		const imageElement = document.getElementById('sequence');

		function updateFrame() {
			imageElement.src = `results/${currentDirectory}/frame-${currentFrame}.png`;
		}

		function nextFrame() {
			currentFrame = (currentFrame + 1) % numFrames;
			updateFrame();
		}

		function prevFrame() {
			currentFrame = (currentFrame - 1 + numFrames) % numFrames;
			updateFrame();
		}

		function play() {
			intervalId = setInterval(nextFrame, 250);
		}

		function pause() {
			clearInterval(intervalId);
		}

		function reset() {
			currentFrame = 0;
			updateFrame();
		}

		function togglePlay() {
			isPlaying = !isPlaying;
			if (isPlaying) {
				play();
			} else {
				pause();
			}
		}

		function toggleReset() {
			reset();
		}

		function changeDirectory() {
			currentDirectory = document.getElementById("directorySelect").value;
			// extract numFrames from directory name
			const parts = currentDirectory.split("_");
			if (parts.length === 2) {
				numFrames = parseInt(parts[1], 10);
			} else {
				console.error("Invalid directory name format:", currentDirectory);
				numFrames = 100;
			}
			currentFrame = 0; // reset frame
			updateFrame(); // update the displayed image
		}

		// initial setup (extract numFrames from initial directory)
		const initialParts = currentDirectory.split("_");
		if (initialParts.length === 3) {
			numFrames = parseInt(initialParts[2].split("-")[1], 10);
		} else {
			console.error("Invalid initial directory name format:", currentDirectory);
			numFrames = 100; // or some default value
		}
		updateFrame();// initial display
	</script>
</body> 
</html>"""
