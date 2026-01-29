Student README - HMM Filtering Implementation

Overview
- Implemented a Hidden Markov Model (HMM) forward filter for GridWorld self-localization.
- The filter estimates the belief over states at time t: pi_t[i] = P(X_t = i | o_1:t, theta).
- The dummy baseline path remains unchanged.

Algorithm (in localization.py)
1) Prediction
   state_pred = T.T @ previous_state_dist
   where T[i,j] = P(x_t=j | x_{t-1}=i).

2) Update
   - Convert the raw observation (N,E,S,W) to a 4-tuple of ints.
   - Map the tuple to a column index z using obs_id_lookup.
   - Extract emission likelihoods: emission_vec = O[:, z].
   - Apply the observation model: unnormalized = diag(emission_vec) @ state_pred.

3) Normalize
   state_update = unnormalized / sum(unnormalized).

Validation and edge cases
- Validates shapes for previous_state_dist, transition_matrix, and observation_matrix.
- Ensures previous_state_dist is a proper distribution (non-negative, sums to 1).
- Ensures observation is length 4, values are 0/1, and exists in obs_id_lookup.
- If the observation likelihood is (near) zero, raises a clear error instead of returning NaNs.

Integration changes (utils.py)
- run_sample now passes the previous belief into get_state_probabilities.
- For t == 1, it uses init_probs as the prior; for later steps it uses the last belief in history.

Viterbi decoding (localization.py)
- Implemented ViterbiAlgorithm with online forward recursion and end-of-episode backtracking.
- Initialization (t=0):
  - obs_id = obs_id_lookup[obs_tuple]
  - emission_vec = O[:, obs_id]                       # shape (N,)
  - delta_0 = init_probs * emission_vec               # shape (N,)
  - psi_0 = zeros(N)                                  # shape (N,)
  - Optional scaling: delta_0 /= max(delta_0) if max > 0
- Forward step (t >= 1):
  - scores = delta_last[:, None] * T                  # shape (N, N)
  - psi_t = argmax(scores, axis=0)                    # shape (N,)
  - best_prev = max(scores, axis=0)                   # shape (N,)
  - delta_new = emission_vec * best_prev              # shape (N,)
  - Scaling: delta_new /= max(delta_new) if max > 0
- Backtracking:
  - x_hat[T-1] = argmax(delta_last)
  - x_hat[t] = psi_history[t+1][x_hat[t+1]]
- Scaling does not change the argmax path because it multiplies all delta values by the same positive constant.

Viterbi integration (utils.py)
- run_sample now creates ViterbiAlgorithm after reset when dummy=False.
- initialize() is called once with the first observation; step() is called every timestep with raw 4-bit observations.
- backtrack() runs once at the end to get viterbi_state_ids (length T).
- A Viterbi-only map is saved to results/<run_name>/viterbi_path.png.
- A second GIF with Viterbi overlay is saved to figures/<run_name>_viterbi.gif.

Rendering changes (environment.py)
- render_gridworld now accepts viterbi_state_ids (length T) and viterbi_prefix_len to draw a red Viterbi path.
- draw_viterbi_labels controls drawing dark-red "S"/"F" on top of the grid (after robot/grid lines).
- The "S"/"F" glyphs are vertically flipped before blitting to counter the final image flip.
- Fixed Viterbi GIF robot pose: the wrapper env kept the base env at the final pose during re-rendering, so the blue robot did not move.
- GridWorldEnv now stores agent_pos_history (List[(2,)] length T) on reset/step, and the Viterbi GIF renderer sets base_env.agent_pos = agent_pos_history[k].copy() per frame (using env.unwrapped).

Files touched
- localization.py: forward filter logic and comments.
- utils.py: Viterbi online integration, viterbi_path.png, and Viterbi overlay GIF.
- environment.py: Viterbi path rendering with labels and prefix-length support.
- student.md: updated documentation.

How to run
- Use the existing demo or run a short sample via utils.run_sample.
- The output beliefs are checked for shape, non-negativity, and normalization in run_sample.
- The Viterbi visualization is saved under results/<run_name>/viterbi_path.png.
