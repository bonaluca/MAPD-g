import numpy as np
import logging
import re
import time
from os import path
from itertools import product
from scipy import sparse
from Simulation.exceptions import NotFittedError

class Cache:
    """Cache for the powers of the transition matrix."""

    def __init__(self, size):

        self.size_bytes = size if type(size) == int else 0
        if type(size) == str:
            self.size_bytes = Cache.human2bytes(size)
        self.free_space = self.size_bytes
        self._items = []

    def store(self, matrix):
        matrix_size = matrix.nnz * 8 # 8 bytes float
        if matrix_size <= self.free_space:
            self._items.append(matrix)
            self.free_space -= matrix_size
            logging.debug("Markov Chain: cached transition matrix power %d" \
                % (len(self._items) - 1))
        else:
            self.free_space = 0

    def retrieve(self, t):
        return self._items[t]

    def last_t_stored(self):
        return len(self._items) - 1

    def is_full(self):
        return self.free_space <= 0

    def flush(self):
        self.free_space = self.size_bytes
        self._items = []

    def set_size(self, size):
        new_size_bytes = size if type(size) == int else 0
        if type(size) == str:
            new_size_bytes = Cache.human2bytes(size)

        if new_size_bytes >= self.size_bytes:
            self.free_space += new_size_bytes - self.size_bytes
            self.size_bytes = new_size_bytes
            return
        # If it shrinks we flush
        self.size_bytes = new_size_bytes
        self.flush()

    @staticmethod
    def human2bytes(h_bytestring):
        """Convert human readable size string to the integer number of bytes."""

        SYMBOLS = ['B', 'K', 'M', 'G', 'T']
        regex = re.compile(r'^(\d+\.?\d*) ?([bmkgt])', re.IGNORECASE)
        match = regex.match(h_bytestring)
        if not match:
            raise ValueError(f'Bytestring {h_bytestring} in the wrong format')

        num = float(match[1])
        symbol = match[2].upper()
        return round(num * (1 << 10 * SYMBOLS.index(symbol)))

class MarkovChain(object):

    def __init__(self, width, height, obstacles=None, order=1, prune_thres=None, cache_size=None, init_policy='random'):
        self.width = width
        self.height = height
        self.n_states = self.width * self.height
        self.order = order
        self.obstacles = obstacles or []
        self.prune_thres = prune_thres
        self._not_fitted = True
        self._backoff_in_use = False
        self.init_policy = init_policy

        location_valid = lambda l: l not in self.obstacles \
            and (0 <= l[0] < self.width) \
            and (0 <= l[1] < self.height)

        self.neighbors = lambda l: list(filter(location_valid,
            [(l[0]-1, l[1]), (l[0], l[1]-1), (l[0], l[1]), (l[0], l[1]+1), (l[0]+1, l[1])]
        ))

        self.valid_states = list(filter(lambda l: location_valid(l[0]),
            product(product(range(self.width), range(self.height)), repeat=1))
        )
        for k in range(self.order - 1):
            states = self.valid_states.copy()
            self.valid_states = []
            for state in states:
                    last_location = state[-1]
                    for next_location in self.neighbors(last_location):
                        self.valid_states.append(tuple((*state, next_location)))

        self.state_idx = {
            state: idx
            for idx, state in enumerate(self.valid_states)
        }

        self.valid_transitions = []
        for state in self.valid_states:
            last_location = state[-1]
            for next_location in self.neighbors(last_location):
                self.valid_transitions.append(tuple(
                    (state, (*state[1:], next_location))
                ))

        self.n_valid_states = sum(1 for state in self.valid_states)

        # Allocating transition matrix
        self._transition_matrix = self._make_stochastic(
            sparse.csr_matrix(2 * (self.n_valid_states, ), dtype=np.float64)
        )

        self.obs_counts = np.zeros((self.n_valid_states, ), dtype=int)

        # Lazy caching of the powers of the transition matrix
        self.cache = Cache(cache_size)
        self._last_computation = None

    @property
    def transition_matrix(self):
        if self._backoff_in_use:
            return self._backoff_matrix
        return self._transition_matrix

    def is_valid_transition(self, state, next_state):
        if state[1:] != next_state[:-1]:
            return False
        return next_state[-1] in self.neighbors(state[-1])

    def fit(self, sequence):
        """Fit the transition matrix on a squence of observations."""

        self.cache.flush()

        # Cumulating occurrences
        occurrences = sparse.dok_matrix(self._transition_matrix.shape, dtype=int)
        state_sequence = zip(*[sequence[i:] for i in range(self.order)])
        for state, next_loc in zip(state_sequence, sequence[self.order:]):
            next_state = tuple((*state[1:], next_loc))
            state_idx = self.state_idx[state]
            next_state_idx = self.state_idx[next_state]
            occurrences[next_state_idx, state_idx] += 1

        # Update rule for the transition matrix:
        # T = T * A + (T * B + O) * C
        #   = T * (A + B * C) + O * C
        # where:
        #    T = transition matrix
        #    O = occurrences matrix
        #    A = diag(1 if no occurrences)
        #    B = diag(obs_counts)
        #    C = diag((obs_counts + sum_occurrences)^(-1))

        sum_occurrences = np.asarray(occurrences.sum(axis=0)).ravel()

        # A + B * C matrix
        count_frac = [
            past_occ / (past_occ + new_occ) if new_occ > 0 else 1
            for past_occ, new_occ in zip(self.obs_counts, sum_occurrences)
        ]
        diag_count_frac = sparse.diags(count_frac, dtype=np.float64)

        # C matrix
        inv_counts = [
            1 / (past_occ + new_occ) if new_occ > 0 else 0
            for past_occ, new_occ in zip(self.obs_counts, sum_occurrences)
        ]
        diag_inv_counts = sparse.diags(inv_counts, dtype=np.float64)

        # Update rule
        self._transition_matrix = self._transition_matrix.dot(diag_count_frac) + \
            occurrences.dot(diag_inv_counts)

        # Cumulating observation counts
        self.obs_counts += sum_occurrences

        self._not_fitted = False

    def predict(self, initial_state, time):
        """Return the probability of occupancy at a given time."""

        t = time
        k_step_belief = initial_state
        while t > 0:
            k, k_step_matrix = self.compute_t_step_matrix(t)
            k_step_belief = k_step_matrix.dot(k_step_belief)
            k_step_belief = self._prune_vector(k_step_belief)
            t -= k
        return k_step_belief

    def time_forward(self, initial_state, free_locations=[], busy_location=None):
        """Update the belief state of the markov chain."""

        # Agent was seen at busy_location
        if busy_location is not None:
            # Building fictitious transition matrix that moves to busy_location
            state_with_busy = list(filter(
                lambda state: state[-1] == busy_location,
                self.valid_states
            ))
            # Deterministic transition to busy_location if valid
            valid_transitions_to_busy = [
                (state, tuple((*state[1:], busy_location)))
                for state in self.valid_states
                if self.is_valid_transition(state, tuple((*state[1:], busy_location)))
            ]
            col_ind = list(map(
                lambda t: self.state_idx[t[0]],
                valid_transitions_to_busy
            ))
            row_ind = list(map(
                lambda t: self.state_idx[t[1]],
                valid_transitions_to_busy
            ))
            data = sum(1 for t in valid_transitions_to_busy) * [1,]
            # Uniform over state_with_busy if transition to busy_location is invalid
            state_with_invalid = list(set(range(self.n_valid_states)) - set(col_ind))
            row_ind += len(state_with_invalid) * list(map(
                lambda state: self.state_idx[state],
                state_with_busy
            ))
            col_ind += [
                i for i in state_with_invalid
                for _ in range(len(state_with_busy))
            ]
            data += len(state_with_invalid) * len(state_with_busy) * [1. / len(state_with_busy),]
            # Fictitious transition matrix
            fictitious_t_matrix = sparse.csr_matrix(
                (data, (row_ind, col_ind)),
                shape=self.transition_matrix.shape,
                dtype=np.float64
            )
            belief = fictitious_t_matrix.dot(initial_state)
            return belief

        # Agent was not seen at free_locations
        free_loc_indices = list(
            map(lambda state: self.state_idx[state],
                filter(lambda state: state[-1] in free_locations,
                self.valid_states))
        )

        # Update belief
        belief = self.transition_matrix.dot(initial_state)

        # Nullify rows corresponding to free locactions
        one_minus_diag = sparse.csr_matrix(
            (len(free_loc_indices) * [1,], (free_loc_indices, len(free_loc_indices) * [0,])),
            shape=(self.n_valid_states, 1)
        )
        diag = np.asarray(
            np.ones((self.n_valid_states, 1)) - one_minus_diag
        ).ravel()

        row_selector = sparse.diags(diag, dtype=np.float64, format="csr")
        belief = row_selector.dot(belief)

        # Normalization requires some care: may end up in a zero belief vector
        if belief.sum() == 0:
            logging.warning('Markov chain belief vanished!')
            # Recover by assuming agent is wherever but in free_locations
            belief = row_selector.dot(self.obs_counts)
        if belief.sum() == 0:
            # Recover by assuming uniform belief
            belief = np.ones((self.n_valid_states, )) / self.n_valid_states

        belief /= belief.sum() # Normalize
        return belief

    def compute_t_step_matrix(self, t, only_cached=True):
        """Compute and cache the t-step transition matrix.

        Return a tuple (t, A) where A is the t-th power of the transition
        matrix.
        For values of t that exceed cache capacity, it returns the t-step
        transition matrix with the largest t that fits in cache.
        """

        last_t_cached = self.cache.last_t_stored()

        # Populating cache
        while last_t_cached < t and not self.cache.is_full():
            if last_t_cached == -1:
                t_step_matrix = sparse.eye(self.n_valid_states,
                    dtype=np.float64, format="csr")
            else:
                step_t_minus_one = self.cache.retrieve(last_t_cached)
                t_step_matrix = self.transition_matrix.dot(step_t_minus_one)
            t_step_matrix = self._prune_matrix(t_step_matrix)
            self.cache.store(t_step_matrix)
            last_t_cached = self.cache.last_t_stored()
        # L1 cache hit
        if self._last_computation and self._last_computation['time'] == t:
            return t, self._last_computation['matrix']
        # L2 cache hit
        if t <= last_t_cached:
            t_step_matrix = self.cache.retrieve(t)
            return t, t_step_matrix
        # Not using L2 cache
        if last_t_cached < 1:
            if only_cached:
                t = min(t, 1)
            t_step_matrix = self.transition_matrix ** t
            t_step_matrix = self._prune_matrix(t_step_matrix)
            return t, t_step_matrix
        # Cache capacity is exceeded
        if not only_cached:
            last_cached_matrix = self.cache.retrieve(last_t_cached)
            if self._last_computation and t > self._last_computation['time'] > last_t_cached:
                last_t_cached = self._last_computation['time']
                last_cached_matrix = self._last_computation['matrix']
            t_step_matrix = self.transition_matrix ** (t - last_t_cached)
            t_step_matrix = t_step_matrix.dot(last_cached_matrix)
            t_step_matrix = self._prune_matrix(t_step_matrix)
            # Updating last computation
            self._last_computation = {
                'time': t,
                'matrix': t_step_matrix
            }
            return t, t_step_matrix
        return last_t_cached, self.cache.retrieve(last_t_cached)

    def state_to_belief(self, state):
        """Turn state into a belief over the set of possible locations.

        For markov chains of order > 1 the state is a probability distribution
        over all possible sequences of locations.
        This method accumulates the probability for sequences that end up in
        the same location.
        """

        data = self.n_valid_states * [1,]
        col_index = list(self.state_idx.values())
        # Row i contains 1 in position j if state j ends up in location i
        row_index = list(map(
            lambda state: state[-1][0] * self.height + state[-1][1],
            self.state_idx.keys()
        ))

        sum_matrix = sparse.csr_matrix((data, (row_index, col_index)),
            shape=(self.width * self.height, self.n_valid_states))
        state = sum_matrix.dot(state)
        return state.reshape((self.width, self.height))

    def _make_stochastic(self, matrix, policy=None, eps=0.7):
        """Return a stochastic matrix.

        Parameters
        ----------
        policy : str
            - Specifying 'random' in case a column is full of zeros, the
            probability is uniformly distributed on all possible moves,
            excluding staying still.
            - Specifying 'selfloop' in case a column is full of zeros, all
            probability is given to the self-loop.
            - Speciying 'heading' in case a column is full of zeros, 1 - eps
            probability is given to the move that maintains the direction and
            eps probability is uniformly distributed on the rest of the moves,
            excluding staying still (equivalent to 'random' for 1st order).
        """

        if policy is None:
            policy = self.init_policy

        # Make sure all columns are non-null
        sum_cols = np.ravel(matrix.sum(axis=0))
        null_cols = np.where(sum_cols == 0)[0].tolist()
        if policy == 'selfloop':
            # Give 100% probability to the self-loop
            col_ind = null_cols
            row_ind = [self.state_idx[tuple((*s[1:], s[-1]))]
                    for s in map(lambda j: self.valid_states[j], col_ind)]
            data = len(col_ind) * [1,]
        elif policy == 'random':
            col_ind = []
            row_ind = []
            data = []
            # Spread probability uniformly over valid (non-stopping) next states
            for col_idx in null_cols:
                start_state = self.valid_states[col_idx]
                valid_next_states = list(map(lambda t: t[1],
                    filter(lambda t: \
                    t[0] == start_state and t[1] != start_state,
                    self.valid_transitions)
                ))
                for next_state in valid_next_states:
                    col_ind.append(col_idx)
                    row_ind.append(self.state_idx[next_state])
                    data.append(1. / len(valid_next_states))
        elif policy == 'heading':
            # TODO @bonaluca
            raise NotImplementedError()
        else:
            raise ValueError(f'Unknown policy {policy}')
        missing_prob = sparse.csr_matrix((data, (row_ind, col_ind)),
                shape=matrix.shape, dtype=np.float64)
        matrix += missing_prob

        # Normalize
        sum_cols = np.ravel(matrix.sum(axis=0))
        inv_sum_diag = sparse.diags(sum_cols ** -1, dtype=np.float64)
        return matrix.dot(inv_sum_diag)

    def _prune_matrix(self, matrix):
        """Nullify entries of a sparse matrix if lower than a threshold."""

        if self.prune_thres is not None:
            matrix.data = np.array(list(map(
                lambda x: 0 if x <= self.prune_thres else x,
                matrix.data
            )))
            sparse.csr_matrix.eliminate_zeros(matrix)
        return self._make_stochastic(matrix)

    def _prune_vector(self, vector):
        """Nullify entries of a numpy array if lower than a threshold."""

        if self.prune_thres is None:
            return vector

        pruned = np.where(vector <= self.prune_thres, 0, vector)
        if pruned.sum() == 0:
            pruned = np.zeros_like(vector)
            pruned[np.argmax(vector)] = 1.

        return pruned / pruned.sum()

    @staticmethod
    def good_turing(freq_of_freqs):
        """Simple Good-Turing smoothing.

        Geoffrey Sampson (2005): http://www.grsampson.net/D_SGT.c
        """

        # Computing frequency N_r of frequencies r
        r = list(freq_of_freqs.keys())
        N_r = list(freq_of_freqs.values())

        if len(r) < 2:
            return {r: r for r in r}

        # Quantity Z_r
        q = [0,] + r[:-1]
        t = r[1:] + [2 * r[-1] - r[-2]]
        Z_r = [2 * Nr / (t - q) for (q, Nr, t) in zip(q, N_r, t)]

        # Least-squares fit in log-log space
        log_r = np.log(r)
        mean_r = np.mean(log_r)
        log_r = log_r - mean_r

        log_z = np.log(Z_r)
        mean_z = np.mean(log_z)
        log_z = log_z - mean_z

        x_squares = np.sum(log_r ** 2)
        xys = np.sum(log_r * log_z)
        slope = xys / x_squares
        intercept = mean_z - slope * mean_r

        # Smoothing: S(N_r) = exp{intercept + slope * log(r)}
        smooth = lambda r: (r ** slope) * np.exp(intercept)
        r_star = {r: (r + 1) * smooth(r + 1) / smooth(r) for r in r}

        return r_star

    def backoff(self):
        """Apply Katz backoff."""

        # Compute ngram counts
        obs_count_diag = sparse.diags(self.obs_counts, dtype=np.float64, format="csr")
        ngram_counts = self._transition_matrix.dot(obs_count_diag)

        # Compute frequency N_r of frequencies r
        nonzero_counts = np.rint(ngram_counts.data)
        #nonzero_counts = np.ravel(ngram_counts[ngram_counts.nonzero()])
        obs_count_freqs = np.bincount(np.array(nonzero_counts, dtype=int))
        freq_of_freqs = {
            r: obs_count_freqs[r]
            for r in np.nonzero(obs_count_freqs)[0]
        }

        # Simple Good-Turing discount
        r_star = MarkovChain.good_turing(freq_of_freqs)

        # Discounted probabilities matrix
        discounted_counts = list(map(lambda r: r_star[r], nonzero_counts))
        discounted_matrix = sparse.csr_matrix((discounted_counts, ngram_counts.nonzero()),
                shape=self._transition_matrix.shape, dtype=np.float64)
        inv_counts = [1/c if c != 0 else 0 for c in self.obs_counts]
        inv_counts_diag = sparse.diags(inv_counts, dtype=np.float64, format="csr")
        discounted_matrix = discounted_matrix.dot(inv_counts_diag)

        # Leftover probability
        leftover = 1 - np.ravel(discounted_matrix.sum(axis=0))

        # k-gram conditional probabilities P(w_n | w_n-1, ...)
        kgram_counts = self.obs_counts
        backoff_prob = np.zeros(self.obs_counts.shape, dtype=np.float64)
        k_states = self.valid_states
        for k in range(1, self.order + 1):
            # Aggregate counts with same history (with repetition)
            history = list(map(lambda state: state[:-1], k_states))
            k_blocks, km1_blocks = [], []
            hist_elem = history[0]
            km1_states = [hist_elem, ]
            count = 1
            for elem in history[1:]:
                if elem != hist_elem:
                    k_blocks.append(np.ones(2 * (count, ), dtype=int))
                    km1_blocks.append(np.ones((1, count), dtype=int))
                    hist_elem = elem
                    km1_states.append(hist_elem)
                    count = 1
                else:
                    count += 1
            k_blocks.append(np.ones(2 * (count, ), dtype=int))
            km1_blocks.append(np.ones((1, count), dtype=int))

            k_block_diag = sparse.block_diag(k_blocks, format="csr")
            if k == 1:
                n_block_diag = k_block_diag
            km1_block_diag = sparse.block_diag(km1_blocks, format="csr")
            sum_counts = k_block_diag.dot(kgram_counts)

            # Compute kgram conditional probabilities
            sum_counts = np.where(sum_counts == 0, 1, sum_counts) # Prevent division by 0
            kgram_prob = kgram_counts / sum_counts

            # Backoff: estimate zero probabilities by conditioning on shorter history
            where_bo_zero = np.where(backoff_prob == 0)[0]
            kgram_idxs = [k_states.index(self.valid_states[i][k-1:]) for i in where_bo_zero]
            backoff_prob[where_bo_zero] = kgram_prob[kgram_idxs]

            # Reshape for following iteration
            k_states = km1_states
            kgram_counts = km1_block_diag.dot(kgram_counts)

        # Strongly discourage giving probability to idle moves
#        non_idle_diag = list(map(
#            lambda state: 1 if len(state) == 1 or state[-1] != state[-2] else 0,
#            self.valid_states
#        ))
#        non_idle = sparse.diags(non_idle_diag, dtype=np.float64)
#        backoff_prob = non_idle.dot(backoff_prob)

        # Normalize backoff probability
        sum_prob = n_block_diag.dot(backoff_prob)
        sum_prob = np.where(sum_prob == 0, 1, sum_prob) # Prevent division by 0
        backoff_prob = backoff_prob / sum_prob

        # Construct backoff matrix

        nonzero_counts_idx = [(i, j) for i, j in zip(*ngram_counts.nonzero())]
        nonzero_counts_idx += [2 * (self.n_valid_states, ), ]
        nonzero_counts_idx.sort(key=lambda t: t[1] * self.n_valid_states + t[0])
        k = 0
        data, row_ind, col_ind = [], [], []
        for j, i in sorted(map(
            lambda t: (self.state_idx[t[0]], self.state_idx[t[1]]),
            self.valid_transitions),
            key=lambda x: x[0] * self.n_valid_states + x[1]):
            nzi, nzj = nonzero_counts_idx[k]
            if j < nzj or (j == nzj and i < nzi):
                row_ind += [i,]
                col_ind += [j,]
                data += [backoff_prob[i],]
            else:
                k += 1
        backoff_matrix = sparse.csr_matrix((data, (row_ind, col_ind)),
            shape=self._transition_matrix.shape, dtype=np.float64)

        # Normalize to leftover probability
        sum_prob = np.ravel(backoff_matrix.sum(axis=0))
        sum_prob = np.where(sum_prob == 0, 1, sum_prob) # Prevent division by 0
        leftover_diag = sparse.diags(leftover / sum_prob, dtype=np.float64)
        backoff_matrix = backoff_matrix.dot(leftover_diag)

        # Update transition matrix
        backoff_matrix += discounted_matrix
        self._backoff_matrix = self._make_stochastic(backoff_matrix)
        self._backoff_in_use = True

    def get_initial_belief(self, sequence=None):
        """Return a vector representing the initial belief of the markov chain."""

        if sequence is None:
            # Uniform belief over valid states
            return np.ones((self.n_valid_states, ), dtype=np.float64) \
                * (1 / self.n_valid_states)

        n_fill = self.order - len(sequence)
        if n_fill > 0:
            sequence = n_fill * [sequence[0]] + sequence

        initial_state_idx = self.state_idx[tuple(sequence[-self.order:])]
        belief = np.zeros((self.n_valid_states, ), dtype=np.float64)
        belief[initial_state_idx] = 1
        return belief

    def load(self, filename):
        """Load model parameters from file."""

        count_matrix = sparse.load_npz(filename)

        self.obs_counts = np.ravel(count_matrix.sum(axis=0))
        self._transition_matrix = self._make_stochastic(count_matrix)
        self._not_fitted = False
        self.cache.flush()

    def save(self, filename, compressed=True):
        """Save model parameter to file."""

        # Turn transition matrix into integer counts
        obs_count_diag = sparse.diags(self.obs_counts, dtype=np.int32, format="csr")
        count_matrix = self._transition_matrix.dot(obs_count_diag)
        count_matrix = sparse.csr_matrix(np.rint(count_matrix), dtype=np.int32)

        arrays_dict = {}
        arrays_dict.update(
            indices=count_matrix.indices,
            indptr=count_matrix.indptr,
            format=count_matrix.format.encode('ascii'),
            shape=count_matrix.shape,
            data=count_matrix.data,
            valid_states=self.valid_states
        )
        if compressed:
            np.savez_compressed(filename, **arrays_dict)
        else:
            np.savez(filename, **arrays_dict)

    def get_obs_counts(self):
        return int(sum(self.obs_counts))

class OccupancyModel(object):
    """Simple occupancy model."""

    def __init__(self, width, height, agents):
        self.width = width
        self.height = height
        self.agent_names = list(map(lambda a: a['name'], agents))
        self.n_agents = len(self.agent_names)

        self.prob_occ = np.zeros((self.width, self.height))
        self.obs_counts = 0
        self.fit_time = 0
        self.inference_time = 0
        self.inference_time_for_replan = 0

    def fit(self, agent_paths, **kwargs):
        """Estimate the probability of occupancy of the grid cells."""

        # Tick
        tock = self.tick()

        # Filter out empty agent paths
        agent_paths = dict(filter(
            lambda item: item[1], agent_paths.items()
        ))
        if not agent_paths:
            return

        # Compute observation time
        max_t = max([max(path, key=lambda step: step['t'])['t'] \
            for path in agent_paths.values()])
        min_t = min([min(path, key=lambda step: step['t'])['t'] \
            for path in agent_paths.values()])
        observation_time = max_t - min_t + 1

        # Update probability of occupancy
        self.prob_occ *= self.obs_counts
        self.obs_counts += observation_time

        for agent_path in agent_paths.values():
            for step in agent_path:
                self.prob_occ[step['x']][step['y']] += 1
        self.prob_occ /= self.obs_counts

        # Tock
        self.fit_time += self.tick() - tock

    def predict(self, location, time, is_replanning=False):
        """Return the probability of occupancy of a location at time t."""

        # Tick
        tock = self.tick()

        prob = self.prob_occ[location[0]][location[1]]

        # Tock
        if not is_replanning:
            self.inference_time += self.tick() - tock
        else:
            self.inference_time_for_replan += self.tick() - tock

        return prob

    def time_forward(self, free_locations=[], seen_agents={}):
        """Collect observations coming from guests."""

        pass

    def load(self, file):
        """Load model parameters from file."""

        file, extension = path.splitext(file)
        if extension == '':
            extension = '.npz'
        file = file + extension

        data = np.load(file)
        self.prob_occ = data['occupancy_matrix']
        self.obs_counts = int(data['obs_counts'])

    def save(self, file):
        """Save model parameters to file."""

        np.savez_compressed(file, occupancy_matrix=self.prob_occ, obs_counts=self.obs_counts)

    def get_obs_counts(self):
        return self.obs_counts

    def get_probability_matrix(self):
        return self.prob_occ.tolist()

    def get_prob_matrix_90deg(self):
        """Return the estimated probability of occupancy matrix.

        This method exists just for backward compatibility
        """
        return np.rot90(self.prob_occ, k=1, axes=(0, 1)).tolist()

    def get_fit_time(self):
        return self.fit_time

    def get_inference_time(self):
        return self.inference_time

    def get_inference_time_for_replan(self):
        return self.inference_time_for_replan

    def tick(self):
        """Return the current time"""
        return time.time()

class OracleModel(OccupancyModel):
    """Oracle model that knows the exact plans of the agents."""

    def __init__(self, width, height, agents, token_getter):
        super().__init__(width, height, agents)
        self.token_getter = token_getter

    def fit(self, agent_paths):
        pass

    def predict(self, location, time):
        agent_plans = self.token_getter()['agents'].values()
        agent_pos_at_t = list(map(
            lambda plan: tuple(plan[time]) if time < len(plan) else tuple(plan[-1]),
            agent_plans))
        if location in agent_pos_at_t:
            return 1
        return 0

    def time_forward(self, free_locations=[], seen_agents={}):
        pass

class MarkovianOccupancyModel(OccupancyModel):
    """Markovian occupancy model."""

    def __init__(self, width, height, agents, obstacles=None, order=1,
        cache_size=None, prune_thres=None, shared=True, backoff=False,
        init_policy='random'):
        super().__init__(width, height, agents)

        self.order = order
        self.shared = shared
        self.backoff = backoff

        # Sizing cache
        cache_size_bytes = cache_size if type(cache_size) == int else 0
        if type(cache_size) == str:
            cache_size_bytes = Cache.human2bytes(cache_size)

        # Equally share cache between occupancy model and the n markov chains
        self.cache_entries = min(
            cache_size_bytes // (self.width * self.height * 8 * (1 + self.n_agents)),
            500)
        logging.info('Occupancy model: n. of aggregate prediction cache entries %d' % self.cache_entries)
        self.cache_entries = max(self.cache_entries, 1) # At least one cache entry
        cache_size_bytes -= self.cache_entries * self.width * self.height * 8

        self.mc_cache_size = cache_size_bytes
        if not shared:
            self.mc_cache_size //= self.n_agents

        self._not_fitted = True
        self._predict_cache = [None] * self.cache_entries

        if shared:
            # Agents share the same markov chain
            mc = MarkovChain(
                self.width, self.height, obstacles=obstacles,
                cache_size=self.mc_cache_size, prune_thres=prune_thres, order=self.order,
                init_policy=init_policy
            )
            self.agent_models = {
                name: mc for name in self.agent_names
            }
        else:
            self.agent_models = {
                    name:
                    MarkovChain(
                        self.width, self.height, obstacles=obstacles,
                        cache_size=self.mc_cache_size, prune_thres=prune_thres, order=self.order,
                        init_policy=init_policy
                    )
                for name in self.agent_names
            }

        # Initial belief is uniform over the whole map
        self.agent_beliefs = {
            agent: self.agent_models[agent].get_initial_belief()
            for agent in self.agent_names
        }

    def fit(self, agent_paths, set_initial_belief=False):
        """Fit the transition probabilities of the Markov chains."""

        # Tick
        tock = self.tick()

        for agent, paths in agent_paths.items():
            # Split into contiguous chunks
            chunk_idxs = [
                i + 1 for i, (prev, cur) in enumerate(zip(paths, paths[1:]))
                if cur['t'] != prev['t'] + 1
            ]

            contiguous_chunks = [
                paths[start:end]
                for start, end in zip([0] + chunk_idxs, chunk_idxs + [len(paths)])
            ]

            # Fit Markov chain
            for chunk in contiguous_chunks:
                path = list(map(lambda p: (p['x'], p['y']), chunk))
                self.agent_models[agent].fit(path)

            # Initial belief from last known locations
            if set_initial_belief:
                self.agent_beliefs[agent] = \
                    self.agent_models[agent].get_initial_belief(path)

            if self.backoff:
                self.agent_models[agent].backoff()

        self._not_fitted = False

        # Tock
        self.fit_time += self.tick() - tock

    def predict(self, location, time, is_replanning=False):
        """Return the probability of occupancy of a location at time t."""

        if self._not_fitted:
            raise NotFittedError("Model not fitted")

        # Tick
        tock = self.tick()

        prob = self.compute_t_step_prediction(time)[tuple(location)]

        # Tock
        if not is_replanning:
            self.inference_time += self.tick() - tock
        else:
            self.inference_time_for_replan += self.tick() - tock

        return prob

    def time_forward(self, free_locations=[], seen_agents={}):
        """Collect observations coming from guests."""

        if self._not_fitted:
            raise NotFittedError("Model not fitted")

        # Tick
        tock = self.tick()

        # Flush cache
        self._predict_cache = [None] * self.cache_entries


        for agent_name, model in self.agent_models.items():
            agent_location = seen_agents.get(agent_name, None)
            initial_state = self.agent_beliefs[agent_name]
            new_belief = model.time_forward(initial_state, free_locations=free_locations, busy_location=agent_location)
            self.agent_beliefs[agent_name] = new_belief

        # Tock
        self.inference_time += self.tick() - tock

    def compute_t_step_prediction(self, t):
        """"Compute and cache the t-step model prediction."""

        cache_index = t % self.cache_entries
        cache_entry = self._predict_cache[cache_index]
        if cache_entry and cache_entry['time'] == t:
            # Cache hit
            return cache_entry['prediction']
        else:
            # Cache miss
            beliefs = []
            for agent in self.agent_names:
                markov_chain = self.agent_models[agent]
                initial_state = self.agent_beliefs[agent].copy()
                agent_state = markov_chain.predict(initial_state, t)
                agent_belief = markov_chain.state_to_belief(agent_state)
                beliefs.append(agent_belief)

            prediction = MarkovianOccupancyModel.aggregate_beliefs(beliefs)
            self._predict_cache[cache_index] = {
                'time': t, 'prediction': prediction
            }
            logging.debug("Occupancy model: aggregate predictions cache miss at time %d" % t)
            return prediction

    def load(self, file):
        """Load agent models from file."""

        filename, extension = path.splitext(file)
        if extension == '':
            extension = '.npz'

        if not self.shared:
            for agent_name, model in self.agent_models.items():
                file = filename + '_' + agent_name + extension
                model.load(file)
            return

        file = filename + extension
        for model in self.agent_models.values():
            model.load(file)

        self._not_fitted = False

    def save(self, file):
        """Save agent models to file."""

        filename, extension = path.splitext(file)
        if extension == '':
            extension = '.npz'

        if not self.shared:
            for agent_name, model in self.agent_models.items():
                file = filename + '_' + agent_name + extension
                model.save(file)
            return

        file = filename + extension
        model = next((m for m in self.agent_models.values()))
        model.save(file)

    def get_obs_counts(self):
        if self.shared:
            model = next((m for m in self.agent_models.values()))
            return model.get_obs_counts()
        return sum([m.get_obs_counts() for m in self.agent_models.values()])

    @staticmethod
    def aggregate_beliefs(beliefs):
        """Return a probability of occupancy from single agent beliefs."""

        aggregate = np.array([*beliefs])
        aggregate = np.ones_like(aggregate) - aggregate
        # Workaround to prevent log(1 - p) = log(0) when p = 1
        where_log_zero = np.where(np.min(aggregate, axis=(0, )) <= 0)
        aggregate = np.where(aggregate <= 0, 1e-5, aggregate) # Lower threshold
        aggregate = np.sum(np.log(aggregate), axis=(0, ))
        aggregate = np.ones_like(aggregate) - np.exp(aggregate)
        aggregate[where_log_zero] = 1
        return aggregate
