import numpy as np
import logging
import re
from itertools import product
from scipy import sparse

class NotFittedError(ValueError):
    """Exception class to raise if model is used before fitting."""

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

    def __init__(self, width, height, obstacles=None, order=1, cache_size=None):
        self.width = width
        self.height = height
        self.n_states = self.width * self.height
        self.order = order
        self._not_fitted = True
        self.obstacles = obstacles or []

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
        col_ind = list(self.state_idx.values())
        row_ind = [
            self.state_idx[tuple((*state[1:], state[-1]))]
            for state in self.state_idx.keys()
        ]
        data = (self.n_valid_states) * [1, ]
        self.transition_matrix = sparse.csr_matrix(
            (data, (row_ind, col_ind)),
            shape=2 * (self.n_valid_states, ),
            dtype=np.float64
        )

        self.obs_counts = np.zeros((self.n_valid_states, ), dtype=int)

        # Lazy caching of the powers of the transition matrix
        self.cache = Cache(cache_size)
        self._last_computation = None

    def is_valid_transition(self, state, next_state):
        if state[1:] != next_state[:-1]:
            return False
        return next_state[-1] in self.neighbors(state[-1])

    def fit(self, sequence):
        """Fit the transition matrix on a squence of observations."""

        self.cache.flush()

        # Cumulating occurrences
        occurrences = sparse.dok_matrix(self.transition_matrix.shape, dtype=int)
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
        self.transition_matrix = self.transition_matrix.dot(diag_count_frac) + \
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
                step_t_mat = sparse.eye(self.n_valid_states,
                    dtype=np.float64, format="csr")
            else:
                step_t_minus_one = self.cache.retrieve(last_t_cached)
                step_t_mat = self.transition_matrix.dot(step_t_minus_one)
            self.cache.store(step_t_mat)
            last_t_cached = self.cache.last_t_stored()
        if self._last_computation and self._last_computation['time'] == t:
            return t, self._last_computation['matrix']
        if last_t_cached < 1: # Not using cache
            if only_cached:
                t = min(t, 1)
            return t, self.transition_matrix ** t
        if t <= last_t_cached: # Cache hit
            t_step_matrix = self.cache.retrieve(t)
            return t, t_step_matrix
        if not only_cached:
            last_cached_matrix = self.cache.retrieve(last_t_cached)
            if self._last_computation and t > self._last_computation['time'] > last_t_cached:
                last_t_cached = self._last_computation['time']
                last_cached_matrix = self._last_computation['matrix']
            t_step_matrix = self.transition_matrix ** (t - last_t_cached)
            t_step_matrix = t_step_matrix.dot(last_cached_matrix)
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

    def get_initial_belief(self, sequence):
        """Return a vector representing the initial belief of the markov chain."""

        n_fill = self.order - len(sequence)
        if n_fill > 0:
            sequence = n_fill * [sequence[0]] + sequence

        initial_state_idx = self.state_idx[tuple(sequence[-self.order:])]
        belief = np.zeros((self.n_valid_states, ), dtype=np.float64)
        belief[initial_state_idx] = 1
        return belief

class OccupancyModel(object):
    """Simple occupancy model."""

    def __init__(self, width, height, agents):
        self.width = width
        self.height = height
        self.agent_names = list(map(lambda a: a['name'], agents))
        self.n_agents = len(self.agent_names)

        self.prob_occ = np.zeros((self.width, self.height))

    def fit(self, agent_paths):
        """Estimate the probability of occupancy of the grid cells."""

        # Compute observation time
        max_t = max([max(path, key=lambda step: step['t'])['t'] \
            for path in agent_paths.values()])
        min_t = min([min(path, key=lambda step: step['t'])['t'] \
            for path in agent_paths.values()])
        observation_time = max_t - min_t + 1

        # Compute probability of occupancy
        for agent_path in agent_paths.values():
            for step in agent_path:
                self.prob_occ[step['x']][step['y']] += 1
        self.prob_occ /= observation_time

    def predict(self, location, time):
        """Return the probability of occupancy of a location at time t."""

        return self.prob_occ[location[0]][location[1]]

    def time_forward(self, free_locations=[], seen_agents={}):
        """Collect observations coming from guests."""

        pass

    def get_probability_matrix(self):
        return self.prob_occ.tolist()

    def get_prob_matrix_90deg(self):
        """Return the estimated probability of occupancy matrix.

        This method exists just for backward compatibility
        """
        return np.rot90(self.prob_occ, k=1, axes=(0, 1)).tolist()

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

    def __init__(self, width, height, agents, obstacles=None, order=1, cache_size=None, shared=True):
        super().__init__(width, height, agents)

        self.order = order
        self.shared = shared

        # Sizing cache
        cache_size_bytes = cache_size if type(cache_size) == int else 0
        if type(cache_size) == str:
            cache_size_bytes = Cache.human2bytes(cache_size)

        # Equally share cache between occupancy model and markov chains
        self.cache_entries = min(
            cache_size_bytes // (self.width * self.height * 8 * (1 + self.n_agents)),
            100)
        self.cache_entries = max(self.cache_entries, 1) # At least one cache entry
        cache_size_bytes -= self.cache_entries * self.width * self.height * 8

        self.mc_cache_size = cache_size_bytes
        if not shared:
            self.mc_cache_size //= self.n_agents

        self._not_fitted = True
        self._predict_cache = [None] * self.cache_entries

        if shared:
            # Agents share the same markov chain
            mc = MarkovChain(self.width, self.height, obstacles=obstacles, cache_size=self.mc_cache_size, order=self.order)
            self.agent_models = {
                name: mc for name in self.agent_names
            }
        else:
            self.agent_models = {
                    name:
                    MarkovChain(self.width, self.height, obstacles=obstacles, cache_size=self.mc_cache_size, order=self.order)
                for name in self.agent_names
            }

    def fit(self, agent_paths):
        """Fit the transition probabilities of the Markov chains."""

        self.agent_beliefs = {}

        for agent in agent_paths.keys():
            # Fitting Markov chains
            path = list(map(lambda p: (p['x'], p['y']), agent_paths[agent]))
            self.agent_models[agent].fit(path)

            # Initial belief from last known locations
            self.agent_beliefs[agent] = \
                self.agent_models[agent].get_initial_belief(path)

        self._not_fitted = False

    def predict(self, location, time):
        """Return the probability of occupancy of a location at time t."""

        if self._not_fitted:
            raise NotFittedError("Model not fitted")

        return self.compute_t_step_prediction(time)[tuple(location)]

    def time_forward(self, free_locations=[], seen_agents={}):
        """Collect observations coming from guests."""

        if self._not_fitted:
            raise NotFittedError("Model not fitted")

        # Flush cache
        self._predict_cache = [None] * self.cache_entries


        for agent_name, model in self.agent_models.items():
            agent_location = seen_agents.get(agent_name, None)
            initial_state = self.agent_beliefs[agent_name]
            new_belief = model.time_forward(initial_state, free_locations=free_locations, busy_location=agent_location)
            self.agent_beliefs[agent_name] = new_belief

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
            logging.debug("Occupancy model: cached aggregate predictions at time %d" % t)
            return prediction

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
