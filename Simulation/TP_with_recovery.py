"""
Python implementation of Token Passing algorithms to solve MAPD problems
author: Giacomo Lodigiani (@Lodz97)
"""
from math import fabs
import random
from Simulation.CBS.cbs import CBS, Environment, DynamicEnvironment
import logging
from collections import defaultdict

class TokenPassingRecovery(object):
    random.seed(1234)
    def __init__(self, agents, guests, dimesions, obstacles_agents, obstacles_guests, 
                non_task_endpoints, non_task_endpoints_guests, simulation, a_star_max_iter=4000, 
                k=0,pd=None, p_max=1, p_iter=1, new_recovery=False):
        self.agents = agents
        self.guests = guests
        self.dimensions = dimesions
        self.obstacles_agents = set(obstacles_agents)
        self.obstacles_guests = set(obstacles_guests)
        self.non_task_endpoints = non_task_endpoints
        self.non_task_endpoints_guests = non_task_endpoints_guests
        if len(agents) > len(non_task_endpoints) or len(guests) > len(non_task_endpoints_guests):
            logging.error('There are more agents and guests than non task endpoints, instance is not well-formed.')
            exit(1)
        # TODO: Check all properties for well-formedness
        self.token = {}
        self.simulation = simulation
        self.a_star_max_iter = a_star_max_iter
        self.k = k
        self.p_max = p_max
        if self.p_max != 1:
            logging.error('p_max must be 1.')
            exit(-1)
        if self.k != 0 and self.p_max != 1:
            logging.error('Use of k and p robustness at same time not allowed.')
            exit(-1)
        if self.k < 0:
            logging.error('k should be >= 0.')
            exit(1)
        if self.k == 0 and not new_recovery:
            logging.error('k = 0 not supported for this recovery type.')
            exit(1)
        if self.k == 0:
            if (self.p_max < 0 or self.p_max > 1):
                logging.error('Max conflict probability must be between 0 and 1.')
                exit(1)
            self.pd = pd
            if self.p_max != 1 and not self.pd:
                logging.error('To use p robustness need to set pd.')
                exit(1)
            if self.pd and (pd < 0 or pd > 1):
                logging.error('Probability of must be between 0 and 1.')
                exit(1)
            self.p_iter = p_iter
            if self.p_iter <= 0:
                logging.error('p_iter should be > 0.')
                exit(1)
        else:
            self.p_max = 1
            self.pd = None
            self.p_iter = 1
        self.new_recovery = new_recovery
        self.init_token()

    def init_token(self):
        #it contains the current planned path for each agent / guest
        self.token['agents'] = {}
        self.token['guests'] = {}
        #it contains info about the tasks of agents / guests
        self.token['tasks'] = {}
        self.token['tasks_guest'] = {}
        #it contains info about the start times of tasks of agents / guests
        self.token['start_tasks_times'] = {}
        self.token['start_tasks_times_guest'] = {}
        #it contains info about the completion times of tasks of agents / guests
        self.token['completed_tasks_times'] = {}
        self.token['completed_tasks_times_guest'] = {}
        self.token['assigned_tasks_times_guest'] = {}
        #initialization of the tasks start times of agents / guests
        for t in self.simulation.get_new_tasks():
            self.token['tasks'][t['task_name']] = [t['start'], t['goal']]
            self.token['start_tasks_times'][t['task_name']] = self.simulation.get_time()
        for t in self.simulation.get_new_tasks_guest():
            self.token['tasks_guest'][t['task_name']] = [t['start'], t['goal']]
            self.token['start_tasks_times_guest'][t['task_name']] = self.simulation.get_time()
        #it contains information about the assignment of tasks to agents / guests
        self.token['agents_to_tasks'] = {}
        self.token['guests_to_tasks'] = {}
        #number of tasks completed by agents / guests
        self.token['completed_tasks'] = 0
        self.token['completed_tasks_guest'] = 0
        #number of times a total replan is necessary
        self.token['n_replans'] = 0
        #it contains information about the ends (estremità) of each path: initial point, pick-up, delivery (?)
        self.token['path_ends'] = set()
        self.token['path_ends_guest'] = set()
        #it contains the endpoints that are occupied by agents / guests
        self.token['occupied_non_task_endpoints'] = set()
        self.token['occupied_non_task_endpoints_guests'] = set()
        #information necessary to perform the total replan (recovery)
        self.token['agent_at_end_path'] = []
        self.token['agent_at_end_path_pos'] = []
        self.token['agents_in_recovery_trial'] = []
        self.token['guest_at_end_path'] = []
        self.token['guest_at_end_path_pos'] = []
        self.token['guests_in_recovery_trial'] = []
        #initialization of path_ends with the starting positions of agents / guests
        for a in self.agents:
            self.token['agents'][a['name']] = [a['start']]
            self.token['path_ends'].add(tuple(a['start']))
        for a in self.guests:
            self.token['guests'][a['name']] = [a['start']]
            self.token['path_ends_guest'].add(tuple(a['start']))
        self.token['prob_exceeded'] = False
        #number of times each agent / guest is in a deadlock (0 by default)
        self.token['deadlock_count_per_agent'] = defaultdict(lambda: 0)
        self.token['deadlock_count_per_guest'] = defaultdict(lambda: 0)

    #by idle agents / guests are intended those that do not have a task
    def get_idle_agents(self):
        agents = {}
        for name, path in self.token['agents'].items():
            if len(path) == 1: #their path has length 1 since they are still in the same position
                agents[name] = path
        return agents

    def get_idle_guests(self):
        guests = {}
        for name, path in self.token['guests'].items():
            if name not in self.token['guests_to_tasks'].keys():
                if len(path) == 1:
                    guests[name] = path
        return guests

    #the admissible heuristic chosen in the Manhattan distance
    def admissible_heuristic(self, task_pos, agent_pos):
        """The admissible heuristic chosen is the Manhattan distance."""
        return fabs(task_pos[0] - agent_pos[0]) + fabs(task_pos[1] - agent_pos[1])

    #given a set of tasks, it returns the closest one according to the heuristic chosen
    def get_closest_task_name(self, available_tasks, agent_pos):
        """Given a set of tasks, return the closest one according to the heuristic chosen."""
        closest = list(available_tasks.keys())[0]
        dist = self.admissible_heuristic(available_tasks[closest][0], agent_pos)
        for task_name, task in available_tasks.items():
            if self.admissible_heuristic(task[0], agent_pos) < dist:
                closest = task_name
        return closest

    def get_moving_obstacles(self, agents, time_start):
        """Return the paths of moving agents as obstacles."""
        obstacles = {}
        for name, path in agents.items():
            if len(path) > time_start and len(path) > 1:
                for i in range(time_start, len(path)):
                    k = i - time_start
                    obstacles[(path[i][0], path[i][1], i)] = name
                    for j in range(1, self.k + 1):
                        if i - j >= time_start:
                            obstacles[(path[i][0], path[i][1], i - j)] = name
                        obstacles[(path[i][0], path[i][1], i + j)] = name
                    # Mark last element with negative time to later turn it into idle obstacle
                    if i == len(path) - 1:
                        obstacles[(path[i][0], path[i][1], -i)] = name
        return obstacles

    def get_idle_obstacles(self, agents_paths, time_start):
        """Return the positions of the agents that are still as obstacles."""
        obstacles = set()
        for path in agents_paths:
            if 0 <= len(path) - 1 <= time_start:
                obstacles.add((path[-1][0], path[-1][1]))
        return obstacles

    def check_safe_idle(self, agent_pos):
        """Check if the agent is on a cell that is the pick-up / delivery of a task."""
        for task in self.token['tasks'].items():
            if tuple(task[0]) == tuple(agent_pos) or tuple(task[1]) == tuple(agent_pos):
                return False
        for start_goal in self.get_agents_to_tasks_starts_goals():
            if tuple(start_goal) == tuple(agent_pos):
                return False
        return True

    def check_safe_idle_guest(self, agent_pos):
        """Check if the guest is on a cell that is the pick-up / delivery of a task."""
        for task in self.token['tasks_guest'].items():
            if tuple(task[0]) == tuple(agent_pos) or tuple(task[1]) == tuple(agent_pos):
                return False
        for start_goal in self.get_guests_to_tasks_starts_goals():
            if tuple(start_goal) == tuple(agent_pos):
                return False
        return True

    def get_closest_non_task_endpoint(self, agent_pos):
        """Return the closest endpoint according to the defined heuristic."""
        dist = -1
        res = -1
        for endpoint in self.non_task_endpoints:
            if endpoint not in self.token['occupied_non_task_endpoints']:
                if dist == -1:
                    dist = self.admissible_heuristic(endpoint, agent_pos)
                    res = endpoint
                else:
                    tmp = self.admissible_heuristic(endpoint, agent_pos)
                    if tmp < dist:
                        dist = tmp
                        res = endpoint
        if res == -1:
            logging.error('Error in finding non-task endpoint, is instance well-formed?')
            exit(1)
        return res

    def get_closest_non_task_endpoint_guests(self, agent_pos):
        """Return the closest endpoint according to the defined heuristic."""
        dist = -1
        res = -1
        for endpoint in self.non_task_endpoints_guests:
            if endpoint not in self.token['occupied_non_task_endpoints_guests']:
                if dist == -1:
                    dist = self.admissible_heuristic(endpoint, agent_pos)
                    res = endpoint
                else:
                    tmp = self.admissible_heuristic(endpoint, agent_pos)
                    if tmp < dist:
                        dist = tmp
                        res = endpoint
        if res == -1:
            logging.error('Error in finding non-task endpoint guests, is instance well-formed?')
            exit(1)
        return res

    def update_ends(self, agent_pos):
        """Update of path ends for agents."""
        if tuple(agent_pos) in self.token['path_ends']:
            self.token['path_ends'].remove(tuple(agent_pos))
        elif tuple(agent_pos) in self.token['occupied_non_task_endpoints']:
            self.token['occupied_non_task_endpoints'].remove(tuple(agent_pos))        

    def update_ends_guests(self, agent_pos):
        """Update of path ends for guests."""
        if tuple(agent_pos) in self.token['path_ends_guest']:
            self.token['path_ends_guest'].remove(tuple(agent_pos))
        elif tuple(agent_pos) in self.token['occupied_non_task_endpoints_guests']:
            self.token['occupied_non_task_endpoints_guests'].remove(tuple(agent_pos))

    def get_agents_to_tasks_goals(self):
        goals = set()
        for el in self.token['agents_to_tasks'].values():
            goals.add(tuple(el['goal']))
        return goals

    def get_guests_to_tasks_goals(self):
        goals = set()
        for el in self.token['guests_to_tasks'].values():
            goals.add(tuple(el['goal']))
        return goals

    def get_agents_to_tasks_starts_goals(self):
        """Return the start and goal locations of all occupied agents."""
        starts_goals = set()
        for el in self.token['agents_to_tasks'].values():
            starts_goals.add(tuple(el['goal']))
            starts_goals.add(tuple(el['start']))
        return starts_goals

    def get_guests_to_tasks_starts_goals(self):
        """Return the start and goal locations of all occupied guests."""
        starts_goals = set()
        for el in self.token['guests_to_tasks'].values():
            starts_goals.add(tuple(el['goal']))
            starts_goals.add(tuple(el['start']))
        return starts_goals

    def get_completed_tasks(self):
        return self.token['completed_tasks']

    def get_completed_tasks_guest(self):
        return self.token['completed_tasks_guest']

    def get_completed_tasks_times(self):
        return self.token['completed_tasks_times']

    def get_completed_tasks_times_guest(self):
        return self.token['completed_tasks_times_guest']

    def get_assigned_tasks_times_guest(self):
        return self.token['assigned_tasks_times_guest']

    def get_n_replans(self):
        return self.token['n_replans']

    def get_token(self):
        return self.token

    def get_k(self):
        return self.k

    def search(self, cbs):
        path = None
        if self.p_max == 1:
            path = cbs.search()
        else:
            logging.error("p_max must be 1")
        return path

    def go_to_closest_non_task_endpoint(self, agent_name, agent_pos, all_idle_agents):
        closest_non_task_endpoint = self.get_closest_non_task_endpoint(agent_pos)
        moving_obstacles_agents = self.get_moving_obstacles(self.token['agents'], 0)
        idle_obstacles_agents = self.get_idle_obstacles(all_idle_agents.values(), 0)
        agent = {'name': agent_name, 'start': agent_pos, 'goal': closest_non_task_endpoint}
        env = Environment(self.dimensions, [agent], self.obstacles_agents | idle_obstacles_agents, moving_obstacles_agents,
                          a_star_max_iter=self.a_star_max_iter, graph=self.simulation.get_graph_agents())
        cbs = CBS(env)
        path_to_non_task_endpoint = self.search(cbs)
        if not path_to_non_task_endpoint:
            logging.error("Solution to non-task endpoint not found for agent %s instance is not well-formed.", agent_name)
            self.deadlock_recovery(agent_name, agent_pos, all_idle_agents, 4)
            # exit(1)
        else:
            logging.info('No available task for agent %s moving to safe idling position...', agent_name)
            self.update_ends(agent_pos)
            self.token['occupied_non_task_endpoints'].add(tuple(closest_non_task_endpoint))
            self.token['agents_to_tasks'][agent_name] = {'task_name': 'safe_idle', 'start': agent_pos,
                                                         'goal': closest_non_task_endpoint, 'predicted_cost': 0}
            self.token['agents'][agent_name] = []
            for el in path_to_non_task_endpoint[agent_name]:
                self.token['agents'][agent_name].append([el['x'], el['y']])

    def go_to_closest_non_task_endpoint_guest(self, guest_name, guest_pos, all_idle_guests):
        closest_non_task_endpoint = self.get_closest_non_task_endpoint_guests(guest_pos)
        j = self.token['guests'].copy()
        j.pop(guest_name)
        moving_obstacles_guests = self.get_moving_obstacles(j, 0)
        idle_obstacles_guests = self.get_idle_obstacles(all_idle_guests.values(), 0)
        guest = {'name': guest_name, 'start': guest_pos, 'goal': closest_non_task_endpoint}
        env = DynamicEnvironment(self.dimensions, [guest], self.obstacles_guests | idle_obstacles_guests, moving_obstacles_guests,
                          a_star_max_iter=self.a_star_max_iter, graph=self.simulation.get_graph_guests(),
                          weight_function=self.simulation.weight_function,
                          occupancy_model=self.simulation.occupancy_model)
        cbs = CBS(env)
        path_to_non_task_endpoint = self.search(cbs)
        if not path_to_non_task_endpoint:
            logging.error("Solution to non-task endpoint not found for guest %s instance is not well-formed.1", guest_name)
            self.deadlock_recovery_guest(guest_name, guest_pos, all_idle_guests, 4)
            # exit(1)
        else:
            logging.info('No available task for guest %s moving to safe idling position...', guest_name)
            self.update_ends_guests(guest_pos)
            self.token['occupied_non_task_endpoints_guests'].add(tuple(closest_non_task_endpoint))
            self.token['guests_to_tasks'][guest_name] = {'task_name': 'safe_idle', 'start': guest_pos,
                                                         'goal': closest_non_task_endpoint, 'predicted_cost': 0}
            self.token['guests'][guest_name] = []
            for el in path_to_non_task_endpoint[guest_name]:
                self.token['guests'][guest_name].append([el['x'], el['y']])

    def go_to_closest_non_task_endpoint_guest2(self, guest_name, guest_pos, all_idle_guests):
        closest_non_task_endpoint = self.get_closest_non_task_endpoint_guests(guest_pos)
        j = self.token['guests'].copy()
        j.pop(guest_name)
        moving_obstacles_guests = self.get_moving_obstacles(j, 0)
        idle_obstacles_guests = self.get_idle_obstacles(all_idle_guests.values(), 0)
        guest = {'name': guest_name, 'start': guest_pos, 'goal': closest_non_task_endpoint}
        env = DynamicEnvironment(self.dimensions, [guest], self.obstacles_guests | idle_obstacles_guests, moving_obstacles_guests,
                          a_star_max_iter=self.a_star_max_iter, graph=self.simulation.get_graph_guests(),
                          weight_function=self.simulation.weight_function,
                          occupancy_model=self.simulation.occupancy_model)
        cbs = CBS(env)
        path_to_non_task_endpoint = self.search(cbs)
        if not path_to_non_task_endpoint:
            logging.error("Solution to non-task endpoint not found for guest %s instance is not well-formed.2", guest_name)
            self.deadlock_recovery_guest(guest_name, guest_pos, all_idle_guests, 4)
            # exit(1)
        else:
            logging.info('No available task for guest %s moving to safe idling position2...', guest_name)
            self.update_ends_guests(guest_pos)
            self.token['occupied_non_task_endpoints_guests'].add(tuple(closest_non_task_endpoint))
            self.token['guests_to_tasks'][guest_name] = {'task_name': 'safe_idle', 'start': guest_pos,
                                                         'goal': closest_non_task_endpoint, 'predicted_cost': 0}
            self.token['guests'][guest_name] = []
            self.token['guests'][guest_name].append(guest_pos)
            for el in path_to_non_task_endpoint[guest_name]:
                self.token['guests'][guest_name].append([el['x'], el['y']])

    def go_to_closest_non_task_endpoint_guest3(self, guest_name, guest_pos, all_idle_guests):
        closest_non_task_endpoint = self.get_closest_non_task_endpoint_guests(guest_pos)
        j = self.token['guests'].copy()
        j.pop(guest_name)
        moving_obstacles_guests = self.get_moving_obstacles(self.token['guests'].copy(), 0)
        idle_obstacles_guests = self.get_idle_obstacles(all_idle_guests.values(), 0)
        guest = {'name': guest_name, 'start': guest_pos, 'goal': closest_non_task_endpoint}
        env = DynamicEnvironment(self.dimensions, [guest], self.obstacles_guests | idle_obstacles_guests, moving_obstacles_guests,
                          a_star_max_iter=self.a_star_max_iter, graph=self.simulation.get_graph_guests(),
                          weight_function=self.simulation.weight_function,
                          occupancy_model=self.simulation.occupancy_model)
        cbs = CBS(env)
        path_to_non_task_endpoint = self.search(cbs)
        if not path_to_non_task_endpoint:
            logging.error("Solution to non-task endpoint not found for guest %s instance is not well-formed.3", guest_name)
            self.deadlock_recovery_guest(guest_name, guest_pos, all_idle_guests, 4)
            # exit(1)
        else:
            logging.info('No available task for guest %s moving to safe idling position3...', guest_name)
            self.update_ends_guests(guest_pos)
            self.token['occupied_non_task_endpoints_guests'].add(tuple(closest_non_task_endpoint))
            self.token['guests_to_tasks'][guest_name] = {'task_name': 'safe_idle', 'start': guest_pos,
                                                         'goal': closest_non_task_endpoint, 'predicted_cost': 0}
            self.token['guests'][guest_name] = []
            for el in path_to_non_task_endpoint[guest_name]:
                self.token['guests'][guest_name].append([el['x'], el['y']])

        # Update 'guests_to_tasks' with 'safe_idle' task
        self.token['guests_to_tasks'][guest_name] = {
            'task_name': 'safe_idle',
            'start': guest_pos,
            'goal': closest_non_task_endpoint,
            'predicted_cost': 0
        }

    # def get_random_close_cell_guest(self, guest_pos, r):
    #     while True:
    #         cell = (guest_pos[0] + random.choice(range(-r - 1, r + 1)), guest_pos[1] + random.choice(range(-r - 1, r + 1)))
    #         if cell not in self.obstacles_guests and cell not in self.token['path_ends_guest'] and \
    #                 cell not in self.token['occupied_non_task_endpoints_guests'] \
    #                 and cell not in self.get_guests_to_tasks_goals() \
    #                 and 0 <= cell[0] < self.dimensions[0] and 0 <= cell[1] < self.dimensions[1]:
    #             return cell

    # def get_random_close_cell(self, agent_pos, r):
    #     while True:
    #         cell = (agent_pos[0] + random.choice(range(-r - 1, r + 1)), agent_pos[1] + random.choice(range(-r - 1, r + 1)))
    #         if cell not in self.obstacles_agents and cell not in self.token['path_ends'] and \
    #                 cell not in self.token['occupied_non_task_endpoints'] \
    #                 and cell not in self.get_agents_to_tasks_goals() \
    #                 and 0 <= cell[0] < self.dimensions[0] and 0 <= cell[1] < self.dimensions[1]:
    #             return cell

    # def deadlock_recovery(self, agent_name, agent_pos, all_idle_agents, r):
    #     self.token['deadlock_count_per_agent'][agent_name] += 1
    #     if self.token['deadlock_count_per_agent'][agent_name] >= 5:
    #         self.token['deadlock_count_per_agent'][agent_name] = 0
    #         random_close_cell = self.get_random_close_cell(agent_pos, r)
    #         moving_obstacles_agents = self.get_moving_obstacles(self.token['agents'], 0)
    #         idle_obstacles_agents = self.get_idle_obstacles(all_idle_agents.values(), 0)
    #         agent = {'name': agent_name, 'start': agent_pos, 'goal': random_close_cell}
    #         env = Environment(self.dimensions, [agent], self.obstacles_agents | idle_obstacles_agents, moving_obstacles_agents,
    #                           a_star_max_iter=self.a_star_max_iter, graph=self.simulation.get_graph_agents())
    #         cbs = CBS(env)
    #         path_to_non_task_endpoint = self.search(cbs)
    #         if not path_to_non_task_endpoint:
    #             print("No solution to deadlock recovery for agent", agent_name, " retrying later.")
    #         else:
    #             # Don't consider this a task, so don't add to agents_to_tasks
    #             print('Agent', agent_name, 'causing deadlock, moving to safer position...')
    #             self.update_ends(agent_pos)
    #             self.token['agents'][agent_name] = []
    #             for el in path_to_non_task_endpoint[agent_name]:
    #                 self.token['agents'][agent_name].append([el['x'], el['y']])

    # def deadlock_recovery_guest(self, guest_name, guest_pos, all_idle_guests, r):
    #     self.token['deadlock_count_per_guest'][guest_name] += 1
    #     if self.token['deadlock_count_per_guest'][guest_name] >= 5:
    #         self.token['deadlock_count_per_guest'][guest_name] = 0
    #         random_close_cell = self.get_random_close_cell_guest(guest_pos, r)
    #         moving_obstacles_guests = self.get_moving_obstacles(self.token['guests'], 0)
    #         idle_obstacles_guests = self.get_idle_obstacles(all_idle_guests.values(), 0)
    #         guest = {'name': guest_name, 'start': guest_pos, 'goal': random_close_cell}
    #         env = Environment(self.dimensions, [guest], self.obstacles_guests | idle_obstacles_guests, moving_obstacles_guests,
    #                           a_star_max_iter=self.a_star_max_iter, graph=self.simulation.get_graph_guests())
    #         cbs = CBS(env)
    #         path_to_non_task_endpoint = self.search(cbs)
    #         if not path_to_non_task_endpoint:
    #             print("No solution to deadlock recovery for guest", guest_name, " retrying later.")
    #         else:
    #             # Don't consider this a task, so don't add to guests_to_tasks
    #             print('Guest', guest_name, 'causing deadlock, moving to safer position...')
    #             self.update_ends_guests(guest_pos)
    #             self.token['guests'][guest_name] = []
    #             for el in path_to_non_task_endpoint[guest_name]:
    #                 self.token['guests'][guest_name].append([el['x'], el['y']])

    @property
    def free_guests(self):
        return [guest for guest in self.guests \
            if len(self.token['guests'][guest['name']]) == 1]

    @property
    def occupied_guests(self):
        return [guest for guest in self.guests \
            if len(self.token['guests'][guest['name']]) > 1]

    def current_pos_agents(self, agent_name):
        return tuple(self.token['agents'][agent_name][0])

    def current_pos_guests(self, guest_name):
        return tuple(self.token['guests'][guest_name][0])

    def next_pos_agents(self, agent_name):
        return tuple(self.token['agents'][agent_name][:2][-1])

    def next_pos_guests(self, guest_name):
        return tuple(self.token['guests'][guest_name][:2][-1])

    def surroundings(self, guest_name):
        location_within_grid = lambda loc: \
            0 <= loc[0] < self.dimensions[0] and 0 <= loc[1] < self.dimensions[1]
        surroundings_of = lambda loc: set(filter(
            lambda loc: location_within_grid(loc) and not loc in self.obstacles_agents,
            [
                (loc[0]-2, loc[1]), (loc[0]-1, loc[1]-1), (loc[0]-1, loc[1]), (loc[0]-1, loc[1]+1),
                (loc[0], loc[1]-2), (loc[0], loc[1]-1), (loc[0], loc[1]), (loc[0], loc[1]+1), (loc[0], loc[1]+2),
                (loc[0]+1, loc[1]-1), (loc[0]+1, loc[1]), (loc[0]+1, loc[1]+1), (loc[0]+2, loc[1])
            ]
        ))
        return surroundings_of(self.current_pos_guests(guest_name))

    def agents_nearby(self, guest_name):
        return set([
            agent['name'] for agent in self.agents
            if self.current_pos_agents(agent['name']) in self.surroundings(guest_name)
        ])

    def time_forward(self):
        # Update completed tasks for the agents
        for agent_name in self.token['agents']:
            pos = self.simulation.actual_paths[agent_name][-1]
            if agent_name in self.token['agents_to_tasks'] \
                    and (pos['x'], pos['y']) == tuple(self.token['agents_to_tasks'][agent_name]['goal']) \
                    and len(self.token['agents'][agent_name]) == 1 \
                    and self.token['agents_to_tasks'][agent_name]['task_name'] != 'safe_idle':
                self.token['completed_tasks'] = self.token['completed_tasks'] + 1
                self.token['completed_tasks_times'][
                    self.token['agents_to_tasks'][agent_name]['task_name']] = self.simulation.get_time()
                self.token['agents_to_tasks'].pop(agent_name)
            if agent_name in self.token['agents_to_tasks'] \
                    and (pos['x'], pos['y']) == tuple(self.token['agents_to_tasks'][agent_name]['goal']) \
                    and len(self.token['agents'][agent_name]) == 1 \
                    and self.token['agents_to_tasks'][agent_name]['task_name'] == 'safe_idle':
                self.token['agents_to_tasks'].pop(agent_name)

        # Update completed tasks for the guest
        for guest_name in self.token['guests']:
            pos = self.simulation.actual_paths_guests[guest_name][-1]
            if guest_name in self.token['guests_to_tasks'] \
                    and (pos['x'], pos['y']) == tuple(self.token['guests_to_tasks'][guest_name]['goal']) \
                    and len(self.token['guests'][guest_name]) == 1 \
                    and self.token['guests_to_tasks'][guest_name]['task_name'] != 'safe_idle' \
                    and guest_name not in self.simulation.to_replan_from_start \
                    and guest_name not in self.simulation.to_replan_from_goal:
                self.token['completed_tasks_guest'] = self.token['completed_tasks_guest'] + 1
                self.token['completed_tasks_times_guest'][
                    self.token['guests_to_tasks'][guest_name]['task_name']] = self.simulation.get_time()
                self.token['guests_to_tasks'].pop(guest_name)
            if guest_name in self.token['guests_to_tasks'] \
                    and (pos['x'], pos['y']) == tuple(self.token['guests_to_tasks'][guest_name]['goal']) \
                    and len(self.token['guests'][guest_name]) == 1 \
                    and self.token['guests_to_tasks'][guest_name]['task_name'] == 'safe_idle':
                self.token['guests_to_tasks'].pop(guest_name)

        
        # AGENTS
        # TODO do this maybe only for old recovery
        # Check if somehow an agent path collides with an idle agent
        if not self.new_recovery:
            for name, path in self.get_idle_agents().items():
                self.token['agent_at_end_path'].append(name)
                self.token['agent_at_end_path_pos'].append(path[0])
            for name, path in self.token['agents'].items():
                if name not in self.token['agent_at_end_path']:
                    for i in range(len(path)):
                        if path[i] in self.token['agent_at_end_path_pos']:
                            logging.info('Agent %s will impact end task agent, replanning...', name)
                            # self.update_ends(path[-1])
                            if path[0] in self.non_task_endpoints:
                                self.token['occupied_non_task_endpoints'].add(tuple(path[0]))
                            else:
                                self.token['path_ends'].add(tuple(path[0]))
                            # TODO check this rare keyerror
                            self.token['agents'][name] = [path[0]]
                            break
            self.token['agent_at_end_path'] = []
            self.token['agent_at_end_path_pos'] = []

        #GUESTS
        # TODO do this maybe only for old recovery
        # Check if somehow a guest path collides with an idle guest
        if not self.new_recovery:
            for name, path in self.get_idle_guests().items():
                self.token['guest_at_end_path'].append(name)
                self.token['guest_at_end_path_pos'].append(path[0])
            for name, path in self.token['guests'].items():
                if name not in self.token['guest_at_end_path']:
                    for i in range(len(path)):
                        if path[i] in self.token['guest_at_end_path_pos']:
                            logging.info('Guest %s will impact end task guest, replanning...', name)
                            # self.update_ends(path[-1])
                            if path[0] in self.non_task_endpoints_guests:
                                self.token['occupied_non_task_endpoints_guests'].add(tuple(path[0]))
                            else:
                                self.token['path_ends_guest'].add(tuple(path[0]))
                            # TODO check this rare keyerror
                            self.token['guests'][name] = [path[0]]
                            break
            self.token['guest_at_end_path'] = []
            self.token['guest_at_end_path_pos'] = []

        # Collect new tasks and assign them, if possible
        for t in self.simulation.get_new_tasks():
            self.token['tasks'][t['task_name']] = [t['start'], t['goal']]
            self.token['start_tasks_times'][t['task_name']] = self.simulation.get_time()
        for t in self.simulation.get_new_tasks_guest():
            self.token['tasks_guest'][t['task_name']] = [t['start'], t['goal']]
            self.token['start_tasks_times_guest'][t['task_name']] = self.simulation.get_time()

        #AGENTS
        idle_agents = self.get_idle_agents() #prendo gli agenti che oziano
        while len(idle_agents) > 0:
            agent_name = list(idle_agents.keys())[0]
            # `agent_name` executes now
            all_idle_agents = self.token['agents'].copy()
            all_idle_agents.pop(agent_name)
            agent_pos = idle_agents.pop(agent_name)[0]

            # Collect tasks such that no other path in token ends in task_start or task_goal
            available_tasks = {}
            for task_name, task in self.token['tasks'].items():
                task_start, task_goal = tuple(task[0]), tuple(task[1])
                if task_start not in self.token['path_ends'].difference({tuple(agent_pos)}) \
                        and task_goal not in self.token['path_ends'].difference({tuple(agent_pos)}) \
                        and task_start not in self.get_agents_to_tasks_goals() \
                        and task_goal not in self.get_agents_to_tasks_goals():
                    available_tasks[task_name] = task

            if len(available_tasks) > 0 or agent_name in self.token['agents_to_tasks']:
                if agent_name not in self.token['agents_to_tasks']:
                    closest_task_name = self.get_closest_task_name(available_tasks, agent_pos)
                    closest_task = available_tasks[closest_task_name]
                else:
                    closest_task_name = self.token['agents_to_tasks'][agent_name]['task_name']
                    closest_task = [self.token['agents_to_tasks'][agent_name]['start'],
                            self.token['agents_to_tasks'][agent_name]['goal']]
                logging.info('task_assegnato agente %s', str(closest_task))
                moving_obstacles_agents = self.get_moving_obstacles(self.token['agents'], 0)
                idle_obstacles_agents = self.get_idle_obstacles(all_idle_agents.values(), 0)
                #print('moving_obstacles_agents PER START1',moving_obstacles_agents)
                #print('TOKEN',self.token['agents'])
                agent = {'name': agent_name, 'start': agent_pos, 'goal': closest_task[0]}
                env = Environment(self.dimensions, [agent], self.obstacles_agents | idle_obstacles_agents,
                                moving_obstacles_agents, a_star_max_iter=self.a_star_max_iter, graph=self.simulation.get_graph_agents())
                cbs = CBS(env)
                path_to_task_start = self.search(cbs)
                #print('path_to_task_start',path_to_task_start)
                if not path_to_task_start:
                    logging.info("Solution not found to task start for agent %s idling at current position...", agent_name)
                        # exit(1)
                else:
                    logging.info("Solution found to task start for agent %s searching solution to task goal...", agent_name)
                    #print('path_to_task_start',path_to_task_start)
                    cost1 = env.compute_solution_cost(path_to_task_start)
                    # Use cost - 1 because idle cost is 1
                    time_start = cost1 - 1
                    moving_obstacles_agents = self.get_moving_obstacles(self.token['agents'], time_start)
                    idle_obstacles_agents = self.get_idle_obstacles(all_idle_agents.values(),
                                                                        time_start)
                    #print('moving_obstacles_agents PER GOAL1',moving_obstacles_agents)
                    #print('TOKEN',self.token['agents'])
                    agent = {'name': agent_name, 'start': closest_task[0], 'goal': closest_task[1]}
                    env = Environment(self.dimensions, [agent], self.obstacles_agents | idle_obstacles_agents,
                                    moving_obstacles_agents, a_star_max_iter=self.a_star_max_iter,
                                    time_start=time_start, graph=self.simulation.get_graph_agents())
                    cbs = CBS(env)
                    path_to_task_goal = self.search(cbs)
                    #print('path_to_task_goal',path_to_task_goal)
                    if not path_to_task_goal:
                        logging.info("Solution not found to task goal for agent %s idling at current position...", agent_name)
                            # exit(1)
                    else:
                        logging.info("Solution found to task goal for agent %s doing task...", agent_name)
                        #print('path_to_task_goal',path_to_task_goal)
                        cost2 = env.compute_solution_cost(path_to_task_goal)
                        if agent_name not in self.token['agents_to_tasks']:
                            self.token['tasks'].pop(closest_task_name)
                            task = available_tasks.pop(closest_task_name)
                        else:
                            task = closest_task
                        last_step = path_to_task_goal[agent_name][-1]
                        self.update_ends(agent_pos)
                        self.token['path_ends'].add(tuple([last_step['x'], last_step['y']]))
                        self.token['agents_to_tasks'][agent_name] = {'task_name': closest_task_name, 'start': task[0],
                                                                    'goal': task[1], 'predicted_cost': cost1 + cost2}
                        self.token['agents'][agent_name] = []
                        for el in path_to_task_start[agent_name]:
                            self.token['agents'][agent_name].append([el['x'], el['y']])
                        # Don't repeat twice same step
                        self.token['agents'][agent_name] = self.token['agents'][agent_name][:-1]
                        for el in path_to_task_goal[agent_name]:
                            self.token['agents'][agent_name].append([el['x'], el['y']])
            elif self.check_safe_idle(agent_pos):
                logging.info('No available tasks for agent %s idling at current position...', agent_name)
            else:
                self.go_to_closest_non_task_endpoint(agent_name, agent_pos, all_idle_agents)

        #GUESTS
        idle_guests = self.get_idle_guests() #prendo gli agenti che oziano
        while len(idle_guests) > 0:
            guest_name = list(idle_guests.keys())[0]
            # `guest_name` executes now
            all_idle_guests = self.token['guests'].copy()
            all_idle_guests.pop(guest_name)
            guest_pos = idle_guests.pop(guest_name)[0]

            # Collect tasks such that no other path in token ends in task_start or task_goal
            available_tasks = {}
            for task_name, task in self.token['tasks_guest'].items():
                task_start, task_goal = tuple(task[0]), tuple(task[1])
                if task_start not in self.token['path_ends'].difference({tuple(guest_pos)}) \
                        and task_goal not in self.token['path_ends'].difference({tuple(guest_pos)}) \
                        and task_start not in self.get_guests_to_tasks_goals() \
                        and task_goal not in self.get_guests_to_tasks_goals():
                    available_tasks[task_name] = task

            if len(available_tasks) > 0 or guest_name in self.token['guests_to_tasks']:
                if guest_name not in self.token['guests_to_tasks']:
                    closest_task_name = self.get_closest_task_name(available_tasks, guest_pos)
                    closest_task = available_tasks[closest_task_name]
                else:
                    closest_task_name = self.token['guests_to_tasks'][guest_name]['task_name']
                    closest_task = [self.token['guests_to_tasks'][guest_name]['start'],
                            self.token['guests_to_tasks'][guest_name]['goal']]
                self.token['assigned_tasks_times_guest'][closest_task_name] = self.simulation.get_time()
                logging.info('task_assegnato %s', str(closest_task))
                moving_obstacles_guests = self.get_moving_obstacles(self.token['guests'], 0)
                idle_obstacles_guests = self.get_idle_obstacles(all_idle_guests.values(), 0)
                guest = {'name': guest_name, 'start': guest_pos, 'goal': closest_task[0]}
                env = DynamicEnvironment(self.dimensions, [guest], self.obstacles_guests | idle_obstacles_guests,
                                moving_obstacles_guests, a_star_max_iter=self.a_star_max_iter,
                                graph=self.simulation.get_graph_guests(),
                                weight_function=self.simulation.weight_function,
                                occupancy_model=self.simulation.occupancy_model)
                cbs = CBS(env)
                path_to_task_start = self.search(cbs)
                #print('path_to_task_start',path_to_task_start)
                if not path_to_task_start:
                    logging.info("Solution not found to task start for guest %s idling at current position...", guest_name)
                        # exit(1)
                else:
                    logging.info("Solution found to task start for guest %s searching solution to task goal...", guest_name)
                    #print('path_to_task_start',path_to_task_start)
                    cost1 = env.compute_solution_cost(path_to_task_start)
                    time_start = cost1 - 1
                    # Use cost - 1 because idle cost is 1
                    moving_obstacles_guests = self.get_moving_obstacles(self.token['guests'], time_start)
                    idle_obstacles_guests = self.get_idle_obstacles(all_idle_guests.values(),
                                                                        time_start)
                    guest = {'name': guest_name, 'start': closest_task[0], 'goal': closest_task[1]}
                    env = DynamicEnvironment(self.dimensions, [guest], self.obstacles_guests | idle_obstacles_guests,
                                    moving_obstacles_guests, a_star_max_iter=self.a_star_max_iter, time_start=time_start,
                                    graph=self.simulation.get_graph_guests(),
                                    weight_function=self.simulation.weight_function,
                                    occupancy_model=self.simulation.occupancy_model)
                    cbs = CBS(env)
                    path_to_task_goal = self.search(cbs)
                    #print('path_to_task_goal',path_to_task_goal)
                    if not path_to_task_goal:
                        logging.info("Solution not found to task goal for guest %s idling at current position...", guest_name)
                            # exit(1)
                    else:
                        logging.info("Solution found to task goal for guest %s doing task...", guest_name)
                        #print('path_to_task_goal',path_to_task_goal)
                        cost2 = env.compute_solution_cost(path_to_task_goal)
                        if guest_name not in self.token['guests_to_tasks']:
                            self.token['tasks_guest'].pop(closest_task_name)
                            task = available_tasks.pop(closest_task_name)
                        else:
                            task = closest_task
                        last_step = path_to_task_goal[guest_name][-1]
                        self.update_ends(guest_pos)
                        self.token['path_ends'].add(tuple([last_step['x'], last_step['y']]))
                        self.token['guests_to_tasks'][guest_name] = {'task_name': closest_task_name, 'start': task[0],
                                                                    'goal': task[1], 'predicted_cost': cost1 + cost2}
                        self.token['guests'][guest_name] = []
                        for el in path_to_task_start[guest_name]:
                            self.token['guests'][guest_name].append([el['x'], el['y']])
                        # Don't repeat twice same step
                        self.token['guests'][guest_name] = self.token['guests'][guest_name][:-1]
                        for el in path_to_task_goal[guest_name]:
                            self.token['guests'][guest_name].append([el['x'], el['y']])
            elif self.check_safe_idle(guest_pos):
                logging.info('No available tasks for guest %s idling at current position...', guest_name)
            else:
                self.go_to_closest_non_task_endpoint(guest_name, guest_pos, all_idle_guests)
            
        #GUESTS 
        
        # Advance along paths in the token agents
        if not self.new_recovery:
            for name, path in self.token['agents'].items():
                if len(path) > 1:
                    self.token['agents'][name] = path[1:]

        # Advance along paths in the token guests
        if not self.new_recovery:
            for name, path in self.token['guests'].items():
                if len(path) > 1:
                    self.token['guests'][name] = path[1:]
