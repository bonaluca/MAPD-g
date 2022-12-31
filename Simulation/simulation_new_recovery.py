import logging
import time
import random
from math import fabs
import networkx as nx
from Simulation.CBS.cbs import CBS, DynamicEnvironment
from Simulation.occupancy_model import NotFittedError

class PathNotFoundError(ValueError):
    """Exception class to raise if path not found."""

class SimulationNewRecovery(object):
    random.seed(1234)
    def __init__(self, tasks, tasks_guest, agents, guests, occupancy_model, alpha):
        self.tasks = tasks #tasks of the agents
        self.tasks_guest = tasks_guest #tasks of the guests
        self.agents = agents #the agents
        self.guests = guests #the guests
        self.occupancy_model = occupancy_model #the occupancy model of the agents
        self.alpha = alpha
        self.time = 0 #initialization of the time steps of the algorithm
        self.start_times = [] #it is a list that cointains the starting time of the tasks
        self.start_times_guests = [] #it is a list that cointains the starting time of the tasks of the guests
        self.agents_pos_now = set() #it is a set that contains the current position of the agents
        self.agents_moved = set() #it is a set that contains the agents that have been moved
        self.guests_pos_now = set() #it is a set that contains the actual position of the guests
        self.guests_moved = set() #it is a set that contains the guests that have been moved
        self.actual_paths = {} #it is a dictionary that keeps track step by step of the actual paths of each agent
        self.actual_paths_guests = {} #it is a dictionary that keeps track step by step of the actual paths of each guest
        self.algo_time = 0 #it keeps track of the time (in seconds) that the tp algorithm employs for each iteration
        self.n_replanning_guest = 0 #it counts the number of times a guest has to change its path in order to avoid an agent
        self.to_replan_from_start = []
        self.to_replan_from_goal = []
        self.initialize_simulation()

    def initialize_simulation(self):
        for t in self.tasks: #we add in start_times the starting time of each taks of the agents
            self.start_times.append(t['start_time'])
        for t in self.tasks_guest: #we add in start_times the starting time of each taks of the guests
            self.start_times_guests.append(t['start_time'])
        for agent in self.agents: #we update the actual paths of the agents by adding their current initial position
            self.actual_paths[agent['name']] = [{'t': 0, 'x': agent['start'][0], 'y': agent['start'][1]}]
        for guest in self.guests: #we update the actual paths of the guests by adding their current initial position
            self.actual_paths_guests[guest['name']] = [{'t': 0, 'x': guest['start'][0], 'y': guest['start'][1]}]



    # time_forward is the method in which the agents are effectly moved and the time is shifted one step ahead
    def time_forward(self, algorithm, dimensions, non_task_endpoints_guests, obstacles, obstacles_agents, obstacles_guests, a_star_max_iter,observation_time):

        # TODO @bonaluca: this should not be in here
        if self.time == 0:
            logging.info('Residential matrix inizialization')
            elist_agents = []
            for i in range(0,dimensions[0]):
                for j in range(0,dimensions[1]):
                    if (i,j) not in obstacles_agents:
                        if (i+1,j) not in obstacles_agents and i+1 < dimensions[0]:
                            elist_agents.append(((i,j),(i+1,j),1))
                        if (i-1,j) not in obstacles_agents and i-1 > -1:
                            elist_agents.append(((i,j),(i-1,j),1))
                        if (i,j-1) not in obstacles_agents and j-1 > -1:
                            elist_agents.append(((i,j),(i,j-1),1))
                        if (i,j+1) not in obstacles_agents and j+1 < dimensions[1]:
                            elist_agents.append(((i,j),(i,j+1),1))
                        if (i,j) not in obstacles_agents:
                            elist_agents.append(((i,j),(i,j),1))
            self.G_agents = nx.MultiDiGraph()
            self.G_agents.add_weighted_edges_from(elist_agents)

            #print(elist_agents)
            logging.debug(self.G_agents)

            logging.info('Guest matrix inizialization')
            elist_guest = []
            for i in range(0,dimensions[0]):
                for j in range(0,dimensions[1]):
                    if (i,j) not in obstacles_guests:
                        if (i+1,j) not in obstacles_guests and i+1 < dimensions[0]:
                            elist_guest.append(((i,j),(i+1,j), 1))
                        if (i-1,j) not in obstacles_guests and i-1 > -1:
                            elist_guest.append(((i,j),(i-1,j), 1))
                        if (i,j-1) not in obstacles_guests and j-1 > -1:
                            elist_guest.append(((i,j),(i,j-1), 1))
                        if (i,j+1) not in obstacles_guests and j+1 < dimensions[1]:
                            elist_guest.append(((i,j),(i,j+1), 1))
                        if (i,j) not in obstacles_guests:
                            elist_guest.append(((i,j),(i,j), 1))
            self.G_guests = nx.MultiDiGraph()
            self.G_guests.add_weighted_edges_from(elist_guest)

            diam_g = nx.diameter(self.G_guests)

            logging.debug(self.G_agents)
            logging.debug('Graph diameter: %s' % diam_g)

#            # Weight function that balances distance and probability of occupancy
#            self.weight_function = lambda dist, prob_occ: \
#                1 / max((1 - prob_occ) ** (self.alpha), 5e-4)

            # Weight function that balances distance and probability of occupancy
            if self.alpha != 1:
                self.weight_function = lambda dist, prob_occ: \
                    ((1 - self.alpha) * dist) / diam_g + self.alpha * prob_occ
            else:
                # Prevent zero cost edges
                self.weight_function = lambda dist, prob_occ: \
                    max(prob_occ, 5e-4)

        self.time = self.time + 1
        logging.info('Time: %s', self.time)


        #keep track of the effective time in seconds to perform the tp algorithm
        start_time = time.time()
        algorithm.time_forward() # the time_forward recalled here is the method defined in TP
        self.algo_time += time.time() - start_time

        #for each iteration, we define the current positions of agents and guests as empty sets
        self.agents_pos_now = set()
        self.guests_pos_now = set()
        #at the beginning of each iteration, no guests or agents have been moved yet
        self.agents_moved = set()
        self.guests_moved = set()
        #the order of the agents and guests to move is decided by shuffling them
        #agents and guests simply contain all agents and all guests in a random order
        agents = self.agents
        random.shuffle(agents)
        guests = self.guests
        random.shuffle(guests)

        idle_guests = [guest for guest in guests \
                if len(algorithm.get_token()['guests'][guest['name']]) == 1]
        moving_guests = [guest for guest in guests \
                if len(algorithm.get_token()['guests'][guest['name']]) > 1]

        #we create for the agents and for the guests a dictionary in which each agent/guest (key) is associated to a list (value)
        #containing the current position of the agent/guest and the next position

        to_add = []
        for guest_ep in self.guests:
            current_guest_pos_ep = self.actual_paths_guests[guest_ep['name']][-1] #current_guest_pos tells us the current position of the guest
            if tuple([current_guest_pos_ep['x'], current_guest_pos_ep['y']]) in non_task_endpoints_guests:
                to_add.append(tuple([current_guest_pos_ep['x'], current_guest_pos_ep['y']]))
            if guest_ep['name'] in algorithm.get_token()['guests_to_tasks']:
                if algorithm.get_token()['guests_to_tasks'][guest_ep['name']]['goal'] in non_task_endpoints_guests:
                    to_add.append(tuple(algorithm.get_token()['guests_to_tasks'][guest_ep['name']]['goal']))
        algorithm.get_token()['occupied_non_task_endpoints_guests'] = set(to_add)

        current_pos_agents = lambda agent_name: \
            tuple(algorithm.get_token()['agents'][agent_name][0])
        current_pos_guests = lambda guest_name: \
            tuple(algorithm.get_token()['guests'][guest_name][0])
        next_pos_agents = lambda agent_name: \
            tuple(algorithm.get_token()['agents'][agent_name][:2][-1])
        next_pos_guests = lambda guest_name: \
            tuple(algorithm.get_token()['guests'][guest_name][:2][-1])

        # Compute surroundings
        location_within_grid = lambda loc: \
            0 <= loc[0] < dimensions[0] and 0 <= loc[1] < dimensions[1]
        surroundings_of = lambda loc: set(filter(
            lambda loc: location_within_grid(loc) and not loc in obstacles,
            [
                (loc[0]-2, loc[1]), (loc[0]-1, loc[1]-1), (loc[0]-1, loc[1]), (loc[0]-1, loc[1]+1),
                (loc[0], loc[1]-2), (loc[0], loc[1]-1), (loc[0], loc[1]), (loc[0], loc[1]+1), (loc[0], loc[1]+2),
                (loc[0]+1, loc[1]-1), (loc[0]+1, loc[1]), (loc[0]+1, loc[1]+1), (loc[0]+2, loc[1])
            ]
        ))
        neighbors = lambda loc: [
            (loc[0]-1, loc[1]), (loc[0], loc[1]-1), (loc[0], loc[1]), (loc[0], loc[1]+1), (loc[0]+1, loc[1])
        ]
        surroundings = lambda guest_name: \
            surroundings_of(current_pos_guests(guest_name))

        agents_nearby = lambda guest_name: \
            set([
                agent['name'] for agent in agents
                if current_pos_agents(agent['name']) in surroundings(guest_name)
            ])

        #NOW GUESTS
        for guest in moving_guests:
            current_guest_pos = self.actual_paths_guests[guest['name']][-1] #current_guest_pos tells us the current position of the guest
            #self.guests_pos_now.add(tuple([current_guest_pos['x'], current_guest_pos['y']])) #guest current position update

            #if the guest does not have any task assigned and is not in an endpoint, we send it to an endpoint
            #if guest['name'] not in algorithm.get_token()['guests_to_tasks'].keys() and tuple([current_guest_pos['x'], current_guest_pos['y']]) not in non_task_endpoints_guests:
            #    all_idle_guests = algorithm.get_token()['guests'].copy()
            #    all_idle_guests.pop(guest['name'])
            #    algorithm.go_to_closest_non_task_endpoint_guest3(guest['name'], [current_guest_pos['x'], current_guest_pos['y']], all_idle_guests)

            # TODO @bonaluca: more readable but breaks compatibility
            possible_moves_bak = list(filter(
                lambda loc: location_within_grid(loc) and not loc in obstacles_guests,
                neighbors(current_pos_guests(guest['name']))
            ))

            possible_moves = [tuple([current_guest_pos['x'], current_guest_pos['y']]), tuple([current_guest_pos['x']+1, current_guest_pos['y']]),
                                    tuple([current_guest_pos['x']-1, current_guest_pos['y']]), tuple([current_guest_pos['x'], current_guest_pos['y']+1]),
                                    tuple([current_guest_pos['x'], current_guest_pos['y']-1])]

            #remove move if outside the grid or on an obstacle
            to_remove_am = []
            for move_am in possible_moves:
                if (move_am in obstacles_guests) or (move_am[0]<0) or (move_am[1]<0) or (move_am[0]>dimensions[0]-1) or (move_am[1]>dimensions[1]-1):
                    to_remove_am.append(move_am)


            x_new, y_new = algorithm.get_token()['guests'][guest['name']][1]

            #we consider each agent: if its current position is in the surrounding of the guest, we check if its next move is the same as the guest
            #if so, we change the path of the guest
            for agent in agents_nearby(guest['name']):
                #if the next move of the agent is the next move of the guest
                if algorithm.next_pos_agents(agent) == (x_new, y_new):
                    logging.info('VERTEX-CONFLICT with a moving guest (%d, %d) %s', x_new, y_new, algorithm.next_pos_agents(agent))

                # case in which the agent and the guest exchange their positions
                if algorithm.current_pos_agents(agent) == (x_new, y_new) and \
                    algorithm.next_pos_agents(agent) == algorithm.current_pos_guests(guest['name']):
                    logging.info('EDGE-CONFLICT with a moving guest')

            for move in possible_moves:
                #remove move if it is the same as the agent next move
                for agent1 in self.agents:
                    if move == algorithm.next_pos_agents(agent1['name']):
                        to_remove_am.append(move)
                #remove move if there is an exchange between guest and another agent positions
                for agent1 in self.agents:
                    if move == algorithm.current_pos_agents(agent1['name']) and \
                        algorithm.next_pos_agents(agent1['name']) == algorithm.current_pos_guests(guest['name']):
                        to_remove_am.append(move)

#            accepted_moves = list(set(possible_moves) - set(to_remove_am))
            accepted_moves = possible_moves.copy()
            for move_am in list(set(to_remove_am)):
                accepted_moves.remove(move_am)

            if not accepted_moves:
                logging.error('DEADLOCK!2 (%d, %d)', x_new, y_new)
                logging.error(algorithm.get_token()['guests_to_tasks'][guest['name']])
                exit(1)

            # Filter out vertex conflicting guest moves (for get_closest_move)
            accepted_moves = list(filter(
                lambda move: move not in [
                    algorithm.next_pos_guests(other_guest['name'])
                    for other_guest in guests if other_guest['name'] != guest['name']
                ], accepted_moves
            ))

            # Filter out edge conflicting guest moves (for get_closest_move)
            accepted_moves = list(filter(
                lambda move: move not in [
                    algorithm.current_pos_guests(other_guest['name'])
                    for other_guest in guests
                    if other_guest['name'] != guest['name'] and
                    algorithm.next_pos_guests(other_guest['name']) == algorithm.current_pos_guests(guest['name'])
                ], accepted_moves
            ))

            # TODO @bonaluca: this deadlock could be recovered, since it involves only guests
            if not accepted_moves:
                logging.error('DEADLOCK!2 (%d, %d)', x_new, y_new)
                logging.error(algorithm.get_token()['guests_to_tasks'][guest['name']])
                exit(1)

            if (x_new, y_new) in accepted_moves:
                continue

            # TODO @bonaluca: why? can't we just let the low level planner do the whole job?
            if algorithm.get_token()['guests_to_tasks'][guest['name']]['start'] in algorithm.get_token()['guests'][guest['name']]:
                move = self.get_closest_move(accepted_moves, algorithm.get_token()['guests_to_tasks'][guest['name']]['start'])
            else:
                move = self.get_closest_move(accepted_moves, algorithm.get_token()['guests_to_tasks'][guest['name']]['goal'])
            #move = random.choice(possible_moves)
            self.n_replanning_guest += 1
            x_new, y_new = move

#            try:
#                guests_to_replan = [guest]
#                first_moves = {guest['name']: (x_new, y_new)}
#                self.replan2(algorithm, guests_to_replan, first_moves)
#            except PathNotFoundError:
#                break

            try:
                self.replan(algorithm, guest, first_move=(x_new, y_new))
            except PathNotFoundError:
                break

        for guest in idle_guests:
            current_guest_pos = self.actual_paths_guests[guest['name']][-1]

            x_new, y_new = algorithm.get_token()['guests'][guest['name']][0]

            possible_moves = [tuple([current_guest_pos['x'], current_guest_pos['y']]), tuple([current_guest_pos['x']+1, current_guest_pos['y']]),
                            tuple([current_guest_pos['x']-1, current_guest_pos['y']]), tuple([current_guest_pos['x'], current_guest_pos['y']+1]),
                            tuple([current_guest_pos['x'], current_guest_pos['y']-1])]

            #remove move if outside the grid or on an obstacle
            to_remove = []
            for move in possible_moves:
                if (move in obstacles_guests) or (move[0]<0) or (move[1]<0) or (move[0]>dimensions[0]-1) or (move[1]>dimensions[1]-1):
                    to_remove.append(move)

            #we consider each agent: if its current position is in the surrounding of the guest, we check if its next move is the same as the guest
            #if so, we change the path of the guest
            for agent in agents_nearby(guest['name']):
                if algorithm.next_pos_agents(agent) == (x_new, y_new): #if the next move of the agent is the next move of the guest
                    logging.info('VERTEX-CONFLICT with an idle guest',tuple([x_new, y_new]), algorithm.next_pos_agents(agent['name']))

            for guest1 in self.guests:
                if guest1['name'] != guest['name']:
                    if guest1['name'] in algorithm.get_token()['guests_to_tasks'].keys():
                        if move == tuple(algorithm.get_token()['guests_to_tasks'][guest1['name']]['goal']):
                            to_remove.append(move)

            # case in which the agent and the guest exchange their positions
            for agent in agents_nearby(guest['name']):
                if algorithm.current_pos_agents(agent) == (x_new, y_new) and \
                    algorithm.next_pos_agents(agent) == algorithm.current_pos_guests(guest['name']):
                    logging.info('EDGE-CONFLICT with an idle guest')
            for move in possible_moves:
                #remove move if it is the same as the agent next move
                for agent1 in self.agents:
                    if move == algorithm.next_pos_agents(agent1['name']):
                        to_remove.append(move)
#                # TODO @bonaluca: this never happens
#                #remove move if there is an exchange between guest and another agent positions
#                for agent1 in self.agents:
#                    if move == algorithm.current_pos_agents(agent1['name']) and \
#                        algorithm.next_pos_agents(agent1['name']) == algorithm.current_pos_guests(guest['name']):
#                        to_remove.append(move)

            for move in list(set(to_remove)):
                possible_moves.remove(move)

            # TODO @bonaluca
            # Filter out vertex conflicting guest moves (for random.choice)
            accepted_moves = list(filter(
                lambda move: move not in [
                    algorithm.next_pos_guests(other_guest['name'])
                    for other_guest in guests if other_guest['name'] != guest['name']
                ], possible_moves
            ))

            if (x_new, y_new) in accepted_moves:
                continue

            if not accepted_moves:
                logging.error('DEADLOCK ERROR! (%d, %d)', x_new, y_new)
                logging.error(algorithm.get_token()['guests_to_tasks'][guest['name']])
                exit(1)

            # if the presence of conflicts is true, we change the next move of the guest (choosing a random one) and replan
            move = random.choice(accepted_moves)
            logging.info('chosen move %s: (%d, %d)', guest['name'], move[0], move[1])
            x_new, y_new = move
            self.n_replanning_guest += 1

            algorithm.get_token()['guests'][guest['name']].append([x_new, y_new])

        # Move the guests
        for guest in guests:
            current_guest_pos = self.actual_paths_guests[guest['name']][-1]
            guest_plan = algorithm.get_token()['guests'][guest['name']]
            if len(guest_plan) <= 1:
                # Guest is idle
                (x_new, y_new) = (current_guest_pos['x'], current_guest_pos['y'])
            else:
                # Guest has an assigned task
                (x_new, y_new) = tuple(guest_plan[1])
                algorithm.get_token()['guests'][guest['name']] = guest_plan[1:]
            self.actual_paths_guests[guest['name']].append({'t': self.time, 'x': x_new, 'y': y_new})
            self.guests_moved.add(guest['name'])
            self.guests_pos_now.add((x_new, y_new))

        # Move the agents
        for agent in agents:
            current_agent_pos = self.actual_paths[agent['name']][-1]
            agent_plan = algorithm.get_token()['agents'][agent['name']]
            if len(agent_plan) <= 1:
                # Agent is idle
                (x_new, y_new) = (current_agent_pos['x'], current_agent_pos['y'])
            else:
                # Agent has an assigned task
                (x_new, y_new) = tuple(agent_plan[1])
                algorithm.get_token()['agents'][agent['name']] = agent_plan[1:]
            self.actual_paths[agent['name']].append({'t': self.time, 'x': x_new, 'y': y_new})
            self.agents_moved.add(agent['name'])
            self.agents_pos_now.add((x_new, y_new))

        # Occupancy model time forward
        visible_locations = set.union(
            *(algorithm.surroundings(guest['name']) for guest in guests)
        )

        seen_agents = {
            agent_name: algorithm.current_pos_agents(agent_name)
            for agent_name in set.union(
                *(algorithm.agents_nearby(guest['name']) for guest in guests)
            )
        }
        free_locations = visible_locations - set(seen_agents.values())

        try:
            self.occupancy_model.time_forward(free_locations=free_locations, seen_agents=seen_agents)
        except NotFittedError as e:
            pass

        return

    def replan2(self, algorithm, guests, first_moves):
        """Replan for a group of guests altogether."""

        #self.n_replanning_guest += len(guests)
        tasks = {
            guest['name']: algorithm.get_token()['guests_to_tasks'][guest['name']]
            for guest in guests
        }
        new_tokens = {
            guest['name']: [list(algorithm.current_pos_guests(guest['name']))]
            for guest in guests
        }
        reached_goal = lambda guest_name: \
            tasks[guest_name]['goal'] == new_tokens[guest_name][-1]
        total_cost = {}

        guests_list = [
            {
                'name': guest['name'],
                'start': first_moves[guest['name']],
                'goal': tasks[guest['name']]['start'] \
                    if tasks[guest['name']]['start'] in algorithm.get_token()['guests'][guest['name']][1:] \
                    else tasks[guest['name']]['goal']
            }
            for guest in guests
        ]

        other_guests_token = algorithm.get_token()['guests'].copy()
        for guest in guests:
            other_guests_token.pop(guest['name'])

        time_start = 1

        # Replan by taking care of existing guests' paths
        moving_obstacles_guests = algorithm.get_moving_obstacles(other_guests_token, time_start)
        idle_obstacles_guests = algorithm.get_idle_obstacles(other_guests_token.values(), time_start)
        env = DynamicEnvironment(
            algorithm.dimensions, guests_list, set(algorithm.obstacles_guests) | idle_obstacles_guests,
            moving_obstacles_guests, algorithm.a_star_max_iter,
            graph=self.get_graph_guests(),
            time_start=time_start,
            weight_function=self.weight_function,
            occupancy_model=self.occupancy_model
        )
        cbs = CBS(env)
        new_plans = algorithm.search(cbs)
        if not new_plans:
            logging.info('REPLANNING - Solution not found')
            raise PathNotFoundError()

        logging.info('REPLANNING - Solution found')
        for guest in guests:
            new_plan = {guest['name']: new_plans[guest['name']]}
            total_cost[guest['name']] = env.compute_solution_cost(new_plan)
            # Append path to token
            for el in new_plan[guest['name']]:
                new_tokens[guest['name']].append([el['x'], el['y']])

        guests_list = [
            {
                'name': guest['name'],
                'start': tasks[guest['name']]['start'],
                'goal': tasks[guest['name']]['goal']
            }
            for guest in guests if not reached_goal(guest['name'])
        ]

        for guest in guests_list:
            other_guests_token = algorithm.get_token()['guests'].copy()
            other_guests_token.pop(guest['name'])
            time_start = total_cost[guest['name']]
            moving_obstacles_guests = algorithm.get_moving_obstacles(other_guests_token, time_start)
            idle_obstacles_guests = algorithm.get_idle_obstacles(other_guests_token.values(), time_start)
            env = DynamicEnvironment(
                algorithm.dimensions, [guest], set(algorithm.obstacles_guests) | idle_obstacles_guests,
                moving_obstacles_guests, algorithm.a_star_max_iter,
                graph=self.get_graph_guests(),
                time_start=time_start,
                weight_function=self.weight_function,
                occupancy_model=self.occupancy_model
            )
            cbs = CBS(env)
            new_plan = algorithm.search(cbs)

            if not new_plan:
                logging.info('REPLANNING - Solution not found')
                raise PathNotFoundError()

            logging.info('REPLANNING - Solition found')
            total_cost[guest['name']] += env.compute_solution_cost(new_plan)

            # Append path to token
            for el in new_plan[guest['name']][1:]:
                new_tokens[guest['name']].append([el['x'], el['y']])

        for guest in guests:
            last_step = new_tokens[guest['name']][-1]
            algorithm.update_ends_guests(algorithm.current_pos_guests(guest['name']))
            algorithm.get_token()['path_ends_guest'].add(tuple(last_step))
            algorithm.get_token()['guests_to_tasks'][guest['name']] = {
                'task_name': tasks[guest['name']]['task_name'],
                'start': tasks[guest['name']]['start'],
                'goal': tasks[guest['name']]['goal'],
                'predicted_cost': total_cost[guest['name']]
            }
            # Update token
            algorithm.get_token()['guests'][guest['name']] = new_tokens[guest['name']]
            pass

    def replan(self, algorithm, guest, first_move):
        """Replan for a guest agent."""

        time_start = 1
        task = algorithm.get_token()['guests_to_tasks'][guest['name']]
        #new_token = [list(first_move), ]
        new_token = [list(algorithm.current_pos_guests(guest['name'])), ]
        start_location = first_move
        total_cost = 0
        all_idle_guests = algorithm.get_token()['guests'].copy()
        all_idle_guests.pop(guest['name'])

        if task['start'] in algorithm.get_token()['guests'][guest['name']][1:]:

            #we first replan a path from the new location to the pick-up point
            moving_obstacles_guests = algorithm.get_moving_obstacles(all_idle_guests, time_start)
            idle_obstacles_guests = algorithm.get_idle_obstacles(all_idle_guests.values(), time_start)
            guest = {'name': guest['name'], 'start': start_location, 'goal': task['start']}
            env = DynamicEnvironment(
                algorithm.dimensions, [guest], set(algorithm.obstacles_guests) | idle_obstacles_guests,
                moving_obstacles_guests, algorithm.a_star_max_iter,
                graph=self.get_graph_guests(),
                time_start=time_start,
                weight_function=self.weight_function,
                occupancy_model=self.occupancy_model
            )
            cbs = CBS(env)
            path_to_task_start = algorithm.search(cbs)
            if not path_to_task_start:
                logging.info("REPLANNING 1 Solution not found to task start for guest %s idling at current position...", guest['name'])
                raise PathNotFoundError()

            logging.info("REPLANNING 1 Solution found to task start for guest %s searching solution to task goal...", guest['name'])
            cost1 = env.compute_solution_cost(path_to_task_start)
            for el in path_to_task_start[guest['name']][:-1]:
                new_token.append([el['x'], el['y']])
            total_cost += cost1
            time_start = cost1
            start_location = task['start']

        #once the path to the pick-up point has been found, we replan a path to the delivery point
        moving_obstacles_guests = algorithm.get_moving_obstacles(all_idle_guests, time_start)
        idle_obstacles_guests = algorithm.get_idle_obstacles(all_idle_guests.values(), time_start)
        guest = {'name': guest['name'], 'start': start_location, 'goal': task['goal']}
        env = DynamicEnvironment(
            algorithm.dimensions, [guest], algorithm.obstacles_guests | idle_obstacles_guests,
            moving_obstacles_guests, a_star_max_iter=algorithm.a_star_max_iter,
            graph=self.get_graph_guests(),
            time_start=time_start,
            weight_function=self.weight_function,
            occupancy_model=self.occupancy_model
        )
        cbs = CBS(env)
        path_to_task_goal = algorithm.search(cbs)
        if not path_to_task_goal:
            logging.info("REPLANNING 2 - Solution not found to task goal for guest %s idling at current position...", guest['name'])
            raise PathNotFoundError()

        cost2 = env.compute_solution_cost(path_to_task_goal)
        total_cost += cost2
        logging.info("REPLANNING - Solution found to task goal for guest %s doing task...", guest['name'])
        last_step = path_to_task_goal[guest['name']][-1]
        algorithm.update_ends_guests(algorithm.current_pos_guests(guest['name']))
        algorithm.get_token()['path_ends_guest'].add(tuple([last_step['x'], last_step['y']]))
        algorithm.get_token()['guests_to_tasks'][guest['name']] = {'task_name': task['task_name'], 'start': task['start'],
                                                    'goal': task['goal'], 'predicted_cost': total_cost}
        for el in path_to_task_goal[guest['name']]:
            new_token.append([el['x'], el['y']])
        algorithm.get_token()['guests'][guest['name']] = new_token
        pass

    def replan_new(self, guest, algorithm):
        """."""

        time_start = 0
        task = algorithm.get_token()['guests_to_tasks'][guest['name']]
        start_location = list(algorithm.current_pos_guests(guest['name']))
        new_token = []
        total_cost = 0
        all_idle_guests = algorithm.get_token()['guests'].copy()
        all_idle_guests.pop(guest['name'])

        nearby_agent_tokens = {
            agent: [algorithm.current_pos_agents(agent), algorithm.next_pos_agents(agent)]
            for agent in algorithm.agents_nearby(guest['name'])
        }

        if task['start'] in algorithm.get_token()['guests'][guest['name']][1:]:

            #we first replan a path from the new location to the pick-up point

            # Replan by taking care of existing guests' paths
            moving_obstacles_guests = algorithm.get_moving_obstacles(all_idle_guests, time_start)
            idle_obstacles_guests = algorithm.get_idle_obstacles(all_idle_guests.values(), time_start)

            # Replan by taking care of nearby-agents' paths (at most 2 steps in the future)
            moving_obstacles_agents = algorithm.get_moving_obstacles(nearby_agent_tokens, time_start)
            # Remove idle-like obstacles
            moving_obstacles_agents = dict(filter(
                lambda item: item[0][2] >= 0, moving_obstacles_agents.items()
            ))
            # Joining the two moving obstacles dictionaries
            moving_obstacles = {}
            moving_obstacles.update(moving_obstacles_guests.items())
            moving_obstacles.update(moving_obstacles_agents.items())

            guest = {'name': guest['name'], 'start': start_location, 'goal': task['start']}
            env = DynamicEnvironment(
                algorithm.dimensions, [guest], set(algorithm.obstacles_guests) | idle_obstacles_guests,
                moving_obstacles, algorithm.a_star_max_iter,
                graph=self.get_graph_guests(),
                time_start=time_start,
                weight_function=self.weight_function,
                occupancy_model=self.occupancy_model
            )
            cbs = CBS(env)
            path_to_task_start = algorithm.search(cbs)
            if not path_to_task_start:
                logging.info("REPLANNING 1 Solution not found to task start for guest %s idling at current position...", guest['name'])
                raise PathNotFoundError()

            logging.info("REPLANNING 1 Solution found to task start for guest %s searching solution to task goal...", guest['name'])
            cost1 = env.compute_solution_cost(path_to_task_start)
            for el in path_to_task_start[guest['name']][:-1]:
                new_token.append([el['x'], el['y']])
            total_cost += cost1
            time_start = cost1 - 1
            start_location = task['start']
            nearby_agent_tokens = {} # Don't consider agents after first replan

        #once the path to the pick-up point has been found, we replan a path to the delivery point
        moving_obstacles_guests = algorithm.get_moving_obstacles(all_idle_guests, time_start)
        idle_obstacles_guests = algorithm.get_idle_obstacles(all_idle_guests.values(), time_start)

        # Replan by taking care of nearby-agents' paths (at most 2 steps in the future)
        moving_obstacles_agents = algorithm.get_moving_obstacles(nearby_agent_tokens, time_start)
        # Remove idle-like obstacles
        moving_obstacles_agents = dict(filter(
            lambda item: item[0][2] >= 0, moving_obstacles_agents.items()
        ))
        # Joining the two moving obstacles dictionaries
        moving_obstacles = dict()
        moving_obstacles.update(moving_obstacles_guests.items())
        moving_obstacles.update(moving_obstacles_agents.items())

        guest = {'name': guest['name'], 'start': start_location, 'goal': task['goal']}
        env = DynamicEnvironment(algorithm.dimensions, [guest], algorithm.obstacles_guests | idle_obstacles_guests,
            moving_obstacles, a_star_max_iter=algorithm.a_star_max_iter, time_start=time_start,
            graph=self.get_graph_guests(),
            weight_function=self.weight_function,
            occupancy_model=self.occupancy_model)
        cbs = CBS(env)
        path_to_task_goal = algorithm.search(cbs)
        if not path_to_task_goal:
            logging.info("REPLANNING 2 - Solution not found to task goal for guest %s idling at current position...", guest['name'])
            raise PathNotFoundError()

        cost2 = env.compute_solution_cost(path_to_task_goal)
        total_cost += cost2
        logging.info("REPLANNING - Solution found to task goal for guest %s doing task...", guest['name'])
        last_step = path_to_task_goal[guest['name']][-1]
        algorithm.update_ends_guests(algorithm.current_pos_guests(guest['name']))
        algorithm.get_token()['path_ends_guest'].add(tuple([last_step['x'], last_step['y']]))
        algorithm.get_token()['guests_to_tasks'][guest['name']] = {'task_name': task['task_name'], 'start': task['start'],
                                                    'goal': task['goal'], 'predicted_cost': total_cost}
        for el in path_to_task_goal[guest['name']]:
            new_token.append([el['x'], el['y']])
        algorithm.get_token()['guests'][guest['name']] = new_token

    def get_time(self):
        return self.time

    def get_algo_time(self):
        return self.algo_time

    def get_actual_paths(self):
        return self.actual_paths

    def get_actual_paths_guests(self):
        return self.actual_paths_guests

    def get_new_tasks(self):
        new = []
        for t in self.tasks:
            if t['start_time'] == self.time:
                new.append(t)
        return new

    def get_new_tasks_guest(self):
        new = []
        for t in self.tasks_guest:
            if t['start_time'] == self.time:
                new.append(t)
        return new

    def get_n_replanning_guest(self):
        return self.n_replanning_guest

    def get_graph_agents(self):
        return self.G_agents

    def get_graph_guests(self):
        return self.G_guests

    def admissible_heuristic(self, move, goal_pos):
        return fabs(move[0] - goal_pos[0]) + fabs(move[1] - goal_pos[1])

    #given a set of tasks, it returns the closest one according to the heuristic chosen
    def get_closest_move(self, possible_moves, goal_pos):
        closest = possible_moves[0]
        dist = self.admissible_heuristic(closest, goal_pos)
        for move in possible_moves:
            if self.admissible_heuristic(move, goal_pos) < dist:
                closest = move
        return closest

