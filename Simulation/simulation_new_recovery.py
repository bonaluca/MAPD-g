import logging
import time
import random
from math import fabs
import networkx as nx
from Simulation.CBS.cbs import CBS, DynamicEnvironment
from Simulation.occupancy_model import NotFittedError


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

        free_guests = [guest for guest in guests \
                if len(algorithm.get_token()['guests'][guest['name']]) == 1]
        occupied_guests = [guest for guest in guests \
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

        now_next_agents = {}
        for agent_path in self.agents:
            if len(algorithm.get_token()['agents'][agent_path['name']]) == 1:
                now_next_agents[agent_path['name']] = [tuple(algorithm.get_token()['agents'][agent_path['name']][0]),tuple(algorithm.get_token()['agents'][agent_path['name']][0])]
            if len(algorithm.get_token()['agents'][agent_path['name']]) > 1:
                now_next_agents[agent_path['name']] = [tuple(algorithm.get_token()['agents'][agent_path['name']][0]),tuple(algorithm.get_token()['agents'][agent_path['name']][1])]

        now_next_guests = {}
        for guest_path in self.guests:
            if len(algorithm.get_token()['guests'][guest_path['name']]) == 1:
                now_next_guests[guest_path['name']] = [tuple(algorithm.get_token()['guests'][guest_path['name']][0]),tuple(algorithm.get_token()['guests'][guest_path['name']][0])]
            if len(algorithm.get_token()['guests'][guest_path['name']]) > 1:
                now_next_guests[guest_path['name']] = [tuple(algorithm.get_token()['guests'][guest_path['name']][0]),tuple(algorithm.get_token()['guests'][guest_path['name']][1])]

        # Compute surroundings
        surroundings = {}
        for guest in guests:
            current_guest_pos = self.actual_paths_guests[guest['name']][-1] #current_guest_pos tells us the current position of the guest

            #the list contains all the posisitions that are at most two step ahead from the current one
            surroundings[guest['name']] = [tuple([current_guest_pos['x'], current_guest_pos['y']]),tuple([current_guest_pos['x']+1, current_guest_pos['y']]),
            tuple([current_guest_pos['x']-1, current_guest_pos['y']]), tuple([current_guest_pos['x'], current_guest_pos['y']+1]),
            tuple([current_guest_pos['x'], current_guest_pos['y']-1]),tuple([current_guest_pos['x']+2, current_guest_pos['y']]),
            tuple([current_guest_pos['x']-2, current_guest_pos['y']]), tuple([current_guest_pos['x'], current_guest_pos['y']+2]),
            tuple([current_guest_pos['x'], current_guest_pos['y']-2]),tuple([current_guest_pos['x']+1, current_guest_pos['y']+1]),
            tuple([current_guest_pos['x']+1, current_guest_pos['y']-1]), tuple([current_guest_pos['x']-1, current_guest_pos['y']+1]),
            tuple([current_guest_pos['x']-1, current_guest_pos['y']-1])]

            #we remove every element in the surrounding that is outside of the grid or is an obstacle (we leave only allowed positions)
            to_remove_surr = []
            for el in surroundings[guest['name']]:
                if el[0]<0 or el[1]<0 or el[0]>dimensions[0]-1 or el[1]>dimensions[1]-1 or el in obstacles:
                    to_remove_surr.append(el)
            for el in to_remove_surr:
                surroundings[guest['name']].remove(el)

        #NOW GUESTS
        for guest in occupied_guests:
            current_guest_pos = self.actual_paths_guests[guest['name']][-1] #current_guest_pos tells us the current position of the guest
            #self.guests_pos_now.add(tuple([current_guest_pos['x'], current_guest_pos['y']])) #guest current position update

            #if the guest does not have any task assigned and is not in an endpoint, we send it to an endpoint
            #if guest['name'] not in algorithm.get_token()['guests_to_tasks'].keys() and tuple([current_guest_pos['x'], current_guest_pos['y']]) not in non_task_endpoints_guests:
            #    all_idle_guests = algorithm.get_token()['guests'].copy()
            #    all_idle_guests.pop(guest['name'])
            #    algorithm.go_to_closest_non_task_endpoint_guest3(guest['name'], [current_guest_pos['x'], current_guest_pos['y']], all_idle_guests)

            accepted_moves = [tuple([current_guest_pos['x'], current_guest_pos['y']]), tuple([current_guest_pos['x']+1, current_guest_pos['y']]),
                                    tuple([current_guest_pos['x']-1, current_guest_pos['y']]), tuple([current_guest_pos['x'], current_guest_pos['y']+1]),
                                    tuple([current_guest_pos['x'], current_guest_pos['y']-1])]

            #remove move if outside the grid or on an obstacle
            to_remove_am = []
            for move_am in accepted_moves:
                if (move_am in obstacles_guests) or (move_am[0]<0) or (move_am[1]<0) or (move_am[0]>dimensions[0]-1) or (move_am[1]>dimensions[1]-1):
                    to_remove_am.append(move_am)


            x_new, y_new = algorithm.get_token()['guests'][guest['name']][1]

            #we consider each agent: if its current position is in the surrounding of the guest, we check if its next move is the same as the guest
            #if so, we change the path of the guest
            for agent in self.agents:
                if now_next_agents[agent['name']][0] in surroundings[guest['name']]: #if the agent is at most two step ahead of the guest
                    if now_next_agents[agent['name']][1] == tuple([x_new, y_new]): #if the next move of the agent is the next move of the guest
                        presence_conflicts = True
                        logging.info('VERTEX-CONFLICT with an idle guest (%d, %d) %s' ,x_new, y_new, now_next_agents[agent['name']][1])
                        #remove move if it is the same as the agent next move
                        for move in accepted_moves:
                            for agent1 in self.agents:
                                if move == now_next_agents[agent1['name']][1]:
                                    to_remove_am.append(move)
                        #remove move if there is an exchange between guest and another agent positions
                            for agent1 in self.agents:
                                if move == now_next_agents[agent1['name']][0] and now_next_agents[agent1['name']][1] == tuple([current_guest_pos['x'], current_guest_pos['y']]):
                                    to_remove_am.append(move)
                        #remove move if the move is the same as the one of another guest
                            for guest1 in self.guests:
                                if guest1 != guest['name']:
                                    if move == now_next_guests[guest1['name']][1]:
                                        to_remove_am.append(move)
                        #remove move if there is an exchange between guest and another guest positions
                            for guest1 in self.guests:
                                if guest1 != guest['name']:
                                    if move == now_next_guests[guest1['name']][0] and now_next_guests[guest1['name']][1] == tuple([current_guest_pos['x'], current_guest_pos['y']]):
                                        to_remove_am.append(move)

            # case in which the agent and the guest exchange their positions
            for agent in self.agents:
                if now_next_agents[agent['name']][0] in surroundings[guest['name']]:
                    if tuple([x_new, y_new])== now_next_agents[agent['name']][0] and now_next_agents[agent['name']][1] == tuple([current_guest_pos['x'], current_guest_pos['y']]):
                        presence_conflicts = True
                        #print('EDGE-CONFLICT with an idle guest')
                        #remove move if it is the same as the agent next move
                        for move in accepted_moves:
                            for agent1 in self.agents:
                                if move == now_next_agents[agent1['name']][1]:
                                    to_remove_am.append(move)
                        #remove move if there is an exchange between guest and another agent positions
                            for agent1 in self.agents:
                                if move == now_next_agents[agent1['name']][0] and now_next_agents[agent1['name']][1] == tuple([current_guest_pos['x'], current_guest_pos['y']]):
                                    to_remove_am.append(move)
                        #remove move if the move is the same as the one of another guest
                            for guest1 in self.guests:
                                if guest1 != guest['name']:
                                    if move == now_next_guests[guest1['name']][1]:
                                        to_remove_am.append(move)
                        #remove move if there is an exchange between guest and another guest positions
                            for guest1 in self.guests:
                                if guest1 != guest['name']:
                                    if move == now_next_guests[guest1['name']][0] and now_next_guests[guest1['name']][1] == tuple([current_guest_pos['x'], current_guest_pos['y']]):
                            
                                        to_remove_am.append(move)

            for move_am in list(set(to_remove_am)):
                accepted_moves.remove(move_am)

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


            if algorithm.get_token()['guests_to_tasks'][guest['name']]['start'] in algorithm.get_token()['guests'][guest['name']][1:]:

                task = algorithm.get_token()['guests_to_tasks'][guest['name']]

                #we first replan a path from the new location to the pick-up point
                all_idle_guests = algorithm.get_token()['guests'].copy()
                all_idle_guests.pop(guest['name'])
                moving_obstacles_guests = algorithm.get_moving_obstacles(all_idle_guests, 1)
                #print(moving_obstacles_guests)
                #print('moving_obstacles_guests PER START1',moving_obstacles_guests)
                #print('TOKEN',algorithm.get_token()['guests'])
                idle_obstacles_guests = algorithm.get_idle_obstacles(all_idle_guests.values(), 1)
                #print('idle obstacle guests', idle_obstacles_guests)
                guest = {'name': guest['name'], 'start': [x_new, y_new], 'goal': task['start']}
                env = DynamicEnvironment(dimensions, [guest], set(obstacles_guests) | idle_obstacles_guests,
                                moving_obstacles_guests, a_star_max_iter, self.get_graph_guests(),
                                weight_function=self.weight_function,
                                occupancy_model=self.occupancy_model)
                cbs = CBS(env)
                path_to_task_start = algorithm.search(cbs)
                if not path_to_task_start:
                    logging.info("REPLANNING 1 Solution not found to task start for guest %s idling at current position...", guest['name'])
                    break
                    # exit(1)

                #once the path to the pick-up point has been found, we replan a path to the delivery point
                logging.info("REPLANNING 1 Solution found to task start for guest %s searching solution to task goal...", guest['name'])
                #print('path_to_task_start',path_to_task_start)
                cost1 = env.compute_solution_cost(path_to_task_start)

                time_start = cost1
                moving_obstacles_guests = algorithm.get_moving_obstacles(all_idle_guests, time_start)
                idle_obstacles_guests = algorithm.get_idle_obstacles(all_idle_guests.values(), time_start)
                #print('moving_obstacles_guests PER GOAL1',moving_obstacles_guests)
                #print(algorithm.get_token()['guests'])
                guest = {'name': guest['name'], 'start': task['start'], 'goal':task['goal']}
                env = DynamicEnvironment(dimensions, [guest], set(obstacles_guests) | idle_obstacles_guests,
                                moving_obstacles_guests, a_star_max_iter, self.get_graph_guests(),
                                weight_function=self.weight_function,
                                occupancy_model=self.occupancy_model)
                cbs = CBS(env)
                path_to_task_goal = algorithm.search(cbs)
                if not path_to_task_goal:
                    logging.info("REPLANNING 1 Solution not found to task goal for guest %s idling at current position...", guest['name'])
                    break
                    # exit(1)

                logging.info("REPLANNING 1 Solution found to task goal for guest %s doing task...", guest['name'])
                #print('path_to_task_goal',path_to_task_goal)
                cost2 = env.compute_solution_cost(path_to_task_goal)
                last_step = path_to_task_goal[guest['name']][-1]
                algorithm.update_ends_guests([current_guest_pos['x'], current_guest_pos['y']])
                algorithm.get_token()['path_ends_guest'].add(tuple([last_step['x'], last_step['y']]))
                algorithm.get_token()['guests_to_tasks'][guest['name']] = {'task_name': task['task_name'], 'start': task['start'],
                                                            'goal': task['goal'], 'predicted_cost': cost1 + cost2}
                algorithm.get_token()['guests'][guest['name']] = []
                algorithm.get_token()['guests'][guest['name']].append([current_guest_pos['x'], current_guest_pos['y']])
                for el in path_to_task_start[guest['name']]:
                    algorithm.get_token()['guests'][guest['name']].append([el['x'], el['y']])
                # Don't repeat twice same step
                algorithm.get_token()['guests'][guest['name']] = algorithm.get_token()['guests'][guest['name']][:-1]
                for el in path_to_task_goal[guest['name']]:
                    algorithm.get_token()['guests'][guest['name']].append([el['x'], el['y']])
            else:
                task = algorithm.get_token()['guests_to_tasks'][guest['name']]
                all_idle_guests = algorithm.get_token()['guests'].copy()
                all_idle_guests.pop(guest['name'])
                moving_obstacles_guests = algorithm.get_moving_obstacles(all_idle_guests, 1)
                idle_obstacles_guests = algorithm.get_idle_obstacles(all_idle_guests.values(), 1)
                #print('moving_obstacles_agents PER START1',moving_obstacles_agents)
                #print('TOKEN',self.token['agents'])
                guest = {'name': guest['name'], 'start': [x_new, y_new], 'goal': task['goal']}
                env = DynamicEnvironment(algorithm.dimensions, [guest], algorithm.obstacles_guests | idle_obstacles_guests,
                    moving_obstacles_guests, a_star_max_iter=algorithm.a_star_max_iter,
                    graph=self.get_graph_guests(),
                    weight_function=self.weight_function,
                    occupancy_model=self.occupancy_model)
                cbs = CBS(env)
                path_to_task_goal = algorithm.search(cbs)
                if not path_to_task_goal:
                    logging.info("REPLANNING 2 - Solution not found to task goal for guest %s idling at current position...", guest['name'])
                    break
                    # exit(1)

                cost2 = env.compute_solution_cost(path_to_task_goal)
                logging.info("REPLANNING 2 - Solution found to task goal for guest %s doing task...", guest['name'])
                last_step = path_to_task_goal[guest['name']][-1]
                algorithm.update_ends_guests([current_guest_pos['x'], current_guest_pos['y']])
                algorithm.token['path_ends'].add(tuple([last_step['x'], last_step['y']]))
                algorithm.token['guests_to_tasks'][guest['name']] = {'task_name': task['task_name'], 'start': task['start'],
                                                            'goal': task['goal'], 'predicted_cost': cost2}
                algorithm.token['guests'][guest['name']] = []
                algorithm.get_token()['guests'][guest['name']].append([current_guest_pos['x'], current_guest_pos['y']])
                for el in path_to_task_goal[guest['name']]:
                    algorithm.token['guests'][guest['name']].append([el['x'], el['y']])

        for guest in free_guests:
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

            presence_conflicts = False

            #we consider each agent: if its current position is in the surrounding of the guest, we check if its next move is the same as the guest
            #if so, we change the path of the guest
            for agent in self.agents:
                if now_next_agents[agent['name']][0] in surroundings[guest['name']]: #if the agent is at most two step ahead of the guest
                    if now_next_agents[agent['name']][1] == tuple([x_new, y_new]): #if the next move of the agent is the next move of the guest
                        presence_conflicts = True
                        logging.info('VERTEX-CONFLICT with an idle guest',tuple([x_new, y_new]), now_next_agents[agent['name']][1])
                        #remove move if it is the same as the agent next move
                        for move in possible_moves:
                            for agent1 in self.agents:
                                if move == now_next_agents[agent1['name']][1]:
                                    to_remove.append(move)
                        #remove move if there is an exchange between guest and another agent positions
                            for agent1 in self.agents:
                                if move == now_next_agents[agent1['name']][0] and now_next_agents[agent1['name']][1] == tuple([current_guest_pos['x'], current_guest_pos['y']]):
                                    to_remove.append(move)
                        #remove move if the move is the same as the one of another guest
                            for guest1 in self.guests:
                                if guest1['name'] != guest['name']:
                                    if move == now_next_guests[guest1['name']][1]:
                                        to_remove.append(move)
                        #remove move if there is an exchange between guest and another guest positions
                            for guest1 in self.guests:
                                if guest1['name'] != guest['name']:
                                    if move == now_next_guests[guest1['name']][0] and now_next_guests[guest1['name']][1] == tuple([current_guest_pos['x'], current_guest_pos['y']]):
                                        to_remove.append(move)

                            for guest1 in self.guests:
                                if guest1['name'] != guest['name']:
                                    if guest1['name'] in algorithm.get_token()['guests_to_tasks'].keys():
                                        if move == tuple(algorithm.get_token()['guests_to_tasks'][guest1['name']]['goal']):
                                            to_remove.append(move)

            # case in which the agent and the guest exchange their positions
            for agent in self.agents:
                if now_next_agents[agent['name']][0] in surroundings[guest['name']]:
                    if tuple([x_new, y_new])== now_next_agents[agent['name']][0] and now_next_agents[agent['name']][1] == tuple([current_guest_pos['x'], current_guest_pos['y']]):
                        presence_conflicts = True
                        logging.info('EDGE-CONFLICT with an idle guest')
                        #remove move if it is the same as the agent next move
                        for move in possible_moves:
                            for agent1 in self.agents:
                                if move == now_next_agents[agent1['name']][1]:
                                    to_remove.append(move)
                        #remove move if there is an exchange between guest and another agent positions
                            for agent1 in self.agents:
                                if move == now_next_agents[agent1['name']][0] and now_next_agents[agent1['name']][1] == tuple([current_guest_pos['x'], current_guest_pos['y']]):
                                    to_remove.append(move)
                        #remove move if the move is the same as the one of another guest
                            for guest1 in self.guests:
                                if guest1['name'] != guest['name']:
                                    if move == now_next_guests[guest1['name']][1]:
                                        to_remove.append(move)
                        #remove move if there is an exchange between guest and another guest positions
                            for guest1 in self.guests:
                                if guest1['name'] != guest['name']:
                                    if move == now_next_guests[guest1['name']][0] and now_next_guests[guest1['name']][1] == tuple([current_guest_pos['x'], current_guest_pos['y']]):
                                        to_remove.append(move)


            for move in list(set(to_remove)):
                possible_moves.remove(move)


            #if there are conflicts but there is no possible move for the guest, then there is a deadlock
            if presence_conflicts is True and possible_moves == []:
                logging.error('DEADLOCK ERROR! (%d, %d)', x_new, y_new)
                logging.error(algorithm.get_token()['guests_to_tasks'][guest['name']])
                exit(1)

            # if the presence of conflicts is true, we change the next move of the guest (choosinig a random one) and replan
            if presence_conflicts is True:
                move = random.choice(possible_moves)
                logging.info('chosen move %s: (%d, %d)', guest['name'], move[0], move[1])
                x_new, y_new = move
                self.n_replanning_guest += 1

                algorithm.get_token()['guests'][guest['name']].append([x_new, y_new])

                #for guest1 in self.guests:
                #    if guest1['name'] != guest['name']:
                #        if guest1['name'] in algorithm.get_token()['guests_to_tasks'].keys():
                #            if move == tuple(algorithm.get_token()['guests_to_tasks'][guest1['name']]['goal']):
                #                algorithm.get_token()['guests'][guest['name']].append(algorithm.get_token()['guests_to_tasks'][guest1['name']]['goal'])



                #self.actual_paths_guests[guest['name']].append({'t': self.time, 'x': x_new, 'y': y_new})

                #algorithm.update_ends_guests([x_new, y_new])
                #all_idle_guests = algorithm.get_token()['guests'].copy()
                #all_idle_guests.pop(guest['name'])
                #algorithm.go_to_closest_non_task_endpoint_guest2(guest['name'], [x_new, y_new], all_idle_guests)

        # TODO @bonaluca: clean this up and move it to the very end
        fields_of_view = {}
        for guest in self.guests:
            current_guest_pos = self.actual_paths_guests[guest['name']][-1]
            current_guest_pos = (current_guest_pos['x'], current_guest_pos['y'])
            adjacent_locations = self.G_guests.neighbors(current_guest_pos)
            fields_of_view[guest['name']] = set([*adjacent_locations, current_guest_pos])
            fields_of_view[guest['name']] |= set([self.G_guests.neighbors(n) for n in adjacent_locations])
        visible_locations = set.union(*fields_of_view.values())

        agent_locations = {agent['name']: (
            self.actual_paths[agent['name']][-1]['x'],
            self.actual_paths[agent['name']][-1]['y']
            ) for agent in self.agents}
        seen_agents = {agent['name']: agent_locations[agent['name']] \
            for agent in self.agents \
                if agent_locations[agent['name']] in visible_locations}
        free_locations = visible_locations - set(seen_agents.values())

        try:
            self.occupancy_model.time_forward(free_locations=free_locations, seen_agents=seen_agents)
            pass
        except NotFittedError as e:
            pass


        #NOW GUESTS
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

        #AGENTS
        # here we move the agents
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

