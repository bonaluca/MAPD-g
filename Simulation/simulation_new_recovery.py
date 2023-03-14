import logging
import time
import random
from math import fabs
import networkx as nx
from Simulation.CBS.cbs import CBS, DynamicEnvironment
from Simulation.exceptions import NotFittedError, PathNotFoundError

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
        self.deadlock = False
        for t in self.tasks: #we add in start_times the starting time of each taks of the agents
            self.start_times.append(t['start_time'])
        for t in self.tasks_guest: #we add in start_times the starting time of each taks of the guests
            self.start_times_guests.append(t['start_time'])
        for agent in self.agents: #we update the actual paths of the agents by adding their current initial position
            self.actual_paths[agent['name']] = [{'t': 0, 'x': agent['start'][0], 'y': agent['start'][1]}]
        for guest in self.guests: #we update the actual paths of the guests by adding their current initial position
            self.actual_paths_guests[guest['name']] = [{'t': 0, 'x': guest['start'][0], 'y': guest['start'][1]}]



    # time_forward is the method in which the agents are effectly moved and the time is shifted one step ahead
    def time_forward(self, algorithm, dimensions, non_task_endpoints_guests, obstacles, obstacles_agents, obstacles_guests, a_star_max_iter):

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
        guests = self.guests
        random.shuffle(agents)
        random.shuffle(guests)
        agents = list(map(lambda agent: agent['name'], agents))
        guests = list(map(lambda guest: guest['name'], guests))

        idle_guests = [guest for guest in guests \
                if len(algorithm.get_token()['guests'][guest]) == 1]
        moving_guests = [guest for guest in guests \
                if len(algorithm.get_token()['guests'][guest]) > 1]

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

        # Helpers
        location_within_grid = lambda loc: \
            0 <= loc[0] < dimensions[0] and 0 <= loc[1] < dimensions[1]
        neighbors = lambda loc: [
            #(loc[0]-1, loc[1]), (loc[0], loc[1]-1), (loc[0], loc[1]), (loc[0], loc[1]+1), (loc[0]+1, loc[1])
            (loc[0], loc[1]), (loc[0]+1, loc[1]), (loc[0]-1, loc[1]), (loc[0], loc[1]+1), (loc[0], loc[1]-1)
        ]

        #NOW GUESTS
        for guest in moving_guests:
            self.guest_presence = False
            current_guest_pos = self.actual_paths_guests[guest][-1] #current_guest_pos tells us the current position of the guest
            #self.guests_pos_now.add(tuple([current_guest_pos['x'], current_guest_pos['y']])) #guest current position update

            #if the guest does not have any task assigned and is not in an endpoint, we send it to an endpoint
            #if guest['name'] not in algorithm.get_token()['guests_to_tasks'].keys() and tuple([current_guest_pos['x'], current_guest_pos['y']]) not in non_task_endpoints_guests:
            #    all_idle_guests = algorithm.get_token()['guests'].copy()
            #    all_idle_guests.pop(guest['name'])
            #    algorithm.go_to_closest_non_task_endpoint_guest3(guest, [current_guest_pos['x'], current_guest_pos['y']], all_idle_guests)

            x_new, y_new = algorithm.get_token()['guests'][guest][1]

            #we consider each agent: if its current position is in the surrounding of the guest, we check if its next move is the same as the guest
            #if so, we change the path of the guest
            for agent in algorithm.agents_nearby(guest):
                #if the next move of the agent is the next move of the guest
                if algorithm.next_pos_agents(agent) == (x_new, y_new):
                    logging.info('VERTEX-CONFLICT with a moving guest (%d, %d) %s', x_new, y_new, algorithm.next_pos_agents(agent))

                # case in which the agent and the guest exchange their positions
                if algorithm.current_pos_agents(agent) == (x_new, y_new) and \
                    algorithm.next_pos_agents(agent) == algorithm.current_pos_guests(guest):
                    logging.info('EDGE-CONFLICT with a moving guest')

            # possible_moves denotes all available moves, excluding those
            # that are conflicting with agents' plans
            possible_moves = list(filter(
                lambda move: \
                    location_within_grid(move) and
                    move not in obstacles_guests and
                    move not in [algorithm.next_pos_agents(agent) for agent in agents] and
                    move not in [
                        algorithm.current_pos_agents(agent) for agent in agents
                        if algorithm.next_pos_agents(agent) == algorithm.current_pos_guests(guest)
                    ],
                neighbors(algorithm.current_pos_guests(guest))
            ))

            if not possible_moves:
                self.deadlock = True
                self.deadlock_task = algorithm.get_token()['guests_to_tasks'][guest]
                self.deadlock_guest = guest
                logging.error('DEADLOCK!2 (%d, %d)', x_new, y_new)
                logging.error(self.deadlock_task)
                return

            # Filter out conflicting guest moves
            accepted_moves = list(filter(lambda move:
                move not in [ # Vertex conflicting moves
                    algorithm.next_pos_guests(other_guest)
                    for other_guest in guests if other_guest != guest
                ] and
                move not in [ # Edge conflicting moves
                    algorithm.current_pos_guests(other_guest)
                    for other_guest in guests
                    if other_guest != guest and
                    algorithm.next_pos_guests(other_guest) == algorithm.current_pos_guests(guest)
                ], possible_moves
            ))

            if (x_new, y_new) in accepted_moves:
                continue

            if accepted_moves:
                # TODO @bonaluca: why? can't we just let the low level planner do the whole job?
                if algorithm.get_token()['guests_to_tasks'][guest]['start'] in algorithm.get_token()['guests'][guest]:
                    move = self.get_closest_move(accepted_moves, algorithm.get_token()['guests_to_tasks'][guest]['start'])
                else:
                    move = self.get_closest_move(accepted_moves, algorithm.get_token()['guests_to_tasks'][guest]['goal'])
                #move = random.choice(possible_moves)
                self.n_replanning_guest += 1
                x_new, y_new = move

#                try:
#                    guests_to_replan = [guest]
#                    first_moves = {guest: (x_new, y_new)}
#                    self.replan2(algorithm, guests_to_replan, first_moves)
#                except PathNotFoundError:
#                    break

                try:
                    self.replan(algorithm, guest, first_move=(x_new, y_new))
                    continue
                except PathNotFoundError:
                    # Path not found and no other viable way
                    if not set(possible_moves) - set(accepted_moves):
                        self.deadlock = True
                        self.deadlock_guest = guest
                        self.deadlock_task = algorithm.get_token()['guests_to_tasks'][guest]
                        return

            # Possible deadlock that may be recoverable, since it involves only guests
            self.deadlock_guest = guest
            self.deadlock_task = algorithm.get_token()['guests_to_tasks'][guest]
            logging.info('Possible deadlock for %s at (%d, %d) involving other guests' % (guest, x_new, y_new))
            logging.info(self.deadlock_task)

###############################################################################################################################################################

            accepted_moves = possible_moves.copy()

            guests_change_move = []
            #prima di tutto il guest in deadlock cambia mossa
            if algorithm.get_token()['guests_to_tasks'][guest]['start'] in algorithm.get_token()['guests'][guest]:
                # TODO @bonaluca: BUGFIX accepted_moves may be empty (env1.3.yaml)
                move = self.get_closest_move(accepted_moves, algorithm.get_token()['guests_to_tasks'][guest]['start'])
            else:
                move = self.get_closest_move(accepted_moves, algorithm.get_token()['guests_to_tasks'][guest]['goal'])

            for guest_check in guests:
                # TODO @bonaluca: this only inserts vertex conflicting guests, edge conflicts to be added
                if algorithm.next_pos_guests(guest_check) == move:
                    guests_change_move.append(guest_check)

            #ind_new_move = guests_interfere_moves.index(move)
            #guests_change_move.append(guests_interfere[ind_new_move])
            self.n_replanning_guest += 1
            x_new = move[0]
            y_new = move[1]
            algorithm.get_token()['guests'][guest] = []
            algorithm.get_token()['guests'][guest].append([current_guest_pos['x'], current_guest_pos['y']])
            algorithm.get_token()['guests'][guest].append([x_new, y_new])


################################################################################################################################################################
            guests_replan = []
            guests_start = []
            guests_replan.append(guest)
            while len(guests_change_move) > 0:
                logging.info('guests_change_move %s' % str(guests_change_move))
                g = guests_change_move[0]

                current_guest_pos = self.actual_paths_guests[g][-1] #current_guest_pos tells us the current position of the guest
                self.guests_pos_now.add(tuple([current_guest_pos['x'], current_guest_pos['y']])) #guest current position update

                # possible_moves denotes all available moves, excluding those
                # that are conflicting with agents' plans
                possible_moves = list(filter(
                    lambda move: \
                        location_within_grid(move) and
                        move not in obstacles_guests and
                        move not in [algorithm.next_pos_agents(agent) for agent in agents] and
                        move not in [
                            algorithm.current_pos_agents(agent) for agent in agents
                            if algorithm.next_pos_agents(agent) == algorithm.current_pos_guests(g)
                        ],
                    neighbors(algorithm.current_pos_guests(g))
                ))

                accepted_moves = possible_moves.copy()

                # case in which the guest has a task assigned
                if len(algorithm.get_token()['guests'][g]) > 1:
                    x_new, y_new = algorithm.get_token()['guests'][g][1]
                    logging.info('x_new, y_new (%d, %d) %s' % (x_new, y_new, g))

                    interfering_guests = dict()

                    for move in accepted_moves:
                        for other_guest in guests:
                            if other_guest == g:
                                continue

                            #remove move if the move is the same as the one of another guest
                            if move == algorithm.next_pos_guests(other_guest):
                                interfering = interfering_guests.setdefault(move, [])
                                interfering.append(other_guest)

                            #remove move if there is an exchange between guest and another guest positions
                            if move == algorithm.current_pos_guests(other_guest) and \
                                algorithm.next_pos_guests(other_guest) == algorithm.current_pos_guests(g):
                                interfering = interfering_guests.setdefault(move, [])
                                interfering.append(other_guest)

                    accepted_moves = list(filter(lambda move:
                        move not in interfering_guests.keys(),
                        possible_moves
                    ))

                    if (x_new, y_new) not in accepted_moves:
                        if accepted_moves:

                            #here the move is changed according to the possible ones
                            if algorithm.get_token()['guests_to_tasks'][g]['start'] in algorithm.get_token()['guests'][g]:
                                move = self.get_closest_move(accepted_moves, algorithm.get_token()['guests_to_tasks'][g]['start'])
                            else:
                                move = self.get_closest_move(accepted_moves, algorithm.get_token()['guests_to_tasks'][g]['goal'])

                            self.n_replanning_guest += 1
                            x_new = move[0]
                            y_new = move[1]
                            guests_replan.append(guests_change_move[0])
                            guests_change_move.remove(guests_change_move[0])

                            if algorithm.get_token()['guests_to_tasks'][g]['start'] in algorithm.get_token()['guests'][g]:
                                guests_start.append(g)


                            algorithm.get_token()['guests'][g] = []
                            algorithm.get_token()['guests'][g].append([current_guest_pos['x'], current_guest_pos['y']])
                            algorithm.get_token()['guests'][g].append([x_new, y_new])

                        elif possible_moves:
                            moves = list(interfering_guests.keys())
                            if algorithm.get_token()['guests_to_tasks'][g]['start'] in algorithm.get_token()['guests'][g]:
                                move = self.get_closest_move(moves, algorithm.get_token()['guests_to_tasks'][g]['start'])
                            else:
                                move = self.get_closest_move(moves, algorithm.get_token()['guests_to_tasks'][g]['goal'])

#                            ind_new_move = guests_interfere_moves.index(move)
#                            guests_change_move.append(guests_interfere[ind_new_move])
                            guests_change_move += interfering_guests[move]
                            self.n_replanning_guest += 1
                            x_new = move[0]
                            y_new = move[1]
                            guests_replan.append(guests_change_move[0])
                            guests_change_move.remove(guests_change_move[0])

                            if algorithm.get_token()['guests_to_tasks'][g]['start'] in algorithm.get_token()['guests'][g]:
                                guests_start.append(g)

                            algorithm.get_token()['guests'][g] = []
                            algorithm.get_token()['guests'][g].append([current_guest_pos['x'], current_guest_pos['y']])
                            algorithm.get_token()['guests'][g].append([x_new, y_new])

                        else:
                            logging.error('Deadlock due to the move of %s' % g)
                            self.deadlock = True
                            self.deadlock_guest = g
                            self.deadlock_task = algorithm.get_token()['guests_to_tasks'][g]
                            break

                else: # guest to change move is idle
                    x_new, y_new = algorithm.get_token()['guests'][g][0]

                    logging.info('x_new, y_new (%d, %d) %s' % (x_new, y_new, g))

                    #remove move if the move is the same as the one of another guest
                    interfering_guests = dict()

                    for move in accepted_moves:
                        for other_guest in guests:
                            if other_guest == g:
                                continue

                            if move == algorithm.next_pos_guests(other_guest):
                                interfering = interfering_guests.setdefault(move, [])
                                interfering.append(other_guest)

                    accepted_moves = list(filter(lambda move:
                        move not in interfering_guests.keys(),
                        possible_moves
                    ))

                    if (x_new, y_new) not in accepted_moves:
                        if accepted_moves:
                            #here the move is changed according to the possible ones
                            move = random.choice(accepted_moves)

                            self.n_replanning_guest += 1
                            x_new, y_new = move
                            guests_replan.append(guests_change_move[0])
                            guests_change_move.remove(guests_change_move[0])

                            algorithm.get_token()['guests'][g] = []
                            algorithm.get_token()['guests'][g].append([current_guest_pos['x'], current_guest_pos['y']])
                            algorithm.get_token()['guests'][g].append([x_new, y_new])

                        elif possible_moves:
                            move = random.choice(list(interfering_guests.keys()))

#                            ind_new_move = guests_interfere_moves.index(move)
#                            guests_change_move.append(guests_interfere[ind_new_move])
                            guests_change_move += interfering_guests[move]
                            self.n_replanning_guest += 1
                            x_new, y_new = move
                            guests_replan.append(guests_change_move[0])
                            guests_change_move.remove(guests_change_move[0])

                            algorithm.get_token()['guests'][g] = []
                            algorithm.get_token()['guests'][g].append([current_guest_pos['x'], current_guest_pos['y']])
                            algorithm.get_token()['guests'][g].append([x_new, y_new])

                        else:
                            logging.error('Deadlock due to the move of %s' % g)
                            self.deadlock = True
                            self.deadlock_guest = g
                            self.deadlock_task = algorithm.get_token()['guests_to_tasks'][g]
                            break

######################################################################################################################################################################
            #ora tutti i guest che hanno dovuto cambiare mossa devono fare replanning
            for g_rep in sorted(set(guests_replan)):
                # Skip free guests
                if not g_rep in algorithm.get_token()['guests_to_tasks']:
                    continue

                # TODO @bonaluca: ugly workaround to get the proper replanning towards task start and goal
                if g_rep in guests_start:
                    task_start = algorithm.get_token()['guests_to_tasks'][g_rep]['start'].copy()
                    algorithm.get_token()['guests'][g_rep].append(task_start)

                x_new, y_new = algorithm.get_token()['guests'][g_rep][1]
                self.replan(algorithm, g_rep, (x_new, y_new))
                #functions.planning_function(g_rep, algorithm, x_new, y_new, obstacles_guests, dimensions, a_star_max_iter, self, guests_start)

            if self.deadlock:
                logging.error('Deadlock with one guest only')
                return

        for guest in idle_guests:
            self.guest_presence = False
            current_guest_pos = self.actual_paths_guests[guest][-1]

            x_new, y_new = algorithm.get_token()['guests'][guest][0]

            #we consider each agent: if its current position is in the surrounding of the guest, we check if its next move is the same as the guest
            #if so, we change the path of the guest
            for agent in algorithm.agents_nearby(guest):
                #if the next move of the agent is the next move of the guest
                if algorithm.next_pos_agents(agent) == (x_new, y_new):
                    logging.info('VERTEX-CONFLICT with an idle guest (%d, %d) %s' % (x_new, y_new, str(algorithm.next_pos_agents(agent))))
                # case in which the agent and the guest exchange their positions
                # TODO @bonaluca: this never happens
                if algorithm.current_pos_agents(agent) == (x_new, y_new) and \
                    algorithm.next_pos_agents(agent) == algorithm.current_pos_guests(guest):
                    logging.info('EDGE-CONFLICT with an idle guest')

            # possible_moves denotes all available moves, excluding those
            # that are conflicting with agents' plans
            possible_moves = list(filter(
                lambda move: \
                    location_within_grid(move) and
                    move not in obstacles_guests and
                    move not in [algorithm.next_pos_agents(agent) for agent in agents] and
                    move not in [
                        algorithm.current_pos_agents(agent) for agent in agents
                        if algorithm.next_pos_agents(agent) == algorithm.current_pos_guests(guest)
                    ],
                neighbors(algorithm.current_pos_guests(guest))
            ))

            # Filter out vertex conflicting guest moves
            accepted_moves = list(filter(
                lambda move: move not in [
                    algorithm.next_pos_guests(other_guest)
                    for other_guest in guests if other_guest != guest
                ], possible_moves
            ))

            to_remove = []
            for move in accepted_moves:
                for other_guest in guests:
                    if other_guest != guest:
                        if other_guest in algorithm.get_token()['guests_to_tasks'].keys():
                            if move == tuple(algorithm.get_token()['guests_to_tasks'][other_guest]['goal']):
                                to_remove.append(move)

            for move in list(set(to_remove)):
                accepted_moves.remove(move)

            if (x_new, y_new) in accepted_moves:
                continue

            if not accepted_moves:
                self.deadlock = True
                self.deadlock_guest = guest
                self.deadlock_task = None
                logging.error('DEADLOCK ERROR! (%d, %d)', x_new, y_new)
                return

            # if the presence of conflicts is true, we change the next move of the guest (choosing a random one) and replan
            move = random.choice(accepted_moves)
            logging.info('chosen move %s: (%d, %d)', guest, move[0], move[1])
            x_new, y_new = move
            self.n_replanning_guest += 1

            algorithm.get_token()['guests'][guest].append([x_new, y_new])

        # Move the guests
        for guest in guests:
            current_guest_pos = self.actual_paths_guests[guest][-1]
            guest_plan = algorithm.get_token()['guests'][guest]
            if len(guest_plan) <= 1:
                # Guest is idle
                (x_new, y_new) = (current_guest_pos['x'], current_guest_pos['y'])
            else:
                # Guest has an assigned task
                (x_new, y_new) = tuple(guest_plan[1])
                algorithm.get_token()['guests'][guest] = guest_plan[1:]
            self.actual_paths_guests[guest].append({'t': self.time, 'x': x_new, 'y': y_new})
            self.guests_moved.add(guest)
            self.guests_pos_now.add((x_new, y_new))

        # Move the agents
        for agent in agents:
            current_agent_pos = self.actual_paths[agent][-1]
            agent_plan = algorithm.get_token()['agents'][agent]
            if len(agent_plan) <= 1:
                # Agent is idle
                (x_new, y_new) = (current_agent_pos['x'], current_agent_pos['y'])
            else:
                # Agent has an assigned task
                (x_new, y_new) = tuple(agent_plan[1])
                algorithm.get_token()['agents'][agent] = agent_plan[1:]
            self.actual_paths[agent].append({'t': self.time, 'x': x_new, 'y': y_new})
            self.agents_moved.add(agent)
            self.agents_pos_now.add((x_new, y_new))

        # Occupancy model time forward
        visible_locations = set.union(
            *(algorithm.surroundings(guest) for guest in guests)
        )

        seen_agents = {
            agent_name: algorithm.current_pos_agents(agent_name)
            for agent_name in set.union(
                *(algorithm.agents_nearby(guest) for guest in guests)
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
            guest: algorithm.get_token()['guests_to_tasks'][guest]
            for guest in guests
        }
        new_tokens = {
            guest: [list(algorithm.current_pos_guests(guest))]
            for guest in guests
        }
        reached_goal = lambda guest_name: \
            tasks[guest_name]['goal'] == new_tokens[guest_name][-1]
        total_cost = {}

        guests_list = [
            {
                'name': guest,
                'start': first_moves[guest],
                'goal': tasks[guest]['start'] \
                    if tasks[guest]['start'] in algorithm.get_token()['guests'][guest][1:] \
                    else tasks[guest]['goal']
            }
            for guest in guests
        ]

        other_guests_token = algorithm.get_token()['guests'].copy()
        for guest in guests:
            other_guests_token.pop(guest)

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
            new_plan = {guest: new_plans[guest]}
            total_cost[guest] = env.compute_solution_cost(new_plan)
            # Append path to token
            for el in new_plan[guest]:
                new_tokens[guest].append([el['x'], el['y']])

        guests_list = [
            {
                'name': guest,
                'start': tasks[guest]['start'],
                'goal': tasks[guest]['goal']
            }
            for guest in guests if not reached_goal(guest)
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
            last_step = new_tokens[guest][-1]
            algorithm.update_ends_guests(algorithm.current_pos_guests(guest))
            algorithm.get_token()['path_ends_guest'].add(tuple(last_step))
            algorithm.get_token()['guests_to_tasks'][guest] = {
                'task_name': tasks[guest]['task_name'],
                'start': tasks[guest]['start'],
                'goal': tasks[guest]['goal'],
                'predicted_cost': total_cost[guest]
            }
            # Update token
            algorithm.get_token()['guests'][guest] = new_tokens[guest]
            pass

    def replan(self, algorithm, guest, first_move):
        """Replan for a guest agent."""

        time_start = 1
        task = algorithm.get_token()['guests_to_tasks'][guest]
        #new_token = [list(first_move), ]
        new_token = [list(algorithm.current_pos_guests(guest)), ]
        start_location = first_move
        total_cost = 0
        all_idle_guests = algorithm.get_token()['guests'].copy()
        all_idle_guests.pop(guest)

        if task['start'] in algorithm.get_token()['guests'][guest][1:]:

            #we first replan a path from the new location to the pick-up point
            moving_obstacles_guests = algorithm.get_moving_obstacles(all_idle_guests, time_start)
            idle_obstacles_guests = algorithm.get_idle_obstacles(all_idle_guests.values(), time_start)
            guests = [{'name': guest, 'start': start_location, 'goal': task['start']}]
            env = DynamicEnvironment(
                algorithm.dimensions, guests, set(algorithm.obstacles_guests) | idle_obstacles_guests,
                moving_obstacles_guests, algorithm.a_star_max_iter,
                graph=self.get_graph_guests(),
                time_start=time_start,
                weight_function=self.weight_function,
                occupancy_model=self.occupancy_model
            )
            cbs = CBS(env)
            path_to_task_start = algorithm.search(cbs)
            if not path_to_task_start:
                logging.info("REPLANNING 1 Solution not found to task start for guest %s idling at current position...", guest)
                raise PathNotFoundError()

            logging.info("REPLANNING 1 Solution found to task start for guest %s searching solution to task goal...", guest)
            cost1 = env.compute_solution_cost(path_to_task_start)
            for el in path_to_task_start[guest][:-1]:
                new_token.append([el['x'], el['y']])
            total_cost += cost1
            time_start = cost1
            start_location = task['start']

        #once the path to the pick-up point has been found, we replan a path to the delivery point
        moving_obstacles_guests = algorithm.get_moving_obstacles(all_idle_guests, time_start)
        idle_obstacles_guests = algorithm.get_idle_obstacles(all_idle_guests.values(), time_start)
        guests = [{'name': guest, 'start': start_location, 'goal': task['goal']}]
        env = DynamicEnvironment(
            algorithm.dimensions, guests, algorithm.obstacles_guests | idle_obstacles_guests,
            moving_obstacles_guests, a_star_max_iter=algorithm.a_star_max_iter,
            graph=self.get_graph_guests(),
            time_start=time_start,
            weight_function=self.weight_function,
            occupancy_model=self.occupancy_model
        )
        cbs = CBS(env)
        path_to_task_goal = algorithm.search(cbs)
        if not path_to_task_goal:
            logging.info("REPLANNING 2 - Solution not found to task goal for guest %s idling at current position...", guest)
            raise PathNotFoundError()

        cost2 = env.compute_solution_cost(path_to_task_goal)
        total_cost += cost2
        logging.info("REPLANNING - Solution found to task goal for guest %s doing task...", guest)
        last_step = path_to_task_goal[guest][-1]
        algorithm.update_ends_guests(algorithm.current_pos_guests(guest))
        algorithm.get_token()['path_ends_guest'].add(tuple([last_step['x'], last_step['y']]))
        algorithm.get_token()['guests_to_tasks'][guest] = {'task_name': task['task_name'], 'start': task['start'],
                                                    'goal': task['goal'], 'predicted_cost': total_cost}
        for el in path_to_task_goal[guest]:
            new_token.append([el['x'], el['y']])
        algorithm.get_token()['guests'][guest] = new_token
        pass

    def replan_new(self, guest, algorithm):
        """."""

        time_start = 0
        task = algorithm.get_token()['guests_to_tasks'][guest]
        start_location = list(algorithm.current_pos_guests(guest))
        new_token = []
        total_cost = 0
        all_idle_guests = algorithm.get_token()['guests'].copy()
        all_idle_guests.pop(guest)

        nearby_agent_tokens = {
            agent: [algorithm.current_pos_agents(agent), algorithm.next_pos_agents(agent)]
            for agent in algorithm.agents_nearby(guest)
        }

        if task['start'] in algorithm.get_token()['guests'][guest][1:]:

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

            guests = [{'name': guest, 'start': start_location, 'goal': task['start']}]
            env = DynamicEnvironment(
                algorithm.dimensions, guests, set(algorithm.obstacles_guests) | idle_obstacles_guests,
                moving_obstacles, algorithm.a_star_max_iter,
                graph=self.get_graph_guests(),
                time_start=time_start,
                weight_function=self.weight_function,
                occupancy_model=self.occupancy_model
            )
            cbs = CBS(env)
            path_to_task_start = algorithm.search(cbs)
            if not path_to_task_start:
                logging.info("REPLANNING 1 Solution not found to task start for guest %s idling at current position...", guest)
                raise PathNotFoundError()

            logging.info("REPLANNING 1 Solution found to task start for guest %s searching solution to task goal...", guest)
            cost1 = env.compute_solution_cost(path_to_task_start)
            for el in path_to_task_start[guest][:-1]:
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

        guests = [{'name': guest, 'start': start_location, 'goal': task['goal']}]
        env = DynamicEnvironment(
            algorithm.dimensions, guests, algorithm.obstacles_guests | idle_obstacles_guests,
            moving_obstacles, a_star_max_iter=algorithm.a_star_max_iter, time_start=time_start,
            graph=self.get_graph_guests(),
            weight_function=self.weight_function,
            occupancy_model=self.occupancy_model)
        cbs = CBS(env)
        path_to_task_goal = algorithm.search(cbs)
        if not path_to_task_goal:
            logging.info("REPLANNING 2 - Solution not found to task goal for guest %s idling at current position...", guest)
            raise PathNotFoundError()

        cost2 = env.compute_solution_cost(path_to_task_goal)
        total_cost += cost2
        logging.info("REPLANNING - Solution found to task goal for guest %s doing task...", guest)
        last_step = path_to_task_goal[guest][-1]
        algorithm.update_ends_guests(algorithm.current_pos_guests(guest))
        algorithm.get_token()['path_ends_guest'].add(tuple([last_step['x'], last_step['y']]))
        algorithm.get_token()['guests_to_tasks'][guest] = {'task_name': task['task_name'], 'start': task['start'],
                                                    'goal': task['goal'], 'predicted_cost': total_cost}
        for el in path_to_task_goal[guest]:
            new_token.append([el['x'], el['y']])
        algorithm.get_token()['guests'][guest] = new_token

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

