import argparse
import yaml
import random
import json
import os
import logging
from Simulation.TP_with_recovery import TokenPassingRecovery
from Simulation.simulation_new_recovery import SimulationNewRecovery
from Simulation.occupancy_model import OccupancyModel, OracleModel, MarkovianOccupancyModel
import RoothPath
import subprocess
import sys
from collections import defaultdict

if __name__ == '__main__':
    random.seed(1234)
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', help='Robustness parameter for k-TP', default=None, type=int)
    parser.add_argument('-p', help='Robustness parameter for p-TP', default=None, type=float)
    parser.add_argument('-pd', help='Expected probability of an agent of being at any time step (p-TP)',
                        default=0, type=float)
    parser.add_argument('-p_iter', help='Number of times a new path can be recalculated if the one calculated '
                                        'before exceeds the probability threshold (p-TP)',
                        default=1, type=int)
    parser.add_argument('-a_star_max_iter', help='Maximum number of states explored by the low-level algorithm',
                        default=5000, type=int)
    parser.add_argument('-slow_factor', help='Slow factor of visualization', default=1, type=int)
    parser.add_argument('-not_rand', help='Use if input has fixed tasks', action='store_true')
    parser.add_argument('-alpha', help='Parameter for balance between distance and occupancy', default=None, type=float)
    parser.add_argument('-map_name', help='Name of map chosen', default=None, type=str)
    parser.add_argument('-obs_time', help='Observation time of guests agents', default=300, type=int)
    parser.add_argument('-strict_idle', help='Guests can idle only at non-task endpoints', action='store_true')
    parser.add_argument('-order', help='Order of the model', default=0, type=int, choices=[0, 1, 2, 3])
    parser.add_argument('-smoothing', help='Apply smoothing to the transition matrix', action='store_true')
    parser.add_argument('-d', help='Output directory', default='output')
    parser.add_argument('-log', help='Log level of the application', default='WARN', type=str)

    args = parser.parse_args()

    if args.k is None:
        args.k = 0
    if args.p is None:
        args.p = 1

    # Configure logging
    log_num_level = getattr(logging, args.log.upper(), logging.WARN)
    logging.basicConfig(level=log_num_level, format='[%(levelname)s]: %(message)s')

    with open(os.path.join(RoothPath.get_root(), 'config.json'), 'r') as json_file:
        config = json.load(json_file)
    args.param = os.path.join(RoothPath.get_root(), os.path.join(config['input_path'], str(args.map_name)))

    args.output = os.path.join(RoothPath.get_root(), args.d)
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    args.output = os.path.join(args.output, str(args.alpha)+'-'+str(args.map_name))


    # Read from input file
    with open(args.param, 'r') as param_file:
        try:
            param = yaml.load(param_file, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)

    dimensions = param['map']['dimensions']
    obstacles = list(map(lambda x: tuple(x), param['map']['obstacles']))
    non_task_endpoints = list(map(lambda x: tuple(x), param['map']['non_task_endpoints']))
    non_task_endpoints_guests = list(map(lambda x: tuple(x), param['map']['non_task_endpoints_guests']))
    delivery_agents = list(map(lambda x: tuple(x), param['map']['delivery_agents']))
    delivery_guests = list(map(lambda x: tuple(x), param['map']['delivery_guests']))
    obstacles_agents = obstacles+non_task_endpoints_guests+delivery_guests
    obstacles_guests = obstacles+non_task_endpoints+delivery_agents
    agents = param['agents']
    guests = param['guests']
    tasks = param['tasks']
    tasks_guest = param['tasks_guest']
    param['tasks'] = tasks
    param['tasks_guest'] = tasks_guest
    with open(args.param + config['visual_postfix'], 'w') as param_file:
        yaml.safe_dump(param, param_file)

    # Occupancy model
    #occupancy_model = OracleModel(*dimensions, agents, lambda: globals()['tp'].get_token())
    if args.order == 0:
        occupancy_model = OccupancyModel(*dimensions, agents)
    else:
        occupancy_model = MarkovianOccupancyModel(
            *dimensions, agents, obstacles=obstacles_agents,
            order=args.order, cache_size="1G", backoff=args.smoothing
        )

    # Simulate
    simulation = SimulationNewRecovery(tasks, tasks_guest, agents, guests, occupancy_model, alpha=args.alpha)
    a_star_max_iter = 4000
    observation_time=args.obs_time
    tp = TokenPassingRecovery(agents, guests, dimensions, obstacles_agents, obstacles_guests, non_task_endpoints, non_task_endpoints_guests, simulation,
                              a_star_max_iter=args.a_star_max_iter, k=args.k, pd=args.pd, p_max=args.p, p_iter=args.p_iter,
                              new_recovery=True, strict_idle=args.strict_idle)
    
    
    while (tp.get_completed_tasks() != len(tasks) or tp.get_completed_tasks_guest() != len(tasks_guest)):
        # TODO @bonaluca: never true if observation_time = 0
        if simulation.time == observation_time - 1:
            occupancy_model.fit(simulation.actual_paths)
        simulation.time_forward(tp, dimensions, non_task_endpoints_guests, obstacles, obstacles_agents, obstacles_guests, a_star_max_iter)

        if simulation.deadlock == True:
            #logging.error('Deadlock')
            break

        if simulation.time == 3000:
            logging.warning('Simulation timeout elapsed')
            break

    if simulation.deadlock == False:

        # Computation of the whole cost
        cost = 0
        for path in simulation.actual_paths.values():
            cost = cost + len(path)
        cost_guests = 0
        for path_guest in simulation.actual_paths_guests.values():
            cost_guests = cost_guests + len(path_guest)

        # Computation of the number of conflicts occurred
        schedule = {**simulation.actual_paths, **simulation.actual_paths_guests}
        # schedule è una tupla che contiene un dizionario; ogni dizionario contiene una lista di dizionari

        schedule_guests = simulation.actual_paths_guests,
        schedule_agents = simulation.actual_paths,

        conflicts = []
        for agent in schedule_agents[0]:
            for guest in schedule_guests[0]:
                conflicts = conflicts + [x for x in schedule_agents[0][agent] if x in schedule_guests[0][guest]]
        n_conflicts = len(conflicts)


        timespan = 0
        for task in tasks_guest:
            timespan += tp.get_completed_tasks_times_guest()[task['task_name']] - task['start_time']
        timespan = timespan/len(tasks_guest)

        execution_time_tasks = 0
        for task in tasks_guest:
            execution_time_tasks += tp.get_completed_tasks_times_guest()[task['task_name']] - tp.get_assigned_tasks_times_guest()[task['task_name']]
        execution_time_tasks = execution_time_tasks/len(tasks_guest)

        team_cost = max(tp.get_completed_tasks_times().values()) - min([time for time in simulation.start_times if time > observation_time])
        team_cost_guest = max(tp.get_completed_tasks_times_guest().values()) - min(simulation.start_times_guests)

        #print(schedule_guests)

        counter_moves_guests = {}
        for guest in schedule_guests[0]:
            counter_moves_guests[guest] = 0
            for i in range(len(schedule_guests[0][guest])-1):
                if schedule_guests[0][guest][i]['x'] != schedule_guests[0][guest][i+1]['x'] or schedule_guests[0][guest][i]['y'] != schedule_guests[0][guest][i+1]['y']:
                    counter_moves_guests[guest] += 1

        counter_moves = {}
        for agent in schedule_agents[0]:
            counter_moves[agent] = 0
            for i in range(len(schedule_agents[0][agent])-1):
                if schedule_agents[0][agent][i]['x'] != schedule_agents[0][agent][i+1]['x'] or schedule_agents[0][agent][i]['y'] != schedule_agents[0][agent][i+1]['y']:
                    counter_moves[agent] += 1


        output = {'schedule': schedule,
                'cost_residentials': cost,
                'cost_guests':cost_guests,
                'n_team_cost': team_cost,
                'n_team_cost_guest':team_cost_guest,
                'completed_tasks_times': tp.get_completed_tasks_times(),
                'assigned_tasks_times_guest': tp.get_assigned_tasks_times_guest(),
                'completed_tasks_times_guest': tp.get_completed_tasks_times_guest(),
                'conflicts': conflicts,
                'prob_mat': occupancy_model.get_prob_matrix_90deg(),
                'n_conflicts': n_conflicts,
                'n_timespan': timespan,
                'execution_time_tasks': execution_time_tasks,
                'n_replanning_guest': simulation.get_n_replanning_guest(),
                'count_moves_guests': counter_moves_guests,
                'count_moves_agents': counter_moves,
                'n_replans': tp.get_n_replans()}
        with open(args.output, 'w') as output_yaml:
            yaml.safe_dump(output, output_yaml)

    
    if simulation.deadlock == True:
        args.output = os.path.join(RoothPath.get_root(), 'output', str(args.alpha)+'-DL-'+str(args.map_name))

        schedule = {**simulation.actual_paths, **simulation.actual_paths_guests}

        schedule_guests = simulation.actual_paths_guests,
        schedule_agents = simulation.actual_paths,

        conflicts = []
        for agent in schedule_agents[0]:
            for guest in schedule_guests[0]:
                conflicts = conflicts + [x for x in schedule_agents[0][agent] if x in schedule_guests[0][guest]]
        n_conflicts = len(conflicts)

        output = {'schedule': schedule,
        #'cost_residentials': cost,
        #'cost_guests':cost_guests,
        #'n_team_cost': team_cost,
        #'n_team_cost_guest':team_cost_guest,
        'completed_tasks_times': tp.get_completed_tasks_times(),
        'assigned_tasks_times_guest': tp.get_assigned_tasks_times_guest(),
        'completed_tasks_times_guest': tp.get_completed_tasks_times_guest(),
        'conflicts': conflicts,
        'guest_involved': simulation.guest_presence,
        'deadlock_task': simulation.deadlock_task,
        'deadlock_guest': simulation.deadlock_guest,
        #'prob_mat': occupancy_model.get_prob_matrix_90deg(),
        'n_conflicts': n_conflicts,
        #'n_timespan': timespan,
        #'execution_time_tasks': execution_time_tasks,
        'n_replanning_guest': simulation.get_n_replanning_guest(),
        #'count_moves_guests': counter_moves_guests,
        #'count_moves_agents': counter_moves,
        #'n_replans': tp.get_n_replans()
        }
        with open(args.output, 'w') as output_yaml:
            yaml.safe_dump(output, output_yaml)

        exit(1)

#    create = [sys.executable, '-m', 'Utils.Visualization.visualize', '-slow_factor', str(args.slow_factor), '-alpha',str(args.alpha),'-map_name',str(args.map_name), '-deadlock', str(simulation.deadlock)]
#    subprocess.call(create)

