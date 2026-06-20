import argparse
import os
import numpy as np
import yaml
import re
from itertools import product

def window_type(arg_value, patt=re.compile('^([0-9]+)[:,]([0-9]+)$')):
    if not patt.match(arg_value):
        raise argparse.ArgumentTypeError('invalid value')
    split = patt.split(arg_value)
    window_start, window_end = int(split[1]), int(split[2])
    if window_start > window_end:
        raise argparse.ArgumentTypeError('start must be lower than end')
    return window_start, window_end

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help='file containing the ascii map')
    parser.add_argument('-d', '--dir', help='directory where to write output files', default='.')
    parser.add_argument('-o', '--out', help='output filename')
    parser.add_argument('-t', '--w-guests', help='list of windows of arrivals', metavar='WINDOW',
        nargs='+', required=True, type=window_type)
    parser.add_argument('-T', '--w-agents', help='list of windows of arrivals', metavar='WINDOW',
        nargs='+', required=True, type=window_type)
    parser.add_argument('-i', '--inter', help='interarrival time between tasks', default=20, type=float)
    parser.add_argument('-n', '--num', help='number of schedules to be generated', default=1, type=int)
    parser.add_argument('-u', '--rules', help='file containing traffic rules for agents', default=None, type=argparse.FileType('r'))
    parser.add_argument('-s', '--swap', help='swap pickup and delivery locations for guests', action='store_true')

    return parser.parse_args()

def read_ascii_map(filename):
    """Build a dictionary reading from an ascii map.
    
    'T' chars represent obstacles
    'a' chars represent the initial position of an agent
    'g' chars represent the initial position of a guest
    '@' chars represent a delivery location for the agents
    '#' chars represent a delivery location for the guests
    '.' chars represent a possible pickup location for both agents and guests
    '+' chars represent a possible pickup location for guests
    '-' chars represent a possible pickup location for agents
    """

    n_agents = 0
    n_guests = 0
    env_dict = {
        'guests': [],
        'agents': [],
        'map': {
            'dimensions': [0, 0],
            'obstacles': [],
            'non_task_endpoints': [],
            'pickup_agents': [],
            'delivery_agents': [],
            'non_task_endpoints_guests': [],
            'pickup_guests': [],
            'delivery_guests': [],
            'invalid_pickups': [],
            }
        }

    cwd = os.getcwd()
    with open(os.path.join(cwd, filename)) as ascii_map:
        width, height = (0, 0)
        y = 0
        line = ascii_map.readline().strip()
        while line:
            if y + 1 > height:
                height = y + 1

            for x, cell in enumerate(line):
                if x + 1 > width:
                    width = x + 1

                if cell == 'T':
                    env_dict['map']['obstacles'].append((x,y))
                    continue
                if cell == 'a':
                    n_agents += 1
                    env_dict['agents'].append({'start': [x, y], 'name': 'agent' + str(n_agents)})
                    env_dict['map']['non_task_endpoints'].append((x, y))
                    continue
                if cell == 'g':
                    n_guests += 1
                    env_dict['guests'].append({'start': [x, y], 'name': 'guest' + str(n_guests)})
                    env_dict['map']['non_task_endpoints_guests'].append((x, y))
                    continue
                if cell == '@':
                    env_dict['map']['delivery_agents'].append((x, y))
                    continue
                if cell == '#':
                    env_dict['map']['delivery_guests'].append((x, y))
                    continue
                if cell == '.':
                    env_dict['map']['pickup_agents'].append((x, y))
                    env_dict['map']['pickup_guests'].append((x, y))
                    continue
                if cell == '+':
                    env_dict['map']['pickup_guests'].append((x, y))
                    continue
                if cell == '-':
                    env_dict['map']['pickup_agents'].append((x, y))
                    continue
                # Every other character is not a valid pickup location
                env_dict['map']['invalid_pickups'].append((x, y))

            y += 1
            line = ascii_map.readline().strip()
        
        env_dict['map']['dimensions'] = [width, height]
        # Mirror tuples along y-axis
        mirror = lambda t: (t[0], height - t[1] - 1)
        env_dict['map']['obstacles'] = list(map(mirror, env_dict['map']['obstacles']))
        env_dict['map']['non_task_endpoints'] = list(map(mirror, env_dict['map']['non_task_endpoints']))
        env_dict['map']['non_task_endpoints_guests'] = list(map(mirror, env_dict['map']['non_task_endpoints_guests']))
        env_dict['map']['delivery_agents'] = list(map(mirror, env_dict['map']['delivery_agents']))
        env_dict['map']['delivery_guests'] = list(map(mirror, env_dict['map']['delivery_guests']))
        env_dict['map']['pickup_agents'] = list(map(mirror, env_dict['map']['pickup_agents']))
        env_dict['map']['pickup_guests'] = list(map(mirror, env_dict['map']['pickup_guests']))
        env_dict['map']['invalid_pickups'] = list(map(mirror, env_dict['map']['invalid_pickups']))
        def mirror_start(agent_dict):
            agent_dict['start'] = list(mirror(agent_dict['start']))
            return agent_dict
        env_dict['agents'] = list(map(mirror_start, env_dict['agents']))
        env_dict['guests'] = list(map(mirror_start, env_dict['guests']))

    return env_dict

def read_traffic_rule_map(traffic_map):
    """Return the set of illegal directions reading from an ascii map.

    '>' chars represent a one-way street heading east
    '<' chars represent a one-way street heading west
    'v' chars represent a one-way street heading south
    '^' chars represent a one-way street heading north
    """

    forbidden_moves = []
    width, height = (0, 0)
    y = 0
    line = traffic_map.readline().strip()
    while line:
        if y + 1 > height:
            height = y + 1

        for x, cell in enumerate(line):
            if x + 1 > width:
                width = x + 1

            if cell == '>':
                forbidden_moves.append(((x, y), (x-1, y)))
                continue
            if cell == '<':
                forbidden_moves.append(((x, y), (x+1, y)))
                continue
            if cell == 'v':
                # Beware: it's the opposite but it gets mirrored later
                forbidden_moves.append(((x, y), (x, y-1)))
                continue
            if cell == '^':
                # Beware: it's the opposite but it gets mirrored later
                forbidden_moves.append(((x, y), (x, y+1)))
                continue
            # Every other character is not considered

        y += 1
        line = traffic_map.readline().strip()

    # Mirror tuples along y-axis
    mirror = lambda t: ((t[0][0], height - t[0][1] - 1), (t[1][0], height - t[1][1] - 1))
    forbidden_moves = list(map(mirror, forbidden_moves))

    # Eliminate oob illegal moves
    forbidden_moves = list(filter(lambda t: \
        0 <= t[1][0] < width and 0 <= t[1][1] < height,
        forbidden_moves
    ))

    return {
        'forbidden_moves': forbidden_moves,
        'width': width,
        'height': height
    }

def sample_task_locations(map_size, obstacles, delivery):
    """Randomly generate a task pickup and a task delivery."""

    width, height = map_size
    if not set(delivery) & set(obstacles):
        obstacles |= delivery
    n_start_locations = width * height - len(obstacles)

    # Task start
    start_idx = np.random.randint(n_start_locations)

    obstacle_matrix = np.zeros((width, height), dtype=int)
    for obstacle in obstacles:
        obstacle_matrix[tuple(obstacle)] = 1
    obstacle_matrix = obstacle_matrix.reshape((-1,))
    i = 0
    while i < start_idx or obstacle_matrix[i] == 1:
        start_idx += obstacle_matrix[i]
        i += 1

    start_x = int(start_idx // height)
    start_y = int(start_idx % height)
    task_start = [start_x, start_y]
    assert tuple(task_start) not in obstacles

    # Task goal
    goal_idx = np.random.randint(len(delivery))
    task_goal = list(delivery[goal_idx])

    return (task_start, task_goal)

def generate_task_assignment(n_agents, map_size, obstacles, delivery, interarrival_time, end_time, start_time=0, task_count_ofs=0):
    """."""

    tasks = []

    arrivals = np.random.binomial(1, p=(1 / interarrival_time), size=(end_time - start_time, n_agents))
    t = 0
    while t < end_time - start_time:
        for _ in range(sum(arrivals[t])):
            start, goal = sample_task_locations(map_size, obstacles, delivery)
            task = {
                'start_time': t + start_time,
                'start': start,
                'goal': goal,
                'task_name': f'task{len(tasks) + task_count_ofs}'
            }
            tasks.append(task)
        t += 1

    return tasks

def save_instance(env_dict, tasks_agent, tasks_guest, name, out_dir):
    env_dict['tasks'] = tasks_agent
    env_dict['tasks_guest'] = tasks_guest

    dir_name = os.path.join(os.getcwd(), out_dir)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    with open(os.path.join(dir_name, name + '.yaml'), 'w') as instance_file:
        #yaml.safe_dump(env_dict, instance_file, default_flow_style=None)
        yaml.safe_dump(env_dict, instance_file)
        

def main():
    args = parse_args()

    env_dict = read_ascii_map(args.filename)
    invalid_pickups = env_dict['map'].pop('invalid_pickups')
    pickup_agents = env_dict['map'].pop('pickup_agents')
    pickup_guests = env_dict['map'].pop('pickup_guests')
    #print(env_dict)

    n_agents = len(env_dict['agents'])
    n_guests = len(env_dict['guests'])
    map_size = tuple(env_dict['map']['dimensions'])

    if args.rules is not None:
        traffic_rule_dict = read_traffic_rule_map(args.rules)
        forbidden_moves = traffic_rule_dict.pop('forbidden_moves')
        rule_width = traffic_rule_dict.pop('width')
        rule_height = traffic_rule_dict.pop('height')
        if (rule_width, rule_height) != map_size:
            raise ValueError('Mismatching size between map and traffic rule file')
        env_dict['map'].update([('forbidden_moves_agents', forbidden_moves)])

    invalid_task_start_locations_agents = \
        set(product(range(map_size[0]), range(map_size[1]))) - \
        set(pickup_agents)

#    invalid_task_start_locations_agents = \
#        set(env_dict['map']['obstacles']) \
#        | set(env_dict['map']['delivery_agents']) \
#        | set(env_dict['map']['delivery_guests']) \
#        | set(env_dict['map']['non_task_endpoints']) \
#        | set(env_dict['map']['non_task_endpoints_guests']) \
#        | set(invalid_pickups)

    if args.swap:
        env_dict['map']['delivery_guests'] = pickup_agents

    invalid_task_start_locations_guests = \
        set(product(range(map_size[0]), range(map_size[1]))) - \
        set(pickup_guests)

#    invalid_task_start_locations_guests = \
#        set(env_dict['map']['obstacles']) \
#        | set(env_dict['map']['delivery_agents']) \
#        | set(env_dict['map']['delivery_guests']) \
#        | set(env_dict['map']['non_task_endpoints']) \
#        | set(env_dict['map']['non_task_endpoints_guests']) \
#        | set(invalid_pickups)

    for i in range(args.num):
        tasks_agent = []
        tasks_guest = []
        for window_start, window_end in args.w_agents:
            tasks = generate_task_assignment(
                n_agents,
                map_size,
                invalid_task_start_locations_agents,
                delivery = env_dict['map']['delivery_agents'],
                interarrival_time = args.inter,
                start_time = window_start,
                end_time = window_end,
                task_count_ofs=len(tasks_agent))
            tasks_agent += tasks

        for window_start, window_end in args.w_guests:
            tasks = generate_task_assignment(
                n_guests,
                map_size,
                invalid_task_start_locations_guests,
                delivery = env_dict['map']['delivery_guests'],
                interarrival_time = args.inter,
                start_time = window_start,
                end_time = window_end,
                task_count_ofs=len(tasks_agent) + len(tasks_guest))
            tasks_guest += tasks

        if args.out is not None:
            instance_name = args.out
        else:
            instance_name = os.path.splitext(os.path.basename(args.filename))[0]
        if args.swap:
            instance_name += '_swap'
        instance_name += '_' + str(i + 1).rjust(len(str(args.num)), '0')
        save_instance(env_dict, tasks_agent, tasks_guest, name = instance_name, out_dir = args.dir)



if __name__ == '__main__':
    main()
