"""
AStar search
author: Ashwin Bose (@atb033)
author: Giacomo Lodigiani (@Lodz97)
"""
import heapq
from itertools import count
import networkx as nx
import logging
from collections import deque
from heapq import heappush, heappop, heapify
from itertools import count
import networkx as nx

class AStar:
    def __init__(self, env):
        self.agent_dict = env.agent_dict
        self.admissible_heuristic = env.admissible_heuristic
        self.is_at_goal = env.is_at_goal
        self.get_neighbors = env.get_neighbors
        self.max_iter = env.a_star_max_iter
        self.iter = 0
        self.time_start = env.time_start

    def reconstruct_path(self, predecessors, current):
        total_path = [current]
        while current in predecessors.keys():
            current = predecessors[current]
            total_path.append(current)
        return total_path[::-1]




    def _weight_function(self, graph, weight):
        if callable(weight):
            return weight
        # If the weight keyword argument is not callable, we assume it is a
        # string representing the edge attribute containing the weight of
        # the edge.
        if graph.is_multigraph():
            return lambda u, v, d: min(attr.get(weight, 1) for attr in d.values())
        return lambda u, v, data: data.get(weight, 1)

    def _dijkstra(self, source, weight="weight", pred=None, paths=None, cutoff=None, target=None):
        return self._dijkstra_multisource(
            [source], weight, pred=pred, paths=paths, cutoff=cutoff, target=target
            )


    def dijkstra_path(self, source, target):
        (length, path) = self.single_source_dijkstra(source, target=target)
        return path

    def dijkstra_path_length(self, graph, source, target, weight="weight"):
        if source not in graph:
            raise nx.NodeNotFound(f"Node {source} not found in graph")
        if source == target:
            return 0
        weight = self._weight_function(graph, weight)
        length = self._dijkstra(graph, source, weight, target=target)
        try:
            return length[target]
        except KeyError as err:
            raise nx.NetworkXNoPath(f"Node {target} not reachable from {source}") from err

    def single_source_dijkstra(self, source, target=None, cutoff=None):
        return self.multi_source_dijkstra({source}, cutoff=cutoff, target=target)

    def multi_source_dijkstra(self, sources, target=None, cutoff=None):
        if not sources:
            raise ValueError("sources must not be empty")
        if target in sources:
            return (0, [target])
        paths = {source: [source] for source in sources}  # dictionary of paths
        dist, v = self._dijkstra_multisource(
            sources, paths=paths, cutoff=cutoff, target=target
        )
        if target is None:
            return (dist, paths)
        try:
            final_path = [node[1:3] for node in paths[v]]
            #print('final path',final_path)
            return (dist[v], final_path)
        except KeyError as err:
            raise nx.NetworkXNoPath(f"No path to {target}.") from err

    def _dijkstra_multisource(self, sources, pred=None, paths=None, cutoff=None, target=None):
        push = heappush
        pop = heappop
        dist = {}  # dictionary of final distances
        seen = {}
        #paths={}
        # fringe is heapq with 3-tuples (distance,c,node)
        # use the count c to avoid comparing State objects in case of ties
        c = count()
        fringe = []
        for source in sources:
            seen[source] = 0
            push(fringe, (0, next(c), source))

        while fringe:
            (d, mezzo, v) = pop(fringe)

            neighbor_list=self.get_neighbors(list(v))

            dist[v] = d

            if v[0]>3000:
                break
            if v[1:3] == target:
                break
            neighbor_list_tuple = [((i.time,i.location.x, i.location.y), cost) for (i, cost) in neighbor_list]
            for u, cost in neighbor_list_tuple:
                #print('siamo in',v,'e il costo di',u,'è',cost)
                if cost is None:
                    continue
                vu_dist = dist[v] + cost
                if cutoff is not None and vu_dist > cutoff:
                    continue
                if u in dist:   # u has already been popped from fringe
                    u_dist = dist[u]
                    if vu_dist < u_dist:
                        logging.error("Contradictory paths found: negative weights?")
                        raise ValueError("Contradictory paths found:", "negative weights?")
                    elif pred is not None and vu_dist == u_dist:
                        pred[u].append(v)
                elif u not in seen or vu_dist < seen[u]:
                    seen[u] = vu_dist
                    push(fringe, (vu_dist, next(c), u))
                    if paths is not None:
                        paths[u] = paths[v] + [u]
                        #print(paths)
                    if pred is not None:
                        pred[u] = [v]
                elif vu_dist == seen[u]:
                    if pred is not None:
                        pred[u].append(v)
        return dist, v


    def search(self, agent_name):
        """Low level search."""

        initial_state = self.agent_dict[agent_name]["start"]
        step_cost = 1

        closed_set = set()
        open_set = {initial_state}

        came_from = {}

        g_score = {}
        g_score[initial_state] = 0

        f_score = {}
        #h1_score = self.admissible_heuristic(initial_state, agent_name)
        #f_score[initial_state] = h1_score

        heap = []
        index = count(0)
        #heapq.heappush(heap, (f_score[initial_state], h1_score, next(index), initial_state))

        source = (self.time_start, initial_state.location.x, initial_state.location.y)
        target=(self.agent_dict[agent_name]["goal"].location.x,self.agent_dict[agent_name]["goal"].location.y)


        try:
            path_def = self.dijkstra_path(source=source,target=target)
            return path_def
        except: # TODO @bonaluca: don't catch just any exception
            logging.info('Path not found')
            return False

class Dijkstra(AStar):
    def __init__(self, env):
        super().__init__(env)

    def search(self, agent_name):
        """Low level search."""

        initial_state = self.agent_dict[agent_name]['start']

        predecessors = {}
        tentative_dists = {}
        visited = {}
        tentative_dists[initial_state] = 0

        open_set = {initial_state}
        heap = []
        heappush(heap, (tentative_dists[initial_state], initial_state))

        while open_set:
            current = heappop(heap)[1]
            open_set -= {current}
            visited[current] = tentative_dists[current]

            if self.is_at_goal(current, agent_name):
                return self.reconstruct_path(predecessors, current)

            for (neighbor, dist) in self.get_neighbors(current):
                neighbor_dist = tentative_dists[current] + dist
                if neighbor_dist < tentative_dists.setdefault(neighbor, float("inf")):
                    if neighbor in open_set:
                        heap.remove((tentative_dists[neighbor], neighbor))
                        heapify(heap)
                    if neighbor in visited.keys():
                        logging.error('Dijkstra algorithm found a negative distance')
                        return False

                    # Update tentative distance
                    tentative_dists[neighbor] = neighbor_dist
                    predecessors[neighbor] = current
                    open_set |= {neighbor}
                    heappush(heap, (neighbor_dist, neighbor))
        return False


