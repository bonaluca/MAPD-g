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
from heapq import heappush, heappop
from itertools import count
import networkx as nx
from networkx.algorithms.shortest_paths.generic import _build_paths_from_predecessors

class AStar:
    def __init__(self, env):
        self.agent_dict = env.agent_dict
        self.admissible_heuristic = env.admissible_heuristic
        self.is_at_goal = env.is_at_goal
        self.get_neighbors = env.get_neighbors
        self.max_iter = env.a_star_max_iter
        self.iter = 0

    def reconstruct_path(self, came_from, current):
        total_path = [current]
        while current in came_from.keys():
            current = came_from[current]
            total_path.append(current)
        return total_path[::-1]




    def _weight_function(self, G, weight):
        if callable(weight):
            return weight
        # If the weight keyword argument is not callable, we assume it is a
        # string representing the edge attribute containing the weight of
        # the edge.
        if G.is_multigraph():
            return lambda u, v, d: min(attr.get(weight, 1) for attr in d.values())
        return lambda u, v, data: data.get(weight, 1)

    def _dijkstra(self, G, source, weight="weight", pred=None, paths=None, cutoff=None, target=None):
        return self._dijkstra_multisource(
            G, [source], weight, pred=pred, paths=paths, cutoff=cutoff, target=target    )


    def dijkstra_path(self, G, source, target, weight="weight"):
        (length, path) = self.single_source_dijkstra(G, source, target=target, weight=weight)
        return path

    def dijkstra_path_length(self, G, source, target, weight="weight"):
        if source not in G:
            raise nx.NodeNotFound(f"Node {source} not found in graph")
        if source == target:
            return 0
        weight = self._weight_function(G, weight)
        length = self._dijkstra(G, source, weight, target=target)
        try:
            return length[target]
        except KeyError as err:
            raise nx.NetworkXNoPath(f"Node {target} not reachable from {source}") from err

    def single_source_dijkstra(self, G, source, target=None, cutoff=None, weight="weight"):  
        return self.multi_source_dijkstra(G, {source}, cutoff=cutoff, target=target, weight=weight)

    def multi_source_dijkstra(self, G, sources, target=None, cutoff=None, weight="weight"):
        if not sources:
            raise ValueError("sources must not be empty")
        for s in sources:
            if s[1:3] not in G:
                raise nx.NodeNotFound(f"Node {s} not found in graph")
        if target in sources:
            return (0, [target])
        weight = self._weight_function(G, weight)
        paths = {source: [source] for source in sources}  # dictionary of paths
        dist, v = self._dijkstra_multisource(
            G, sources, weight, paths=paths, cutoff=cutoff, target=target
        )
        if target is None:
            return (dist, paths)
        try:
            final_path = [node[1:3] for node in paths[v]]
            #print('final path',final_path)
            return (dist[v], final_path)
        except KeyError as err:
            raise nx.NetworkXNoPath(f"No path to {target}.") from err

    def _dijkstra_multisource(self, G, sources, weight='weight', pred=None, paths=None, cutoff=None, target=None):
        G_succ = G._succ if G.is_directed() else G._adj

        push = heappush
        pop = heappop
        dist = {}  # dictionary of final distances
        seen = {}
        #paths={}
        # fringe is heapq with 3-tuples (distance,c,node)
        # use the count c to avoid comparing nodes (may not be able to)
        c = count()
        fringe = []
        for source in sources:
            seen[source] = 0
            push(fringe, (0, next(c), source))

        while fringe:
            (d, mezzo, v) = pop(fringe)
            #print((len(paths[v])-1,d, mezzo, v))
            #print((d, mezzo, v))
            #prova = [len(paths[v])-1, list(v)[0], list(v)[1]]
            #prova = [d, list(v)[0], list(v)[1]]
            neighbor_list=self.get_neighbors(list(v))
            #if v in dist:
            #    continue  # already searched this node.
            dist[v] = d
            #print(v)
            if v[0]>3000:
                break
            if v[1:3] == target:
                break
            neighbor_list_tuple = [((i.time,i.location.x, i.location.y), cost) for (i, cost) in neighbor_list]
            for u, cost in neighbor_list_tuple:
                #cost = G[v[1:3]][u[1:3]][0]["weight"]  # @bonaluca
                #print('siamo in',v,'e il costo di',u,'è',cost)
                if cost is None:
                    continue
                vu_dist = dist[v] + cost
                if cutoff is not None:
                    if vu_dist > cutoff:
                        continue
                if u in dist:
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


    def search(self, agent_name, graph):
        """
        low level search
        """
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

        source = (0,initial_state.location.x,initial_state.location.y)       
        target=(self.agent_dict[agent_name]["goal"].location.x,self.agent_dict[agent_name]["goal"].location.y)
        

        try:
            path_def = self.dijkstra_path(graph,source=source,target=target)
            return path_def
        except:
            logging.info('Path not found')
            return False



