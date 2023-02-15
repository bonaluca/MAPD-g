#!/usr/bin/env python3
import yaml
import matplotlib
# matplotlib.use("Agg")
from matplotlib.patches import Circle, Rectangle, Arrow, RegularPolygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
import matplotlib.animation as manimation
import argparse
import math
import json
import os
import RoothPath


Colors = ['orange', 'blue', 'orange', 'red']


class Animation:
    def __init__(self, map, schedule, slow_factor=10, alpha = 0, map_name=None, deadlock=None):
        self.map = map
        self.schedule = schedule
        self.slow_factor = slow_factor
        self.alpha = alpha
        self.map_name = map_name
        self.deadlock = deadlock
        self.combined_schedule = {}
        self.combined_schedule.update(self.schedule["schedule"])

        aspect = map["map"]["dimensions"][0] / map["map"]["dimensions"][1]

        self.fig = plt.figure(frameon=False, figsize=(4 * aspect, 4))
        self.ax = self.fig.add_subplot(111, aspect='equal')
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=None, hspace=None)
        # self.ax.set_frame_on(False)

        self.patches = []
        self.artists = []
        self.agents = dict()
        self.guests = dict()
        self.agent_names = dict()
        self.guest_names = dict()
        self.tasks = dict()
        self.tasks_guest = dict()
        # Create boundary patch
        xmin = -0.5
        ymin = -0.5
        xmax = map["map"]["dimensions"][0] - 0.5
        ymax = map["map"]["dimensions"][1] - 0.5

        # self.ax.relim()
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        # self.ax.set_xticks([])
        # self.ax.set_yticks([])
        # plt.axis('off')
        # self.ax.axis('tight')
        # self.ax.axis('off')

        self.patches.append(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, facecolor='none', edgecolor='red'))
        for x in range(map["map"]["dimensions"][0]):
            for y in range(map["map"]["dimensions"][1]):
                self.patches.append(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor='none', edgecolor='black'))
        for o in map["map"]["obstacles"]:
            x, y = o[0], o[1]
            self.patches.append(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor='black', edgecolor='black'))
        for e in map["map"]["non_task_endpoints"]:
            x, y = e[0], e[1]
            self.patches.append(Circle((x, y), 0.4, facecolor='green', edgecolor='black'))
        for e in map["map"]["non_task_endpoints_guests"]:
            x, y = e[0], e[1]
            self.patches.append(Circle((x, y), 0.4, facecolor='lightgreen', edgecolor='black'))

        task_colors = np.random.rand(len(map["tasks"]), 3)
        for t, i in zip(map["tasks"], range(len(map["tasks"]))):
            x_s, y_s = t['start'][0], t['start'][1]
            self.tasks[t['task_name']] = [Rectangle((x_s - 0.25, y_s - 0.25), 0.5, 0.5, facecolor=task_colors[i], edgecolor='black', alpha=0)]
            self.patches.append(self.tasks[t['task_name']][0])
        for t, i in zip(map["tasks"], range(len(map["tasks"]))):
            x_g, y_g = t['goal'][0], t['goal'][1]
            self.tasks[t['task_name']].append(RegularPolygon((x_g, y_g - 0.05), 3, 0.2, facecolor=task_colors[i], edgecolor='black', alpha=0))
            self.patches.append(self.tasks[t['task_name']][1])

        task_colors_guest = np.random.rand(len(map["tasks_guest"]), 3)
        for t, i in zip(map["tasks_guest"], range(len(map["tasks_guest"]))):
            x_s, y_s = t['start'][0], t['start'][1]
            self.tasks_guest[t['task_name']] = [RegularPolygon((x_s, y_s - 0.05), 6, 0.2, facecolor=task_colors[i], edgecolor='black', alpha=0)]
            self.patches.append(self.tasks_guest[t['task_name']][0])
        for t, i in zip(map["tasks_guest"], range(len(map["tasks_guest"]))):
            x_g, y_g = t['goal'][0], t['goal'][1]
            self.tasks_guest[t['task_name']].append(RegularPolygon((x_g, y_g - 0.05), 5, 0.2, facecolor=task_colors_guest[i], edgecolor='black', alpha=0))
            self.patches.append(self.tasks_guest[t['task_name']][1])

        # Create agents:
        self.T = 0
        # Draw goals first
        for d, i in zip(map["agents"], range(0, len(map["agents"]))):
            if 'goal' in d:
                self.patches.append(
                    Rectangle((d["goal"][0] - 0.25, d["goal"][1] - 0.25), 0.5, 0.5, facecolor=Colors[0], edgecolor='black',
                              alpha=0.5))
        for d, i in zip(map["agents"], range(0, len(map["agents"]))):
            name = d["name"]
            self.agents[name] = Circle((d["start"][0], d["start"][1]), 0.3, facecolor=Colors[0], edgecolor='black')
            self.agents[name].original_face_color = Colors[0]
            self.patches.append(self.agents[name])
            self.T = max(self.T, schedule["schedule"][name][-1]["t"])
            self.agent_names[name] = self.ax.text(d["start"][0], d["start"][1], name.replace('agent', ''))
            self.agent_names[name].set_horizontalalignment('center')
            self.agent_names[name].set_verticalalignment('center')
            self.artists.append(self.agent_names[name])

        # Create guests:
        self.S = 0
        # Draw goals first
        for d, i in zip(map["guests"], range(0, len(map["guests"]))):
            if 'goal' in d:
                self.patches.append(
                    Rectangle((d["goal"][0] - 0.25, d["goal"][1] - 0.25), 0.5, 0.5, facecolor=Colors[0], edgecolor='black',
                              alpha=0.5))
        for d, i in zip(map["guests"], range(0, len(map["guests"]))):
            name = d["name"]
            self.guests[name] = Circle((d["start"][0], d["start"][1]), 0.3, facecolor=Colors[3], edgecolor='black')
            self.guests[name].original_face_color = Colors[3]
            self.patches.append(self.guests[name])
            self.S = max(self.S, schedule["schedule"][name][-1]["t"])
            self.guest_names[name] = self.ax.text(d["start"][0], d["start"][1], name.replace('guest', ''))
            self.guest_names[name].set_horizontalalignment('center')
            self.guest_names[name].set_verticalalignment('center')
            self.artists.append(self.guest_names[name])

        # self.ax.set_axis_off()
        # self.fig.axes[0].set_visible(False)
        # self.fig.axes.get_yaxis().set_visible(False)

        # self.fig.tight_layout()

        self.M = max(self.T, self.S)

        self.anim = animation.FuncAnimation(self.fig, self.animate_func,
                                            init_func=self.init_func,
                                            frames=int(self.M + 1) * self.slow_factor,
                                            interval=10,
                                            blit=True,
                                            repeat=False)

    def save(self, file_name, speed):
        self.anim.save(
            file_name,
            "ffmpeg",
            fps=20 * speed,
            dpi=200),
        # savefig_kwargs={"pad_inches": 0, "bbox_inches": "tight"})

    def show(self):
        plt.show()

    def init_func(self):
        for p in self.patches:
            self.ax.add_patch(p)
        for a in self.artists:
            self.ax.add_artist(a)
        return self.patches + self.artists

    def animate_func(self, i):
        for agent_name, agent in self.combined_schedule.items():
            if agent_name != 'guest1' and agent_name != 'guest2' and agent_name != 'guest3'and agent_name != 'guest4' and agent_name != 'guest5':
                pos = self.getState(i / self.slow_factor, agent)
                p = (pos[0], pos[1])
                self.agents[agent_name].center = p
                self.agent_names[agent_name].set_position(p)

        for guest_name, guest in self.combined_schedule.items():
            if guest_name == 'guest1' or guest_name == 'guest2' or guest_name == 'guest3' or guest_name == 'guest4' or guest_name == 'guest5':
                pos = self.getState(i / self.slow_factor, guest)
                p = (pos[0], pos[1])
                self.guests[guest_name].center = p
                self.guest_names[guest_name].set_position(p)

        # Reset all colors agents
        for _, agent in self.agents.items():
            agent.set_facecolor(agent.original_face_color)

        # Reset all colors guests
        for _, guest in self.guests.items():
            guest.set_facecolor(guest.original_face_color)

        # Make tasks visible at the right time
        for t in map["tasks"]:
            try:
                if t['start_time'] <= i / self.slow_factor + 1 <= self.schedule['completed_tasks_times'][t['task_name']]:
                    self.tasks[t['task_name']][0].set_alpha(0.5)
                    self.tasks[t['task_name']][1].set_alpha(0.5)
                else:
                    self.tasks[t['task_name']][0].set_alpha(0)
                    self.tasks[t['task_name']][1].set_alpha(0)
            except:
                continue


        # Make guest tasks visible at the right time
        for t in map["tasks_guest"]:
            try:
                if t['start_time'] <= i / self.slow_factor + 1 <= self.schedule['completed_tasks_times_guest'][t['task_name']]:
                    self.tasks_guest[t['task_name']][0].set_alpha(0.5)
                    self.tasks_guest[t['task_name']][1].set_alpha(0.5)
                else:
                    self.tasks_guest[t['task_name']][0].set_alpha(0)
                    self.tasks_guest[t['task_name']][1].set_alpha(0)
            except:
                continue

        # Check drive-drive collisions agents
        agents_array = [agent for _, agent in self.agents.items()]
        for j in range(0, len(agents_array)):
            for k in range(j + 1, len(agents_array)):
                d1 = agents_array[j]
                d2 = agents_array[k]
                pos1 = np.array(d1.center)
                pos2 = np.array(d2.center)
                if np.linalg.norm(pos1 - pos2) < 0.7:
                    d1.set_facecolor('blue')
                    d2.set_facecolor('blue')
                    print("COLLISION! (agent-agent) ({}, {})".format(j, k))

        # Check drive-drive collisions guests
        guests_array = [guest for _, guest in self.guests.items()]
        for j in range(0, len(guests_array)):
            for k in range(j + 1, len(guests_array)):
                d1 = guests_array[j]
                d2 = guests_array[k]
                pos1 = np.array(d1.center)
                pos2 = np.array(d2.center)
                if np.linalg.norm(pos1 - pos2) < 0.7:
                    d1.set_facecolor('blue')
                    d2.set_facecolor('blue')
                    print("COLLISION! (guest-guest) ({}, {})".format(j, k))

        # Check drive-drive collisions agent-guest
        guests_array = [guest for _, guest in self.guests.items()]
        agents_array = [agent for _, agent in self.agents.items()]
        for i in range(0, len(guests_array)):
            for j in range(0, len(agents_array)):
                d1 = guests_array[i]
                d2 = agents_array[j]
                pos1 = np.array(d1.center)
                pos2 = np.array(d2.center)
                if np.linalg.norm(pos1 - pos2) < 0.7:
                    d1.set_facecolor('blue')
                    d2.set_facecolor('blue')
                    print("COLLISION! (guest-agent) ({}, {})".format(i, j))

        return self.patches + self.artists

    def getState(self, t, d):
        idx = 0
        while idx < len(d) and d[idx]["t"] < t:
            idx += 1
        if idx == 0:
            return np.array([float(d[0]["x"]), float(d[0]["y"])])
        elif idx < len(d):
            posLast = np.array([float(d[idx - 1]["x"]), float(d[idx - 1]["y"])])
            posNext = np.array([float(d[idx]["x"]), float(d[idx]["y"])])
        else:
            return np.array([float(d[-1]["x"]), float(d[-1]["y"])])
        dt = d[idx]["t"] - d[idx - 1]["t"]
        t = (t - d[idx - 1]["t"]) / dt
        pos = (posNext - posLast) * t + posLast
        return pos


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-map", help="input file containing map")
    parser.add_argument("-schedule", help="schedule for agents")
    parser.add_argument('-slow_factor', help='Slow factor of visualization', default=1, type=int)
    parser.add_argument('-alpha', help='Slow factor of visualization', type=float)
    parser.add_argument('-map_name', help='Slow factor of visualization')
    parser.add_argument('--video', dest='video', default=None,
                        help="output video file (or leave empty to show on screen)")
    parser.add_argument("--speed", type=int, default=1, help="speedup-factor")
    parser.add_argument('-deadlock', help='Presence of deadlock')
    args = parser.parse_args()

    if args.map is None:
        with open(os.path.join(RoothPath.get_root(), 'config.json'), 'r') as json_file:
            config = json.load(json_file)
        args.map = os.path.join(RoothPath.get_root(), os.path.join(config['input_path'], str(args.map_name) + config['visual_postfix'],))
        if args.deadlock == str(True):
            args.schedule = os.path.join(RoothPath.get_root(), 'output', str(args.alpha)+'-DL-'+str(args.map_name))
        if args.deadlock == str(False):
            args.schedule = os.path.join(RoothPath.get_root(), 'output', str(args.alpha)+'-'+str(args.map_name))

    with open(args.map) as map_file:
        map = yaml.load(map_file, Loader=yaml.FullLoader)

    with open(args.schedule) as states_file:
        schedule = yaml.load(states_file, Loader=yaml.FullLoader)

    animation = Animation(map, schedule, slow_factor=args.slow_factor, alpha=args.alpha)

    #animation.save('TP_k=1_collision.mp4', 1)


    if args.video:
        animation.save(args.video, args.speed)
    else:
        animation.show()


