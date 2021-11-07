
import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.image as mpimg
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.image import BboxImage
from matplotlib.transforms import Bbox, TransformedBbox
import numpy as np


FILE = '../output/test_policy_trajectory_log.txt'
LINE = 4
DIMS = (6,6)
ENV_INFO = {
    'P':{'loc':'r26', 'color':'y'},
    'D':{'loc':'r9', 'color':'blue'},
    'R1':{'loc':'r28', 'color':'green'},
    'R2':{'loc':'r16', 'color':'green'}
}

class BasePlot(object):

    def __init__(self, dims):
        self.width = dims[0]
        self.height = dims[1]
        scale = 1.5
        
        # set up plot
        self.fig, self.ax = plt.subplots(figsize=[self.width*scale, self.height*scale], facecolor='xkcd:charcoal grey')
        self.ax.axis([0,self.width,0,self.height])
        self.ax.grid(True)
        self.ax.grid(linewidth=2)
        self.ax.set_xticks(range(0,self.width))
        self.ax.set_yticks(range(0,self.height))
        # get rid of labels
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        # get rid of ticks
        for tic in self.ax.xaxis.get_major_ticks():
            tic.tick1On = tic.tick2On = False
        for tic in self.ax.yaxis.get_major_ticks():
            tic.tick1On = tic.tick2On = False

        self.ax.invert_yaxis()

    def draw_env(self,env_info):
        for l,d in env_info.items():
            r = d['loc'].replace('r', '')
            r = int(r)
            x, y = self._r2c(r)
            c = d['color']
            self._draw_square(x,y,c, label=l)

    def _draw_square(self,x,y,color, label = None):
        x1 = x-0.5
        x2 = x+0.5
        y1 = y-0.5
        y2 = y+0.5
        self.ax.fill([x1, x2, x2, x1], [y2, y2, y1, y1], color, label = label)

    def _r2c(self, r):
        y, x = np.unravel_index(r, (self.height, self.width)) # pylint: disable=unbalanced-tuple-unpacking
        x += 0.5
        y += 0.5
        return x, y

    def show(self):
        plt.show()


class ArrowPlot(BasePlot):

    def __init__(self, dims, trajectory):
        super(ArrowPlot, self).__init__(dims)
        self.trajectory = trajectory

    def draw_path(self, path, arrow_seq=None, color='black', scale=5):
        xdata, ydata = self._path_to_xy(path)
        self._draw_line(xdata, ydata, color=color, scale=scale)
        self._draw_arrows(xdata, ydata, color=color, arrow_seq=arrow_seq, scale=scale)

    def _draw_line(self, xdata, ydata, scale=5, color='black'):
        line = Line2D(xdata, ydata, linewidth=scale, color=color)
        self.ax.add_artist(line)

    def _draw_arrows(self, x, y, arrow_seq=None, color='black', scale=5):
        # Used ideas from multiple answers of https://stackoverflow.com/questions/8247973/how-do-i-specify-an-arrow-like-linestyle-in-matplotlib
        BEFORE_LEN = 0.8
        AFTER_LEN = 0.4
        ARROW_SCALE = 6
        LOOP_DIA = 0.5
        LOOP_ROT_ANG = 0
        LOOP_END_ANG = 300

        if arrow_seq == None:
            raise Exception('Smart arrow making not implemented. Use arrow_seq.')
        elif len(arrow_seq) != len(x):
            raise Exception('Arrow sequence must define arrow type for each movement')
        elif arrow_seq[0] == 'before' or arrow_seq[-1] == 'after':
            raise Exception('Cannot have arrow before first or after last')

        theta = np.arctan2(-(y[1:] - y[:-1]), x[1:] - x[:-1])
        path = np.array([x,y]).T
        dist = np.sum((path[1:] - path[:-1]) ** 2, axis=1) ** .5

        ax0 = []
        ax1 = []
        ay0 = []
        ay1 = []
        for i, arr_type in enumerate(arrow_seq):
            if arr_type == 'none':
                pass
            elif arr_type == 'before':
                ax0.append(x[i-1])
                ax1.append(x[i-1] + dist[i-1] * np.cos(theta[i-1]) * BEFORE_LEN)
                ay0.append(y[i-1])
                ay1.append(y[i-1] + dist[i-1] * -np.sin(theta[i-1]) * BEFORE_LEN)
            elif arr_type == 'after':
                ax0.append(x[i])
                ax1.append(x[i] + dist[i] * np.cos(theta[i]) * AFTER_LEN)
                ay0.append(y[i])
                ay1.append(y[i] + dist[i] * -np.sin(theta[i]) * AFTER_LEN)
            elif arr_type == 'loop':
                # https://stackoverflow.com/a/38208040
                # Line
                arc = mpatches.Arc([x[i], y[i]], LOOP_DIA, LOOP_DIA, angle=LOOP_ROT_ANG, theta1=0, theta2=LOOP_END_ANG, capstyle='round', lw=scale, color=color, zorder=1.5)
                # arrow
                head_ang = LOOP_ROT_ANG+LOOP_END_ANG
                endx=x[i]+(LOOP_DIA/2)*np.cos(np.radians(head_ang)) #Do trig to determine end position
                endy=y[i]+(LOOP_DIA/2)*np.sin(np.radians(head_ang))
                head = mpatches.RegularPolygon((endx,endy), 3, LOOP_DIA/9, np.radians(head_ang), color=color, zorder=1.5)
                self.ax.add_patch(arc)
                self.ax.add_patch(head)
            else:
                raise Exception('Unrecognized arrow type/location: ' + arr_type)
                
        for x1, y1, x2, y2 in zip(ax0, ay0, ax1, ay1):
            self.ax.annotate('', xytext=(x1, y1), xycoords='data',
                xy=(x2, y2), textcoords='data',
                arrowprops=dict(arrowstyle='-|>', mutation_scale=scale*ARROW_SCALE, color='black', zorder=1.5))#, frac=1., ec=c, fc=c))

    def _path_to_xy(self, path):
        coords = [self._r2c(r) for r in path]
        xdata = np.array([c[0] for c in coords])
        ydata = np.array([c[1] for c in coords])
        return xdata, ydata


class AnimPlot(BasePlot):

    AGENT_WIDTH = 0.7
    AGENT_COLOR = 'blue'

    def __init__(self, dims, path, num_steps = 30, time_step = 0.5):
        super(AnimPlot, self).__init__(dims)
        self.path = path
        self.num_steps = num_steps
        self.time_step = time_step

        self.agent_path = self.gen_smooth_path(path, num_steps)

        # init_coords = self._r2c(self.path[0])
        agent_init_xy = self.agent_path[0]
        agent_bbox_xy = self._xy_to_bbox_xy(agent_init_xy)
        agent_bbox = Bbox(agent_bbox_xy)
        # self.bbox = Bbox([[2,2],[3,3]])
        agent_bbox_tf = TransformedBbox(agent_bbox, self.ax.transData)
        self.agent_bbox_image = BboxImage(agent_bbox_tf,
                       cmap=plt.get_cmap('winter'),
                       norm=None,
                       origin=None)
        drone_img = mpimg.imread('../lib/drone.png')
        self.agent_bbox_image.set_data(drone_img)
        self.ax.add_artist(self.agent_bbox_image)
        # plt.show()

        # self.agent_patch = mpatches.Circle(init_coords, self.AGENT_WIDTH, color=self.AGENT_COLOR)

    def anim_update(self,t):
        bbox_xy = self._xy_to_bbox_xy(self.agent_path[t])
        self.agent_bbox_image.bbox._bbox.update_from_data_xy(bbox_xy)
        # self.agent_patch.set(zorder=4)
        return [self.agent_bbox_image]

    def gen_smooth_path(self, r_path, numSteps):
        path = self._r_path_to_xy(r_path)
        self.init = path[0]
        smooth_path = [zip(np.linspace(c[0],c2[0],numSteps), np.linspace(c[1],c2[1],numSteps)) for c,c2 in zip(path[:-1],path[1:])]
        # flatten
        smooth_path = [item for sub in smooth_path for item in sub] # Maybe someday Python will allow unpacking in list comprehensions...
        return smooth_path
        
    def start(self):
        interval = int(float(self.time_step) / (self.num_steps) * 1000.0)
        # interval = 500
        sim_frames = len(self.agent_path)
        self.ani = FuncAnimation(self.fig,self.anim_update, frames = sim_frames, blit=True, interval = interval)
        plt.show()
        # self.fig.show()

    def save(self, filename):
        # gifWriter = PillowWriter(fps=30)
        # self.ani.save(filename, writer = gifWriter)
        writervideo = FFMpegWriter(fps=60)
        self.ani.save(filename, writer=writervideo)

    def set_title(self, title):
        self.ax.set_title(title)

    def _xy_to_bbox_xy(self, xy):
        x0 = xy[0] - self.AGENT_WIDTH/2
        y0 = xy[1] - self.AGENT_WIDTH/2
        x1 = xy[0] + self.AGENT_WIDTH/2
        y1 = xy[1] + self.AGENT_WIDTH/2
        return np.array([[x0,y0],[x1,y1]])

    def _r_path_to_xy(self, r_path):
        coords = [self._r2c(r) for r in r_path]
        return coords




def path_str_to_int(path_str):
    path_str_ls = path_str.split()
    path = [int(r) for r in path_str_ls]
    return path

def ansi_str_to_path_str(ansi_str):
    route = re.sub(r'\033\[(\d|;)+?m', '', ansi_str)
    route, _ = route.split('|')
    route = route.replace('r', '')
    return route

def ansi_str_to_path_int(ansi_str):
    path_str = ansi_str_to_path_str(ansi_str)
    path_int = path_str_to_int(path_str)
    return path_int


def main():
    with open(FILE, 'r') as f:
        lines = f.readlines()
    route_color = lines[LINE]
    route = ansi_str_to_path_int(route_color)
    print(route_color)
    print(route)

    ap = ArrowPlot(DIMS, route)
    # ap.ax.annotate('', xy = (0.5,0.5), xytext=(1.5,1.5), arrowprops=dict(arrowstyle='simple', mutation_scale=30, color='black'))

    # arrow_seq = [
    #     'after',
    #     'after',
    #     'none',
    #     'before',
    #     'before',
    #     'none',
    #     'before',
    #     'after',
    #     'loop'
    # ]

    arr = ['before']*len(route)
    arr[0] = 'none'
    arr[3] = 'loop'
    arr[7] = 'loop'
    # arr[5] = 'none'
    arr[14] = 'loop'
    arr[20] = 'loop'

    ap.draw_path(route, arr)

    # ap.draw_path([0, 1, 2, 3, 7, 6, 5, 4, 9], arrow_seq=arrow_seq)
    ap.draw_env(ENV_INFO)

    # ap.draw_arrow(9,10)
    # ap.draw_arrow(3,11)
    # ap.draw_arrow(4,13)
    ap.show()

def do_animation():
    with open(FILE, 'r') as f:
        lines = f.readlines()
    route_color = lines[LINE]
    route = ansi_str_to_path_int(route_color)
    print(route_color)
    print(route)

    anim = AnimPlot(DIMS, route)
    anim.start()
    # anim.draw_env()
    # anim.start()



if __name__ == '__main__':
    # main()
    do_animation()
