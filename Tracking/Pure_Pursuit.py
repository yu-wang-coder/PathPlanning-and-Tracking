import math
import time
import numpy as np
import matplotlib.pyplot as plt
import draw
import reeds_shepp as rs


class C:
    # PID 配置信息
    Kp = 0.3  # 比例增益

    # 系统配置
    Ld = 2.6  # 前瞻距离
    kf = 0.1  # 前瞻增益
    dt = 0.1  # 时间步长
    dist_stop = 0.7  # 停止距离
    dc = 0.0

    # 一些车辆参数
    RF = 3.3  # [m] distance from rear to vehicle front end of vehicle
    RB = 0.8  # [m] distance from rear to vehicle back end of vehicle
    W = 2.4  # [m] width of vehicle
    WD = 0.7 * W  # [m] distance between left-right wheels
    WB = 2.5  # [m] Wheel base
    TR = 0.44  # [m] Tyre radius
    TW = 0.7  # [m] Tyre width
    MAX_STEER = 0.30
    MAX_ACCELERATION = 5.0


class Node: # 表示路径上的一个节点，包含位置、偏航角、速度、方向
    def __init__(self, x, y, yaw, v, direct):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.direct = direct

    def update(self, a, delta, direct): # 节点位置和状态的更新
        self.x += self.v * math.cos(self.yaw) * C.dt
        self.y += self.v * math.sin(self.yaw) * C.dt
        self.yaw += self.v / C.WB * math.tan(delta) * C.dt
        self.direct = direct
        self.v += self.direct * a * C.dt

    @staticmethod
    def limit_input(delta):
        if delta > 1.2 * C.MAX_STEER:
            return 1.2 * C.MAX_STEER

        if delta < -1.2 * C.MAX_STEER:
            return -1.2 * C.MAX_STEER

        return delta


class Nodes: # 用于管理和存储多个节点的位置、速度等信息
    def __init__(self):
        self.x = []
        self.y = []
        self.yaw = []
        self.v = []
        self.t = []
        self.direct = []

    def add(self, t, node):
        self.x.append(node.x)
        self.y.append(node.y)
        self.yaw.append(node.yaw)
        self.v.append(node.v)
        self.t.append(t)
        self.direct.append(node.direct)


class PATH: # 用来管理参考路径，计算跟踪目标点的索引和距离
    def __init__(self, cx, cy):
        self.cx = cx
        self.cy = cy
        self.ind_end = len(self.cx) - 1
        self.index_old = None

    def target_index(self, node):
        """
        search index of target point in the reference path.
        the distance between target point and current position is ld
        :param node: current information
        :return: index of target point
        """

        if self.index_old is None:
            self.calc_nearest_ind(node)

        Lf = C.kf * node.v + C.Ld

        for ind in range(self.index_old, self.ind_end + 1):
            if self.calc_distance(node, ind) > Lf:
                self.index_old = ind
                return ind, Lf

        self.index_old = self.ind_end

        return self.ind_end, Lf

    def calc_nearest_ind(self, node):
        """
        calc index of the nearest point to current position
        :param node: current information
        :return: index of nearest point
        """

        dx = [node.x - x for x in self.cx]
        dy = [node.y - y for y in self.cy]
        ind = np.argmin(np.hypot(dx, dy))
        self.index_old = ind

    def calc_distance(self, node, ind):
        return math.hypot(node.x - self.cx[ind], node.y - self.cy[ind])


def pure_pursuit(node, ref_path, index_old): # 实现了纯追踪控制器，计算最优转向角度 delta
    """
    pure pursuit controller
    :param node: current information
    :param ref_path: reference path: x, y, yaw, curvature
    :param index_old: target index of last time
    :return: optimal steering angle
    """

    ind, Lf = ref_path.target_index(node)  # target point and pursuit distance
    ind = max(ind, index_old)

    tx = ref_path.cx[ind]
    ty = ref_path.cy[ind]

    alpha = math.atan2(ty - node.y, tx - node.x) - node.yaw
    delta = math.atan2(2.0 * C.WB * math.sin(alpha), Lf)

    return delta, ind


def pid_control(target_v, v, dist, direct): # 实现了PID控制器，设计速度控制策略，并返回期望加速度
    """
    PID controller and design speed profile.
    :param target_v: target speed (forward and backward are different)
    :param v: current speed
    :param dist: distance from current position to end position
    :param direct: current direction
    :return: desired acceleration
    """

    a = 0.3 * (target_v - direct * v)

    if dist < 10.0:
        if v > 3.0:
            a = -2.5
        elif v < -2.0:
            a = -1.0

    return a


def generate_path(s): # 根据给定的目标状态生成路径
    """
    divide paths into some sections, in each section, the direction is the same.
    :param s: target position and yaw
    :return: sections
    """

    max_c = math.tan(C.MAX_STEER) / C.WB  # max curvature

    path_x, path_y, yaw, direct = [], [], [], []
    x_rec, y_rec, yaw_rec, direct_rec = [], [], [], []
    direct_flag = 1.0

    for i in range(len(s) - 1):
        s_x, s_y, s_yaw = s[i][0], s[i][1], np.deg2rad(s[i][2])
        g_x, g_y, g_yaw = s[i + 1][0], s[i + 1][1], np.deg2rad(s[i + 1][2])

        path_i = rs.calc_optimal_path(s_x, s_y, s_yaw,
                                      g_x, g_y, g_yaw, max_c)

        ix = path_i.x
        iy = path_i.y
        iyaw = path_i.yaw
        idirect = path_i.directions

        for j in range(len(ix)):
            if idirect[j] == direct_flag:
                x_rec.append(ix[j])
                y_rec.append(iy[j])
                yaw_rec.append(iyaw[j])
                direct_rec.append(idirect[j])
            else:
                if len(x_rec) == 0 or direct_rec[0] != direct_flag:
                    direct_flag = idirect[j]
                    continue

                path_x.append(x_rec)
                path_y.append(y_rec)
                yaw.append(yaw_rec)
                direct.append(direct_rec)
                x_rec, y_rec, yaw_rec, direct_rec = \
                    [x_rec[-1]], [y_rec[-1]], [yaw_rec[-1]], [-direct_rec[-1]]

    path_x.append(x_rec)
    path_y.append(y_rec)
    yaw.append(yaw_rec)
    direct.append(direct_rec)

    x_all, y_all = [], []

    for ix, iy in zip(path_x, path_y):
        x_all += ix
        y_all += iy

    return path_x, path_y, yaw, direct, x_all, y_all

def design_obstacles(x, y): # 设置障碍物
    ox, oy = [], []

    for i in range(x):
        ox.append(i)
        oy.append(0)
    for i in range(x):
        ox.append(i)
        oy.append(y - 1)
    for i in range(y):
        ox.append(0)
        oy.append(i)
    for i in range(y):
        ox.append(x - 1)
        oy.append(i)
    for i in range(10, 21):
        ox.append(i)
        oy.append(15)
    for i in range(10, 15):
        ox.append(20)
        oy.append(i)
    for i in range(15, 30):
        ox.append(30)
        oy.append(i)
    for i in range(16):
        ox.append(40)
        oy.append(i)

    return ox, oy

def main(): # 主函数负责仿真过程，包括路径生成、控制器调用、动画绘制和误差统计

    states = [(10, 7, 0), (25, 5, 5), (34, 14, 65),(45, 20, 90)]

    x, y, yaw, direct, path_x, path_y = generate_path(states)

    # simulation
    maxTime = 100.0
    yaw_old = 0.0
    x0, y0, yaw0, direct0 = x[0][0], y[0][0], yaw[0][0], direct[0][0]
    x_rec, y_rec = [], []

    # 存储误差
    e_cg_history = []
    theta_e_history = []

    t1 = time.time()
    for cx, cy, cyaw, cdirect in zip(x, y, yaw, direct):
        t = 0.0
        node = Node(x=x0, y=y0, yaw=yaw0, v=0.0, direct=direct0)
        nodes = Nodes()
        nodes.add(t, node)
        ref_trajectory = PATH(cx, cy)
        target_ind, _ = ref_trajectory.target_index(node)

        while t <= maxTime:
            if cdirect[0] > 0:
                target_speed = 30.0 / 3.6
                C.Ld = 4.0
                C.dist_stop = 1.5
                C.dc = -1.1
            else:
                target_speed = 20.0 / 3.6
                C.Ld = 2.5
                C.dist_stop = 0.2
                C.dc = 0.2

            xt = node.x + C.dc * math.cos(node.yaw)
            yt = node.y + C.dc * math.sin(node.yaw)
            dist = math.hypot(xt - cx[-1], yt - cy[-1])

            if dist < C.dist_stop:
                break

            acceleration = pid_control(target_speed, node.v, dist, cdirect[0])
            delta, target_ind = pure_pursuit(node, ref_trajectory, target_ind)

            t += C.dt

            node.update(acceleration, delta, cdirect[0])
            nodes.add(t, node)
            x_rec.append(node.x)
            y_rec.append(node.y)

            dy = (node.yaw - yaw_old) / (node.v * C.dt)
            steer = rs.pi_2_pi(-math.atan(C.WB * dy))

            yaw_old = node.yaw
            x0 = nodes.x[-1]
            y0 = nodes.y[-1]
            yaw0 = nodes.yaw[-1]
            direct0 = nodes.direct[-1]

            # 计算跟踪误差
            e_cg_history.append(abs(math.hypot(cx[target_ind] - node.x, cy[target_ind] - node.y)))
            theta_e_history.append(abs(cyaw[target_ind]-node.yaw))

            # animation
            ox, oy = design_obstacles(51,31)
            
            plt.cla()
            plt.plot(ox, oy, "sk")
            plt.plot(node.x, node.y, marker='.', color='k')
            plt.plot(path_x, path_y, color='gray', linewidth=2)
            plt.plot(x_rec, y_rec, color='darkviolet', linewidth=2)
            plt.plot(cx[target_ind], cy[target_ind], ".r")
            draw.draw_car(node.x, node.y, yaw_old, steer, C)

            for m in range(len(states)):
                draw.Arrow(states[m][0], states[m][1], np.deg2rad(states[m][2]), 2, 'blue')

            plt.axis("equal")
            plt.title("PurePursuit: v=" + str(node.v * 3.6)[:4] + "km/h")
            plt.gcf().canvas.mpl_connect('key_release_event',
                                         lambda event:
                                         [exit(0) if event.key == 'escape' else None])
            plt.pause(0.001)

    t2 = time.time()
    plt.show()
    plt.figure()
    plt.plot(e_cg_history,label='Distance Error')
    plt.plot(theta_e_history, label='Yaw Error/rad')
    plt.xlabel('Time Steps')
    plt.ylabel('Error')
    plt.legend()
    plt.title('Trajectory Tracking Errors')
    plt.show()
    print('running T: ',t2 - t1)
    print('mean distance error: ', np.mean(np.array(e_cg_history)))
    print('mean yaw error: ', np.mean(np.array(theta_e_history)))
if __name__ == '__main__':
    main()
