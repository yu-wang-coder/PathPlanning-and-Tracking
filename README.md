## 路径规划和跟踪算法的实现

这个仓库实现了经典的路径规划和跟踪算法，包括

### 路劲规划算法：

* 快速随机搜索树算法(RRT)
* A*算法
* 混合A*算法

### 跟踪算法：

* 纯跟踪算法(Pure Pursuit)+PID
* 动态线性二次调节器算法(DLQR)+PID
* 运动学线性二次调节器算法(KLQR)+PID

## Requirement
* Python 3.6及以上
* SciPy
* [cvxpy](https://github.com/cvxgrp/cvxpy)
* [pycubicspline](https://github.com/AtsushiSakai/pycubicspline)

## 代码内容介绍
```
│  README.md
│  
├─PathPlanning
│  │  astar.py
│  │  Astar_init.py
│  │  draw.py
│  │  env.py
│  │  hybrid_astar.py                    # 运行A*和混合A*的代码
│  │  plotting.py
│  │  plotting_rrt.py
│  │  reeds_shepp.py
│  │  rrt.py                             # 运行RRT算法的代码
│  │  utils.py 
│          
└─Tracking
    │  config_control.py
    │  draw.py
    │  draw_lqr.py
    │  LQR_Dynamics_Model.py             # 运行DLQR算法的代码
    │  LQR_Kinematic_Model.py            # 运行KLQR算法的代码
    │  Pure_Pursuit.py                   # 运行纯控制算法的代码
    │  reeds_shepp.py
```
## 代码参考
[https://github.com/zhm-real/MotionPlanning](https://github.com/zhm-real/MotionPlanning)