# 以下部分均为可更改部分

from answer_task1 import *

import math
import numpy as np
from pfnn import PhaseFunctionedNetwork

# character, trajectory, pfnn

TRAJECTORY_LENGTH = 120
JOINT_NUM = 20
JOINT_NUM_SITE = 25

OPOS = int(8 + TRAJECTORY_LENGTH/2/10*4 + JOINT_NUM*3*0)
OVEL = int(8 + TRAJECTORY_LENGTH/2/10*4 + JOINT_NUM*3*1)
OROT = int(8 + TRAJECTORY_LENGTH/2/10*4 + JOINT_NUM*3*2)

class CharacterController():
    def __init__(self, controller) -> None:
        self.motions = []
        self.motions.append(BVHMotion('motion_material/long_walk.bvh'))
        # self.motions.append(BVHMotion('motion_material/walk_forward.bvh'))
        self.controller = controller
        self.cur_root_pos = None
        self.cur_root_rot = None
        self.cur_frame = 0
        
        
        # PFNN related
        self.pfnn = PhaseFunctionedNetwork(mode='predict')
        self.phase = 0
        self.strafe_amount = 0
        self.strafe_target = 0
        self.crouched_amount = 0
        self.crouched_target = 0
        self.responsive = 0
        self.extra_joint_smooth = 0.5

        # PFNN standards
        self.joint_positions_g = np.zeros((JOINT_NUM, 3))
        self.joint_rotations_g = np.zeros((JOINT_NUM, 4))
        self.joint_velocities_g = np.zeros((JOINT_NUM, 3))
        self.joint_parents = np.zeros((JOINT_NUM, 1))
        self.reset([0,0])
        
        # Games standards
        self.joint_positions_l_gm = []
        self.joint_velocities_l_gm = []
        self.joint_rotations_l_gm = []
        
        self.last_desired_pos_list = np.zeros((6, 3))
        
        pass
    
    def reset(self, position):
        Yp = self.pfnn.Ymean
        root_position_g = np.array([position[0], 0, position[1]])
        root_rotation_g = R.identity().as_quat()
        
        for i in range(JOINT_NUM):
            # P_i = Q_root * Y.l_i + P_root
            pos_g = R.from_quat(root_rotation_g).apply(Yp[OPOS + i*3: OPOS + i*3 + 3]) + root_position_g
            # Q_i = Q_root * Y.R_i
            rot_g = (R.from_quat(root_rotation_g) * R.from_quat([0, Yp[OROT + i*3 + 0], Yp[OROT + i*3 + 1], Yp[OROT + i*3 + 2]])).as_quat()
            vel_g = R.from_quat(root_rotation_g).apply(Yp[OVEL + i*3: OVEL + i*3 + 3])
            
            self.joint_positions_g[i] = pos_g
            self.joint_velocities_g[i] = vel_g
            self.joint_rotations_g[i] = rot_g
    
    
    def update_state(self, 
                     desired_pos_list, 
                     desired_rot_list,
                     desired_vel_list,
                     desired_avel_list,
                     current_gait
                     ):
        '''
        此接口会被用于获取新的期望状态
        Input: 平滑过的手柄输入,包含了现在(第0帧)和未来20,40,60,80,100帧的期望状态,以及一个额外输入的步态
        简单起见你可以先忽略步态输入,它是用来控制走路还是跑步的
            desired_pos_list: 期望位置, 6x3的矩阵, 每一行对应0，20，40...帧的期望位置(水平)， 期望位置可以用来拟合根节点位置也可以是质心位置或其他
            desired_rot_list: 期望旋转, 6x4的矩阵, 每一行对应0，20，40...帧的期望旋转(水平), 期望旋转可以用来拟合根节点旋转也可以是其他
            desired_vel_list: 期望速度, 6x3的矩阵, 每一行对应0，20，40...帧的期望速度(水平), 期望速度可以用来拟合根节点速度也可以是其他
            desired_avel_list: 期望角速度, 6x3的矩阵, 每一行对应0，20，40...帧的期望角速度(水平), 期望角速度可以用来拟合根节点角速度也可以是其他
        
        Output: 同作业一,输出下一帧的关节名字,关节位置,关节旋转
            joint_name: List[str], 代表了所有关节的名字
            joint_translation: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
            joint_orientation: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
        Tips:
            输出三者顺序需要对应
            controller 本身有一个move_speed属性,是形状(3,)的ndarray,
            分别对应着面朝向移动速度,侧向移动速度和向后移动速度.目前根据LAFAN的统计数据设为(1.75,1.5,1.25)
            如果和你的角色动作速度对不上,你可以在init或这里对属性进行修改
        '''
        # 一个简单的例子，输出第i帧的状态
        joint_name = self.motions[0].joint_name
        joint_translation, joint_orientation = self.motions[0].batch_forward_kinematics()
        joint_translation = joint_translation[self.cur_frame]
        joint_orientation = joint_orientation[self.cur_frame]
        
        self.cur_root_pos = joint_translation[0]
        self.cur_root_rot = joint_orientation[0]
        self.cur_frame = (self.cur_frame + 1) % self.motions[0].motion_length
        
        return joint_name, joint_translation, joint_orientation
    
    
    def sync_controller_and_character(self, controller, character_state):
        '''
        这一部分用于同步你的角色和手柄的状态
        更新后很有可能会出现手柄和角色的位置不一致，这里可以用于修正
        让手柄位置服从你的角色? 让角色位置服从手柄? 或者插值折中一下?
        需要你进行取舍
        Input: 手柄对象，角色状态
        手柄对象我们提供了set_pos和set_rot接口,输入分别是3维向量和四元数,会提取水平分量来设置手柄的位置和旋转
        角色状态实际上是一个tuple, (joint_name, joint_translation, joint_orientation),为你在update_state中返回的三个值
        你可以更新他们,并返回一个新的角色状态
        '''
        # print("--------------------------------")
        
        # 帧窗口根关节的xz坐标与中间帧(第61帧)作为轨迹位置root_possxz,     12x2 = 24
        # 窗口根关节的xz方向作为人体朝向root_dirxz,                      12x2 = 24
        # 手柄输入是 0, 20, 40, 60, 80, 100
        # 过去帧是 -20, -40, -60, -80, -100, -120
        # 意味着要记录120帧的轨迹位置和人体朝向
        
        # 窗口每帧步态的作为步态输入root_gait,                           12x6 = 72
        
        # 当前帧的所有关节的局部位置local_positions,                     20x3 = 60
        # 当前帧的所有关节的局部速度local vel,                           20x3 = 60
        
        # 当前关节附近12帧(已下采样10间隔)的左中右地形高度,                 12x3 = 36
        
        """
        pfnn的逻辑:
        1. 根据 (1)视角 (2)手柄 (3) 老target_vel&target_dir 插值平滑过渡到新值, 然后以velocity判断gait
        2. 根据步骤1的target_vel&target_dir, 计算轨迹的未来60帧的位置,方向,高度,步态
        
        --- [ 步骤1和步骤2可能就用最简单的方式 ]
        方式1: 1~60用老的,61是现在作为输入, 62~120直接用手柄输入
        方式2: 1~60用老的,61是现在作为输入, 62~120混合一下
        方法3: 完全重现pfnn原始方法
        
        3. 根据步骤1&步骤2&当前帧的关节数据 作为网络输入
        4. 预测
        5. 根据网络输出应用
        """
        
        desired_pos_list, desired_rot_list, desired_vel_list, desired_avel_list, current_gait = controller.get_desired_state()
        joint_name, joint_translation, joint_orientation = character_state
        
        # region 处理手柄输入
        # endregion
        # region 更新轨迹的目标方向和速度
        # endregion
        # region 更新轨迹的步态
        # endregion
        
        # region 预测轨迹的位置,方向,高度,步态
        trajectory_pos = np.concatenate((np.array([controller.recorded_pos[i] for i in range(len(controller.recorded_pos) - 1) if i % 20 == 0]), controller.future_pos), axis=0)
        trajectory_rot = np.concatenate((np.array([controller.recorded_rot[i] for i in range(len(controller.recorded_rot) - 1) if i % 20 == 0]), controller.future_rot), axis=0)
        trajectory_vel = np.concatenate((np.array([controller.recorded_vel[i] for i in range(len(controller.recorded_vel) - 1) if i % 20 == 0]), controller.future_vel), axis=0)
        trajectory_avel = np.concatenate((np.array([controller.recorded_avel[i] for i in range(len(controller.recorded_avel) - 1) if i % 20 == 0]), controller.future_avel), axis=0)
        # trajectory_gait = [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]] * 12
        trajectory_gait = [[1.0-current_gait, current_gait, 0.0, 0.0, 0.0, 0.0]] * 12
        print(trajectory_gait[6])
        # endregion
        
        # region 准备网络输入
        # 输入
        # X Input Dimension = 276
        Xp = [0] * 276
        for i in range(0, TRAJECTORY_LENGTH, 10):
            # Point-5: change from 12 -> 120 and with step of 10
            w = int(TRAJECTORY_LENGTH / 10)
            # Input Trajectory Positions / Directions
            # 1-12: 轨迹12帧 根关节的x坐标
            # 13-24: 轨迹12帧 根关节的z坐标
            # 25-36: 轨迹12帧 根关节的x方向
            # 37-48: 轨迹12帧 根关节的z方向
            Xp[int((w*0)+i/10)] = trajectory_pos[int(i/10)][0]
            Xp[int((w*1)+i/10)] = trajectory_pos[int(i/10)][2]
            Xp[int((w*2)+i/10)] = trajectory_vel[int(i/10)][0]
            Xp[int((w*3)+i/10)] = trajectory_vel[int(i/10)][2]
            
            # Input Trajectory Gaits
            # 49-120: 轨迹12帧 6种步态参数12*6
            Xp[int((w*4)+i/10)] = trajectory_gait[int(i/10)][0]
            Xp[int((w*5)+i/10)] = trajectory_gait[int(i/10)][1]
            Xp[int((w*6)+i/10)] = trajectory_gait[int(i/10)][2]
            Xp[int((w*7)+i/10)] = trajectory_gait[int(i/10)][3]
            Xp[int((w*8)+i/10)] = trajectory_gait[int(i/10)][4]
            Xp[int((w*9)+i/10)] = trajectory_gait[int(i/10)][5]
        
        # 121-180: 当前关节的局部位置，注意高度已经减去了地形均值 20*3=60
        # 181-240: 当前关节的局部速度20*3
        # Input Joint Previous Positions / Velocities / Rotations
        # root_position_g = trajectory_pos[int(TRAJECTORY_LENGTH/10/2)]
        # root_rotation_g = trajectory_rot[int(TRAJECTORY_LENGTH/10/2)]
        # prev_root_position_g = trajectory_pos[int(TRAJECTORY_LENGTH/10/2-1)]
        # prev_root_rotation_g = trajectory_rot[int(TRAJECTORY_LENGTH/10/2-1)]
        # Point-6: different representations of root_position & root_rotation
        root_position_g = controller.future_pos[0]
        root_rotation_g = controller.future_rot[0]
        prev_root_position_g = controller.recorded_pos[-1]
        prev_root_rotation_g = controller.recorded_rot[-1]
        
        for i in range(JOINT_NUM):
            o = int(TRAJECTORY_LENGTH / 10 * 10)
            # X.l_i = Q_root_T * (P_i - P_root)
            pos_l = R.from_quat(prev_root_rotation_g).inv().apply(self.joint_positions_g[i] - prev_root_position_g)
            prv_l = R.from_quat(prev_root_rotation_g).inv().apply(self.joint_velocities_g[i])
            Xp[o + (JOINT_NUM*3*0) + i*3 + 0] = pos_l[0]
            Xp[o + (JOINT_NUM*3*0) + i*3 + 1] = pos_l[1]
            Xp[o + (JOINT_NUM*3*0) + i*3 + 2] = pos_l[2]
            Xp[o + (JOINT_NUM*3*1) + i*3 + 0] = prv_l[0]
            Xp[o + (JOINT_NUM*3*1) + i*3 + 1] = prv_l[1]
            Xp[o + (JOINT_NUM*3*1) + i*3 + 2] = prv_l[2]
        
        # endregion
        
        
        # 网络预测
        Yp = self.pfnn.predict(Xp, self.phase)  # 直接使用预计算权重
        # Point-4: Checks the mean value of Yp, sees that the joints are in human shape
        # Yp = self.pfnn.Ymean
        
        
        # region 处理网络输出

        # 输出
        # Y Output Dimension = 212
        # 1: 根关节x方向移动速度
        # 2: 根关节y方向移动速度
        # 3: 根关节旋转速度
        # 4: 相位值变化量
        # 5-8: 当前帧左右脚跟脚尖触地状况
        # 9-20：未来6帧根关节xz位置变化量
        # 21-32：未来6帧根关节xz方向变化量
        # 33-92：当前20个关节局部位置信息
        # 93-152：当前20个关节局部速度信息
        # 153-212：当前20个关节局部旋转信息，是全局旋转矩阵
        root_x_vel_g = Yp[0]
        root_y_vel_g = Yp[1]
        root_rot_vel_g = Yp[2]
        phase_delta = Yp[3]
        current_contacts = Yp[4:8]
        root_xz_pos_delta_6frame_g = Yp[8:20]
        root_xz_dir_delta_6frame_g = Yp[20:32]
        
        # Parse and calculate global transforms from output
        for i in range(JOINT_NUM):
            # P_i = Q_root * Y.l_i + P_root
            pos_g = R.from_quat(root_rotation_g).apply(Yp[OPOS + i*3: OPOS + i*3 + 3]) + root_position_g
            # Q_i = Q_root * Y.R_i
            rot_g = (R.from_quat(root_rotation_g) * R.from_quat([0, Yp[OROT + i*3 + 0], Yp[OROT + i*3 + 1], Yp[OROT + i*3 + 2]])).as_quat()
            vel_g = R.from_quat(root_rotation_g).apply(Yp[OVEL + i*3: OVEL + i*3 + 3])
            
            # Point-3: Checks the impact of velocity
            self.joint_positions_g[i] = ((self.joint_positions_g[i] + vel_g) + pos_g) * self.extra_joint_smooth
            # self.joint_positions_g[i] = (self.joint_positions_g[i] + pos_g) * self.extra_joint_smooth
            self.joint_rotations_g[i] = rot_g
            self.joint_velocities_g[i] = vel_g
        
        
        # Forward Kinematics
        counter = 0
        for i in range(JOINT_NUM_SITE):
            # Point-2: Sees the joints are too far away, but when divided by a large number, the positions are in place.
            if 'end' not in joint_name[i].lower():
                # if 'root' in joint_name[i].lower():
                    # joint_translation[i] = root_position_g
                # else:
                joint_translation[i] = self.joint_positions_g[counter] / 65.0
                joint_orientation[i] = self.joint_rotations_g[counter]
                counter += 1
            else:
                joint_name_parent = joint_name[i][:-4]
                for j in range(JOINT_NUM_SITE):
                    if joint_name[j] == joint_name_parent:
                        joint_translation[i] = joint_translation[j]
                        joint_orientation[i] = joint_orientation[j]
                        break
        
        # 如果是 self.phase = math.pi - self.phase, 会一直正反抽搐, 说明确实是起到用处的
        # Point-1: see phase is rotating the joints smoothly, so knows phase is working
        # Point-8: stand amount was incorrect and set to gait.walk instead of gait.stand
        stand_amount = math.pow(1.0 - trajectory_gait[int(TRAJECTORY_LENGTH/10/2)][0], 0.25)
        self.phase = (self.phase + (stand_amount * 0.9 + 0.1) * 2 * math.pi * phase_delta) % (2 * math.pi)
        print(f"stand_amount: {stand_amount}, phase: {self.phase}")
        
        # endregion
        
        controller.set_pos(root_position_g)
        controller.set_rot(root_rotation_g)
        
        return joint_name, joint_translation, joint_orientation
    
    # 你的其他代码,state matchine, motion matching, learning, etc.
    
    
# class Trajectory:
#     '''
#     positions: (N,3)的ndarray
#     directions: (N,3)的ndarray
#     rotations: (N,4)的ndarray
#     heights: (N,1)的ndarray
#     gaits: (N,6)的ndarray
#     '''
#     def __init__(self, target_dir=np.array([0,0,0]), target_vel=np.array([0,0,1]), width=25):
#         self.positions = np.zeros((WINDOW_SIZE, 3))
#         self.directions = np.zeros((WINDOW_SIZE, 3))
#         self.rotations = np.zeros((WINDOW_SIZE, 4))
#         self.heights = np.zeros((WINDOW_SIZE, 1))
        
#         # [walk, stand, a, b, c, d]
#         self.gaits = np.zeros((WINDOW_SIZE, 6))
        
#         self.width = width
        
#         self.target_dir = target_dir
#         self.target_vel = target_vel
        
#         self.reset([0,0])

#     def reset(self, position):
#         root_position = np.array([position[0], 0, position[1]])
#         root_rotation = R.identity().as_quat()
        
#         for i in range(WINDOW_SIZE):
#             self.positions[i] = root_position
#             self.rotations[i] = root_rotation
#             self.directions[i] = np.array([0,0,1])
#             self.heights[i] = root_position[1]
#             self.gaits[i] = np.array([0, 0, 0, 0, 0, 0])