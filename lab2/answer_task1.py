import numpy as np
import copy
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

import sys
sys.path.append("..//")
import utils

# ------------- lab1里的代码 -------------#
def load_meta_data(bvh_path):
    with open(bvh_path, 'r') as f:
        channels = []
        joints = []
        joint_parents = []
        joint_offsets = []
        end_sites = []

        parent_stack = [None]
        for line in f:
            if 'ROOT' in line or 'JOINT' in line:
                joints.append(line.split()[-1])
                joint_parents.append(parent_stack[-1])
                channels.append('')
                joint_offsets.append([0, 0, 0])

            elif 'End Site' in line:
                end_sites.append(len(joints))
                joints.append(parent_stack[-1] + '_end')
                joint_parents.append(parent_stack[-1])
                channels.append('')
                joint_offsets.append([0, 0, 0])

            elif '{' in line:
                parent_stack.append(joints[-1])

            elif '}' in line:
                parent_stack.pop()

            elif 'OFFSET' in line:
                joint_offsets[-1] = np.array([float(x) for x in line.split()[-3:]]).reshape(1,3)

            elif 'CHANNELS' in line:
                trans_order = []
                rot_order = []
                for token in line.split():
                    if 'position' in token:
                        trans_order.append(token[0])

                    if 'rotation' in token:
                        rot_order.append(token[0])

                channels[-1] = ''.join(trans_order)+ ''.join(rot_order)

            elif 'Frame Time:' in line:
                break
        
    joint_parents = [-1]+ [joints.index(i) for i in joint_parents[1:]]
    channels = [len(i) for i in channels]
    return joints, joint_parents, channels, joint_offsets

def load_motion_data(bvh_path):
    with open(bvh_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Frame Time'):
                break
        motion_data = []
        for line in lines[i+1:]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1,-1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data

# ------------- 实现一个简易的BVH对象，进行数据处理 -------------#

'''
注释里统一N表示帧数，M表示关节数
position, rotation表示局部平移和旋转
translation, orientation表示全局平移和旋转
'''

class BVHMotion():
    def __init__(self, bvh_file_name = None) -> None:
        
        # 一些 meta data
        self.joint_name = []
        self.joint_channel = []
        self.joint_parent = []
        
        # 一些local数据, 对应bvh里的channel, XYZposition和 XYZrotation
        #! 这里我们把没有XYZ position的joint的position设置为offset, 从而进行统一
        self.joint_position = None # (N,M,3) 的ndarray, 局部平移
        self.joint_rotation = None # (N,M,4)的ndarray, 用四元数表示的局部旋转
        
        if bvh_file_name is not None:
            self.load_motion(bvh_file_name)
        pass
    
    #------------------- 一些辅助函数 ------------------- #
    def load_motion(self, bvh_file_path):
        '''
            读取bvh文件，初始化元数据和局部数据
        '''
        self.joint_name, self.joint_parent, self.joint_channel, joint_offset = \
            load_meta_data(bvh_file_path)
        
        motion_data = load_motion_data(bvh_file_path)

        # 把motion_data里的数据分配到joint_position和joint_rotation里
        self.joint_position = np.zeros((motion_data.shape[0], len(self.joint_name), 3))
        self.joint_rotation = np.zeros((motion_data.shape[0], len(self.joint_name), 4))
        self.joint_rotation[:,:,3] = 1.0 # 四元数的w分量默认为1
        
        # meta_data is the initial setup
        # motion_data is the current pose
        cur_channel = 0
        for i in range(len(self.joint_name)):
            if self.joint_channel[i] == 0:
                # if no info exist, set position = offset
                self.joint_position[:,i,:] = joint_offset[i].reshape(1,3)
                continue   
            elif self.joint_channel[i] == 3:
                # if no position exist, set position = offset
                self.joint_position[:,i,:] = joint_offset[i].reshape(1,3)
                rotation = motion_data[:, cur_channel:cur_channel+3]
            elif self.joint_channel[i] == 6:
                # if position exists, set position correspondingly
                self.joint_position[:, i, :] = motion_data[:, cur_channel:cur_channel+3]
                rotation = motion_data[:, cur_channel+3:cur_channel+6]
            # all joints should have rotation info in bvh file
            self.joint_rotation[:, i, :] = R.from_euler('XYZ', rotation,degrees=True).as_quat()
            cur_channel += self.joint_channel[i]
        
        return

    def batch_forward_kinematics(self, joint_position = None, joint_rotation = None):
        '''
        利用自身的metadata进行批量前向运动学
        joint_position: (N,M,3)的ndarray, 局部平移
        joint_rotation: (N,M,4)的ndarray, 用四元数表示的局部旋转
        '''
        if joint_position is None:
            joint_position = self.joint_position
        if joint_rotation is None:
            joint_rotation = self.joint_rotation
        
        joint_translation = np.zeros_like(joint_position)
        joint_orientation = np.zeros_like(joint_rotation)
        joint_orientation[:,:,3] = 1.0 # 四元数的w分量默认为1
        
        # 一个小hack是root joint的parent是-1, 对应最后一个关节
        # 计算根节点时最后一个关节还未被计算，刚好是0偏移和单位朝向
        
        for i in range(len(self.joint_name)):
            pi = self.joint_parent[i]
            parent_orientation = R.from_quat(joint_orientation[:,pi,:]) 
            joint_translation[:, i, :] = joint_translation[:, pi, :] + \
                parent_orientation.apply(joint_position[:, i, :])
            joint_orientation[:, i, :] = (parent_orientation * R.from_quat(joint_rotation[:, i, :])).as_quat()
        return joint_translation, joint_orientation
    
    
    def adjust_joint_name(self, target_joint_name):
        '''
        调整关节顺序为target_joint_name
        '''
        idx = [self.joint_name.index(joint_name) for joint_name in target_joint_name]
        idx_inv = [target_joint_name.index(joint_name) for joint_name in self.joint_name]
        self.joint_name = [self.joint_name[i] for i in idx]
        self.joint_parent = [idx_inv[self.joint_parent[i]] for i in idx]
        self.joint_parent[0] = -1
        self.joint_channel = [self.joint_channel[i] for i in idx]
        self.joint_position = self.joint_position[:,idx,:]
        self.joint_rotation = self.joint_rotation[:,idx,:]
        pass
    
    def raw_copy(self):
        '''
        返回一个拷贝
        '''
        return copy.deepcopy(self)
    
    @property
    def motion_length(self):
        return self.joint_position.shape[0]
    
    
    def sub_sequence(self, start, end):
        '''
        返回一个子序列
        start: 开始帧
        end: 结束帧
        '''
        res = self.raw_copy()
        res.joint_position = res.joint_position[start:end,:,:]
        res.joint_rotation = res.joint_rotation[start:end,:,:]
        return res
    
    def append(self, other):
        '''
        在末尾添加另一个动作
        '''
        other = other.raw_copy()
        other.adjust_joint_name(self.joint_name)
        self.joint_position = np.concatenate((self.joint_position, other.joint_position), axis=0)
        self.joint_rotation = np.concatenate((self.joint_rotation, other.joint_rotation), axis=0)
        pass
    
    #--------------------- 你的任务 -------------------- #
    
    def decompose_rotation_with_yaxis(self, rotation):
        '''
        输入: rotation 形状为(4,)的ndarray, 四元数旋转
        输出: Ry, Rxz，分别为绕y轴的旋转和转轴在xz平面的旋转，并满足R = Ry * Rxz
        '''
        Ry = np.zeros_like(rotation)
        Rxz = np.zeros_like(rotation)
        
        # convert rotation from quat -> matrix
        # use matrix's y direction to create R'(rotation-matrix)
        # then 1. Ry = R'R, 2. Rxz = Ry(T)R
        
        # Method 1, original rotation decomposition
        # !- ARTIFACTS CREATED BY SRC_VEC, 是[:, 1], 不是[1]
        # !- 构造旋转矩阵 R'
        src_rotation = R.from_quat(rotation)
        src_rotation_matrix = src_rotation.as_matrix()
        src_y_vec = src_rotation_matrix[:, 1]
        dst_y_vec = [0, 1, 0]
        dif_rotation_matrix = utils.get_rotation_matrix_by_vectors(src_y_vec, dst_y_vec)
        dif_rotation = R.from_matrix(dif_rotation_matrix)
        # !- 应用旋转矩阵 R'
        Ry_rotation = dif_rotation * src_rotation
        Rxz_rotation = Ry_rotation.inv() * src_rotation
        Ry = Ry_rotation.as_quat()
        Rxz = Rxz_rotation.as_quat()
        
        # Method 2
        # 在XYZ欧拉角序列中，Y轴旋转具有特殊性质
        # Y轴旋转对应着水平面（xz平面）上的转向，这个旋转是独立的
        # 当我们使用XYZ顺序时，Y轴的旋转不会被X和Z的旋转"污染"
        # 任何3D旋转都可以分解为：绕Y轴的旋转（偏航yaw）+ 其他旋转
        # 这种分解是唯一的，因为Y轴旋转直接对应水平面上的方向变化
        # 其他两个轴（X和Z）的旋转组合则负责处理剩余的姿态变化
        
        # 这就是为什么我们可以：
        # 可以直接提取Y轴旋转
        # Ry = R.from_euler("XYZ", [0, euler_angles[1], 0], degrees=True)
        # 但不能：
        # 不能直接提取X或Z轴旋转
        # Rx = R.from_euler("XYZ", [euler_angles[0], 0, 0], degrees=True)  # 错误
        # Rz = R.from_euler("XYZ", [0, 0, euler_angles[2]], degrees=True)  # 错误
        
        # 这种特性在角色动画中特别有用，因为：
        # 角色的转向主要是通过Y轴旋转来控制的
        # 我们经常需要调整角色的朝向而不影响其他姿态
        # 这种分解方式符合人类直觉的运动理解
        
        # Ry = R.from_quat(rotation).as_euler("XYZ", degrees=True)
        # Ry = R.from_euler("XYZ", [0, Ry[1], 0], degrees=True)
        # Rxz = Ry.inv() * R.from_quat(rotation)
        # Ry = Ry.as_quat()
        # Rxz = Rxz.as_quat()
        
        return Ry, Rxz
    
    # part 1
    def translation_and_rotation(self, frame_num, target_translation_xz, target_facing_direction_xz):
        '''
        计算出新的joint_position和joint_rotation
        使第frame_num帧的根节点平移为target_translation_xz, 水平面朝向为target_facing_direction_xz
        frame_num: int
        target_translation_xz: (2,)的ndarray
        target_faceing_direction_xz: (2,)的ndarray，表示水平朝向。你可以理解为原本的z轴被旋转到这个方向。
        Tips:
            主要是调整root节点的joint_position和joint_rotation
            frame_num可能是负数，遵循python的索引规则
            你需要完成并使用decompose_rotation_with_yaxis
            输入的target_facing_direction_xz的norm不一定是1  
        '''
        # NOTE:
        # 1. R_0 is the key, make use of angle-axis instead of creating a matrix from imagination. Follow rodrigue's equation!
        # 2. Follow slide's equation, don't have do forward-kinematics by yourself (even though you could)
        
        res = self.raw_copy() # 拷贝一份，不要修改原始数据
        
        # Method 1, matrix, wrong
        # target_facing_direction_z = utils.get_unit_vector([target_facing_direction_xz[0], 0, target_facing_direction_xz[1]])
        # target_facing_direction_x = utils.get_unit_vector(np.cross([0,1,0], target_facing_direction_z))
        # target_facing_direction_matrix = [target_facing_direction_x, 
        #                                 [0,1,0], 
        #                                 target_facing_direction_z]
        # r_0 = R.from_matrix(target_facing_direction_matrix)
        
        # Method 2, calculate angle with R.from_euler("Y"), correct 
        # sin_theta_xz = np.cross(target_facing_direction_xz, np.array([0, 1])) / np.linalg.norm(target_facing_direction_xz)
        # cos_theta_xz = np.dot(target_facing_direction_xz, np.array([0, 1])) / np.linalg.norm(target_facing_direction_xz)
        # theta = np.arccos(cos_theta_xz)
        # if sin_theta_xz < 0:
        #     theta = 2 * np.pi - theta
        # r_0 = R.from_euler("Y", theta, degrees=False)
            
        # !- 旋转
        # R_i = R_0 * R_1_t * R_1_i
        # R_0 is the original rotation (current/base rotaion when transitioning to a frame)
        # R_1_t is transpose of local rotation of transition frame
        # 第一项就是基础项
        # 第二项看成坐标系信息，第二和第三项相乘就是相对旋转偏移量
        
        # 这里的R_0和t_0就是提供的参数，target_trans_xz和target_facing_dir_xz
        # R_1其实是 joint_rotation[frame_num]
        # 
        # # r_y 是frame_num时刻的y轴旋转
        # r_y_t * r_i 是将r_i中的y轴旋转去掉
        # r_0 * r_y_t * r_i 是r_i从frame_num开始 在r_0坐标系下的旋转
        
        # Method 3, rotate around y-axis by theta, uses previous util rotation-matrix function, correct
        cos_theta_xz = np.dot(target_facing_direction_xz, np.array([0, 1])) / np.linalg.norm(target_facing_direction_xz)
        theta = np.arccos(cos_theta_xz)
        r_0 = R.from_matrix(utils.get_rotation_matrix_by_angle([0,1,0], theta))
        
        r_y, r_xz = res.decompose_rotation_with_yaxis(res.joint_rotation[frame_num, 0])
        res.joint_rotation[:, 0] = R.as_quat(r_0 *
                                             R.from_quat(r_y).inv() * 
                                             R.from_quat(res.joint_rotation[:, 0]))
        
        # !- 位移
        offset = target_translation_xz - res.joint_position[frame_num, 0, [0,2]]
        res.joint_position[:, 0, [0,2]] += offset
        
        # Method 2: Motion transitioning
        for i in range(len(res.joint_position)):
            t_0_i = res.joint_position[i,0]
            t_0 = res.joint_position[frame_num,0]
            
            y = res.joint_position[frame_num,0][1]
            
            # Set x,z coord
            tmp_1 = t_0_i - t_0
            
            # r_0 * r_1_t * (t_1_i - t_1) + t_0
            t_i = (r_0 * R.from_quat(r_y).inv()).apply(tmp_1) + t_0
            
            # Set y coord
            res.joint_position[i,0] = [t_i[0], y, t_i[2]]
        
        
        # Method 1: forward-kinematics
        # trajectory_position = np.empty_like(res.joint_position)
        # # i stands for frame_id
        # for i in range(len(res.joint_rotation)):
        #     if i == 0:
        #         trajectory_position[i,0] = res.joint_position[i,0]
        #     else:
        #         trajectory_position[i,0] = trajectory_position[i-1,0] + (r_0 * R.from_quat(Ry).inv()).apply(res.joint_position[i,0]-res.joint_position[i-1,0])
        # res.joint_position[:,0] = trajectory_position[:,0]
            
        return res

# part2
def blend_two_motions(bvh_motion1, bvh_motion2, alpha):
    '''
    blend两个bvh动作
    假设两个动作的帧数分别为n1, n2
    alpha: 0~1之间的浮点数组，形状为(n3,)
    返回的动作应该有n3帧，第i帧由(1-alpha[i]) * bvh_motion1[j] + alpha[i] * bvh_motion2[k]得到
    i均匀地遍历0~n3-1的同时，j和k应该均匀地遍历0~n1-1和0~n2-1
    
    '''
    # NOTE:
    # When dot product between 2 quaternions is < 0, it means their angle-in-between is > 180
    # 1. SLERP blends two values in a "shortest distance" fashion
    # 2. q = -q, they represent the same rotation
    # Therefore, we can flip EITHER quaternion and the cosine, to make them fall within the same half-circle (1), WITHOUT affecting the rotation (2)
    
    res = bvh_motion1.raw_copy()
    res.joint_position = np.zeros((len(alpha), res.joint_position.shape[1], res.joint_position.shape[2]))
    res.joint_rotation = np.zeros((len(alpha), res.joint_rotation.shape[1], res.joint_rotation.shape[2]))
    res.joint_rotation[...,3] = 1.0

    alpha_count = len(alpha)
    m1_count = len(bvh_motion1.joint_position)
    m2_count = len(bvh_motion2.joint_position)
    print(m1_count, m2_count, alpha_count)
    for i in range(alpha_count):
        j = min(int(i/alpha_count * (m1_count)), m1_count-1)
        k = min(int(i/alpha_count * (m2_count)), m2_count-1)
        
        res.joint_position[i] = (1 - alpha[i]) * bvh_motion1.joint_position[j] + alpha[i] * bvh_motion2.joint_position[k]
        for joint_idx in range(len(bvh_motion1.joint_rotation[0])):
            # 四元数的乘积不是直接相乘
            # 先把四元数变成 [w, v]
            # [  w1 * w2 + v1 dot v2,  w1 * v2 + w2 * v2 + v1 cross v2  ]
            # j1_quat_w, j1_quat_v = j1_quat[0], j1_quat[[1,2,3]]
            # j2_quat_w, j2_quat_v = j2_quat[0], j2_quat[[1,2,3]]
            
            j1_quat = bvh_motion1.joint_rotation[j][joint_idx]
            j2_quat = bvh_motion2.joint_rotation[k][joint_idx]
            
            # Method 1, quaternions
            cos_theta_q = np.dot(j1_quat,j2_quat)
            if cos_theta_q < 0:
                cos_theta_q = -cos_theta_q
                # j1_quat = -j1_quat
                j2_quat = -j2_quat
                
            theta_q = np.arccos(cos_theta_q)
            sin_theta_q = np.sin(theta_q)
            if np.sin(theta_q) != 0:
                r_1_q = (np.sin((1 - alpha[i]) * theta_q) / sin_theta_q) * j1_quat
                r_2_q = (np.sin(alpha[i] * theta_q) / sin_theta_q) * j2_quat
                r_q = r_1_q + r_2_q
            else:
                r_q = r_2_q
            res.joint_rotation[i][joint_idx] = r_q

            # Method 2, euler angles
            # j1_euler = R.from_quat(j1_quat).as_euler('XYZ', degrees=True)
            # j2_euler = R.from_quat(j2_quat).as_euler('XYZ', degrees=True)
            # u, theta_e = utils.get_axis_angle_between_vectors(j1_euler, j2_euler)
            # r_1_e = (np.sin((1 - alpha[i]) * theta_e) / np.sin(theta_e)) * j1_euler
            # r_2_e = (np.sin(alpha[i] * theta_e) / np.sin(theta_e)) * j2_euler
            # r_e = r_1_e + r_2_e
            # r_e = R.from_euler('XYZ', r_e, degrees=True).as_quat()
            # res.joint_rotation[i][joint_idx] = r_e
    
    return res

# part3
def build_loop_motion(bvh_motion):
    '''
    将bvh动作变为循环动作
    由于比较复杂,作为福利,不用自己实现
    (当然你也可以自己实现试一下)
    推荐阅读 https://theorangeduck.com/
    Creating Looping Animations from Motion Capture
    '''
    res = bvh_motion.raw_copy()
    
    from smooth_utils import build_loop_motion
    return build_loop_motion(res)

# part4
# !- 结合了 part1 和 part2
# part1负责拼接两段过渡点的旋转和位移
# part2负责混合动作
def concatenate_two_motions(bvh_motion1: BVHMotion, bvh_motion2: BVHMotion, mix_frame1: int, mix_time: int):
    '''
    将两个bvh动作平滑地连接起来，mix_time表示用于混合的帧数
    混合开始时间是第一个动作的第mix_frame1帧
    虽然某些混合方法可能不需要mix_time，但是为了保证接口一致，我们还是保留这个参数
    Tips:
        你可能需要用到BVHMotion.sub_sequence 和 BVHMotion.append
    '''
    res = bvh_motion1.raw_copy()
    
    # TODO: 你的代码
        
    # Start from mix_frame1, do interporlation between 1&2 according to mix_time
    # 78, 30
    print(mix_frame1, mix_time)
    # 182, 45
    print(len(res.joint_position), len(bvh_motion2.joint_position))
    # Means, from 78th frame in 1, start blend with 2
    # 30 means let last 30 in 1[30:] & first 30 in 2[:30] blend

    # Align bvh2 with bvh1's mixing frame
    bvh_motion2_ry, _ = bvh_motion2.decompose_rotation_with_yaxis(res.joint_rotation[mix_frame1,0])
    bvh_motion2_ry = utils.get_unit_vector([bvh_motion2_ry[1],bvh_motion2_ry[3]])
    # !- 将bvh2的旋转和位移对齐到bvh1的混合帧(mix_frame1)
    bvh_motion2 = bvh_motion2.translation_and_rotation(0, res.joint_position[mix_frame1,0,[0,2]], np.array(bvh_motion2_ry))
    
    joint_position = []
    joint_rotation = []
    
    for i in range(mix_time):
        t = i / mix_time
        
        # Position
        p_1 = bvh_motion1.joint_position[mix_frame1 + i]
        p_2 = bvh_motion2.joint_position[i]
        p = (1-t) * p_1 + t * p_2
        joint_position.append(p)
        
        # Rotation
        tmp = []
        for joint_idx in range(len(bvh_motion1.joint_rotation[0])):
            r_1 = bvh_motion1.joint_rotation[mix_frame1 + i][joint_idx]
            r_2 = bvh_motion2.joint_rotation[i][joint_idx]
            cos_theta_q = np.dot(r_1,r_2)
            if cos_theta_q < 0:
                cos_theta_q = -cos_theta_q
                r_1 = -r_1
                # r_1 = -r_1
            theta_q = np.arccos(cos_theta_q)
            sin_theta_q = np.sin(theta_q)
            if np.sin(theta_q) != 0:
                r_1_q = (np.sin((1 - t) * theta_q) / sin_theta_q) * r_1
                r_2_q = (np.sin(t * theta_q) / sin_theta_q) * r_2
                r_q = r_1_q + r_2_q
            else:
                r_q = r_2_q
            
            if np.linalg.norm(r_q) == 0:
                r_q = r_1
                print(r_q, bvh_motion1.joint_name[joint_idx], r_1)
            
            tmp.append(r_q)
        joint_rotation.append(tmp)
        
    
    res.joint_position = np.concatenate([res.joint_position[:mix_frame1], np.array(joint_position)], axis=0)
    res.joint_position = np.concatenate([res.joint_position, bvh_motion2.joint_position[mix_time:]], axis=0)
    
    res.joint_rotation = np.concatenate([res.joint_rotation[:mix_frame1], np.array(joint_rotation)], axis=0)
    res.joint_rotation = np.concatenate([res.joint_rotation, bvh_motion2.joint_rotation[mix_time:]], axis=0)
    
    return res

