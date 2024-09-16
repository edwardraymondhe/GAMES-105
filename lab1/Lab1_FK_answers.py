import numpy as np
from scipy.spatial.transform import Rotation as R

class Joint:
    def __init__(self, idx, name, translation):
        self.idx = idx
        self.name = name
        self.translation = translation
        self.position = None
        self.rotation = R.identity()
        self.orientation = R.identity()
        self.children = []
        self.parent = None
        
    def log(self):
        print(self.name)
        for i in range(len(self.children)):
            self.children[i].log()
            
    def fk(self):
        # todo: calculate
        # todo: iterate
                    
        # Non-root 
        if self.parent != None:
            # Qi = Q_pi * R
            # Pi = P_pi + Q_i * l
            self.orientation = self.parent.orientation * self.rotation
            self.position = self.parent.position + self.parent.orientation.apply(self.translation)
        else:        
            # Root
            self.orientation = self.rotation
            self.position = self.translation

        for i in range(len(self.children)):
            self.children[i].fk()

joint_list = []

def load_motion_data(bvh_file_path):
    """part2 辅助函数，读取bvh文件"""
    with open(bvh_file_path, 'r') as f:
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



def part1_calculate_T_pose(bvh_file_path):
    global joint_list
    """请填写以下内容
    输入： bvh 文件路径
    输出:
        joint_name: List[str]，字符串列表，包含着所有关节的名字
        joint_parent: List[int]，整数列表，包含着所有关节的父关节的索引,根节点的父关节索引为-1
        joint_offset: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的偏移量

    Tips:
        joint_name顺序应该和bvh一致
    """
    joint_name = []
    joint_parent = []
    joint_offset_arr = []
    joint_offset = None
    
    joint_stack = [-1]
    joint_idx = -1
    # 打开文件
    with open(bvh_file_path, 'r') as file:
        line = file.readline()  # 读取第一行
        while line:  # 当行不为空时继续读取
            line = line.strip()  # 去掉行尾的换行符
            elements = line.split()
            prefix = elements[0]
            
            if prefix == "JOINT" or prefix == "ROOT":
                joint_name.append(elements[1])
            elif prefix == "End":
                joint_name.append(joint_name[-1]+"_end")
            elif prefix == "OFFSET":
                joint_offset_arr.append([float(elements[1]), float(elements[2]), float(elements[3])])
            elif prefix == "{":
                joint_parent.append(joint_stack[-1])
                joint_idx += 1
                joint_stack.append(joint_idx)
            elif prefix == "}":
                joint_stack.pop()

            line = file.readline()

    joint_offset = np.array(joint_offset_arr)
    
    joint_list = []
    for i in range(0, len(joint_parent)):
        pi = joint_parent[i]

        child = Joint(i, joint_name[i], joint_offset[i])
        joint_list.append(child)
        
        if i != 0:
            parent = joint_list[pi]
            parent.children.append(child)
        else:
            parent = None
            
        child.parent = parent
        
    return joint_name, joint_parent, joint_offset


def part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, frame_id):
    global joint_list
    """请填写以下内容
    输入: part1 获得的关节名字，父节点列表，偏移量列表
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数
        frame_id: int，需要返回的帧的索引
    输出:
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
        2. from_euler时注意使用大写的XYZ
    """
    motion_data_curr = motion_data[frame_id]
    joint_len = len(joint_name)
    joint_positions = []
    joint_orientations = []
    
    # Parse Root translation
    joint_list[0].translation = motion_data_curr[0:3]

    # Parse Joint rotations
    read_idx = 0
    for curr_idx in range(0, joint_len):
        curr = joint_list[curr_idx]
        r_r = R.identity()
        
        if len(curr.children) != 0:
            r_euler = motion_data_curr[3 + (read_idx * 3) : 3 + (read_idx * 3 + 3)]
            r_r = R.from_euler('XYZ', r_euler, degrees=True)
            read_idx += 1
        
        curr.rotation = r_r
        
    # Recursive fk() from Root
    joint_list[0].fk()
       
    for i in range(joint_len):
        joint_positions.append(joint_list[i].position)
        joint_orientations.append(joint_list[i].orientation.as_quat())
    return np.array(joint_positions), np.array(joint_orientations)


def part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出: 
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    Tips:
        两个bvh的joint name顺序可能不一致哦(
        as_euler时也需要大写的XYZ
    """
    motion_data = None
    return motion_data
