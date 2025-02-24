import numpy as np
from scipy.spatial.transform import Rotation as R

import sys
sys.path.append("..//")
import utils

def part1_calculate_T_pose(bvh_file_path):
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
    joint_offset = []
    
    joint_stack = [-1]
    joint_idx = -1
    # 打开文件
    with open(bvh_file_path, 'r') as file:
        line = file.readline()  # 读取第一行
        while line:  # 当行不为空时继续读取
            line = line.strip()
            e = line.split()
            pre = e[0]

            if pre in ['JOINT','ROOT']:
                joint_name.append(e[1])
            elif pre == 'End':
                joint_name.append(joint_name[-1] + "_end")
            elif pre == 'OFFSET':
                joint_offset.append([float(e[1]), float(e[2]), float(e[3])])
            elif pre == '{':
                joint_parent.append(joint_stack[-1])
                joint_idx += 1
                joint_stack.append(joint_idx)
            elif pre == '}':
                joint_stack.pop()
            elif pre == 'MOTION':
                break
            
            line = file.readline()
            
    return joint_name, joint_parent, np.array(joint_offset)


def part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, frame_id):
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
    
    # Create joint list
    joint_list = utils.create_list(joint_name, joint_parent, joint_offset=joint_offset)
    
    # Get motion data for current frame
    motion_data_curr = motion_data[frame_id]
    joint_positions = []
    joint_orientations = []
    
    utils.parse_motion_data(joint_list, motion_data_curr, frame_id)
        
    # Recursive fk_by_offset() from Root
    joint_list[0].fk_by_offset()
       
    for i in range(len(joint_list)):
        joint_positions.append(joint_list[i].position)
        joint_orientations.append(joint_list[i].orientation.as_quat())
    return np.array(joint_positions), np.array(joint_orientations)


def part3_retarget_func_unofficial(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出: 
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    Tips:
        两个bvh的joint name顺序可能不一致哦(
        as_euler时也需要大写的XYZ
    """
    joint_name_t, joint_parent_t, joint_offset_t = part1_calculate_T_pose(T_pose_bvh_path)
    joint_name_a, joint_parent_a, joint_offset_a = part1_calculate_T_pose(A_pose_bvh_path)
    
    joint_list_t = utils.create_list(joint_name_t, joint_parent_t, joint_offset=joint_offset_t)
    joint_list_a = utils.create_list(joint_name_a, joint_parent_a, joint_offset=joint_offset_a)

    # 计算2个骨骼的全局旋转
    joint_list_t[0].fk_by_offset()
    joint_list_a[0].fk_by_offset()
    
    # Calculate retarget matrix
    for i in range(len(joint_list_t)):
        joint_t = joint_list_t[i]
        joint_a = utils.get_joint_by_name(joint_list_a, joint_t.name)
        joint_a.retarget = R.identity()
        if joint_a.idx != 0:
            if len(joint_a.children) == 1:
                joint_t_trans_dir = joint_t.translation
                joint_t_child_trans_dir = joint_t.children[0].translation
                joint_t_relative_matrix = utils.get_rotation_matrix_by_vectors(joint_t_trans_dir, joint_t_child_trans_dir)
                
                joint_a_trans_dir = joint_a.translation
                joint_a_child_trans_dir = joint_a.children[0].translation
                joint_a_relative_matrix = utils.get_rotation_matrix_by_vectors(joint_a_trans_dir, joint_a_child_trans_dir)
                
                if (not (joint_t_relative_matrix == joint_a_relative_matrix).all()):
                    retarget_rotation = R.from_matrix(joint_t_relative_matrix) * R.from_matrix(joint_a_relative_matrix).inv()
                    joint_a.retarget = retarget_rotation
        
    # Load raw data
    motion_data_a = utils.load_motion_data(A_pose_bvh_path)
    
    # Calculate retargeting
    # Output according to t's structure
    motion_data_t = []
    # Every frame
    for i in range(len(motion_data_a)):
        motion_data_t_curr = []
        motion_data_a_curr = motion_data_a[i]
        
        # Parse and recalculate
        utils.parse_motion_data(joint_list_a, motion_data_a_curr, i)
        joint_list_a[0].fk_by_offset()
        
        motion_data_t_curr.extend(motion_data_a_curr[0:3])
        
        for j in range(len(joint_list_t)):
            joint_t = joint_list_t[j]
            if len(joint_t.children) == 0:
                continue
            
            joint_a = utils.get_joint_by_name(joint_list_a, joint_t.name)
            rotation_a_t = (joint_a.rotation * joint_a.retarget.inv()).as_euler('XYZ', degrees=True)
            motion_data_t_curr.extend(rotation_a_t)

        motion_data_t.append(motion_data_t_curr)
    
    return np.array(motion_data_t)

def part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path):
    # FROM LECTURE
    joint_name_t, joint_parent_t, joint_offset_t = part1_calculate_T_pose(T_pose_bvh_path)
    joint_name_a, joint_parent_a, joint_offset_a = part1_calculate_T_pose(A_pose_bvh_path)
    
    joint_list_t = utils.create_list(joint_name_t, joint_parent_t, joint_offset=joint_offset_t)
    joint_list_a = utils.create_list(joint_name_a, joint_parent_a, joint_offset=joint_offset_a)

    # 计算2个骨骼的全局旋转
    joint_list_t[0].fk_by_offset()
    joint_list_a[0].fk_by_offset()
    
    # Calculate retarget matrix
    for i in range(len(joint_list_t)):
        joint_t = joint_list_t[i]
        joint_t.retarget = R.identity()
        joint_a = utils.get_joint_by_name(joint_list_a, joint_t.name)
        joint_a.retarget = R.identity()
        if joint_a.idx != 0:
            # 如果T和A的旋转不一致
            # 在每个BVH文件中 Rotation和Orientation都是在相应坐标系下的
            # 全局/局部旋转都是0 在TPose和APose中没有任何区别
            # 因此需要根据位置去计算Q_A2T
            if len(joint_a.children) == 1:
                dir_a = joint_a.children[0].position - joint_a.position
                dir_t = joint_t.children[0].position - joint_t.position
                q_i_a2t = R.from_matrix(utils.get_rotation_matrix_by_vectors(dir_a, dir_t))
                q_i_t2a = q_i_a2t.inv()
                joint_a.retarget = q_i_a2t
                joint_t.retarget = q_i_t2a
        
    # Load raw data
    motion_data_a = utils.load_motion_data(A_pose_bvh_path)
    
    # Calculate retargeting
    # Output according to t's structure
    motion_data_t = []
    # Every frame
    for i in range(len(motion_data_a)):
        motion_data_t_curr = []
        motion_data_a_curr = motion_data_a[i]
        
        # Parse and recalculate
        utils.parse_motion_data(joint_list_a, motion_data_a_curr, i)
        joint_list_a[0].fk_by_offset()
        
        motion_data_t_curr.extend(motion_data_a_curr[0:3])
        
        for j in range(len(joint_list_t)):
            joint_t = joint_list_t[j]
            if len(joint_t.children) == 0:
                continue
            
            joint_a = utils.get_joint_by_name(joint_list_a, joint_t.name)
            if joint_t.parent != None:
                rotation_t = joint_a.parent.retarget * joint_a.rotation * (joint_a.retarget).inv()
            else:
                rotation_t = R.identity() * joint_a.rotation * (joint_a.retarget).inv()
            motion_data_t_curr.extend(rotation_t.as_euler('XYZ', degrees=True))

        motion_data_t.append(motion_data_t_curr)
    
    return np.array(motion_data_t)