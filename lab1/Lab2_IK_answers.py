import numpy as np
from scipy.spatial.transform import Rotation as R

import sys
sys.path.append("..//")
import utils

def part1_inverse_kinematics(meta_data, joint_positions, joint_orientations, target_pose):
    """
    完成函数，计算逆运动学
    输入: 
        meta_data: 为了方便，将一些固定信息进行了打包，见上面的meta_data类
        joint_positions: 当前的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 当前的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
        target_pose: 目标位置，是一个numpy数组，shape为(3,)
    输出:
        经过IK后的姿态
        joint_positions: 计算得到的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 计算得到的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
    """
    
    joint_positions, joint_orientations = ik_ccd_1(meta_data, joint_positions, joint_orientations, target_pose)
    
    return joint_positions, joint_orientations

def ik_ccd_1(meta_data, joint_positions, joint_orientations, target_pose):
    """
    完成函数，计算逆运动学
    输入: 
        meta_data: 为了方便，将一些固定信息进行了打包，见上面的meta_data类
        joint_positions: 当前的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 当前的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
        target_pose: 目标位置，是一个numpy数组，shape为(3,)
    输出:
        经过IK后的姿态
        joint_positions: 计算得到的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 计算得到的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
    """
    
    path, path_name, path1, path2 = meta_data.get_path_from_root_to_end()
    
    distance_threshold = 0.01
    iter_threshold = 5
    
    path_reverse = list(reversed(path))
    print(path)
    print(path_name)
    # Path2 = start -> root
    # Path1 = end -> root-1
    joint_list = utils.create_list(meta_data.joint_name, meta_data.joint_parent, joint_position=meta_data.joint_initial_position)
    joint_list[0].calculate_offset_by_position()
    
    end_idx = path_reverse[0]
    
    for _ in range(iter_threshold):
        
        fk_direction_i = True
        for i, curr_i_idx in enumerate(path_reverse[1:-1]):
            # if i == 8:
            #     break
            
            curr_i = joint_list[curr_i_idx]
            next_i_idx = path_reverse[1:][i+1]
            
            end_position = joint_list[end_idx].position
            
            if fk_direction_i:
                # Positive direction
                start_position = curr_i.position
                from_vec = end_position - start_position
                to_vec = target_pose - start_position
                
                distance = np.linalg.norm(end_position - target_pose)
                if distance < distance_threshold:
                    break
            
                matrix = utils.get_rotation_matrix(from_vec, to_vec)
                rotation = R.from_matrix(matrix)
                
                curr_i.last_rotation = curr_i.rotation
                curr_i.rotation = curr_i.last_rotation * rotation
            else:            
                # Negative direction, When get passes zero, flip the calculation
                # TODO: How about reverse, from toe to wrist?
                if len(curr_i.children) == 1:
                    start_position = curr_i.children[0].position
                    from_vec = end_position - start_position
                    to_vec = target_pose - start_position
                    
                    distance = np.linalg.norm(end_position - target_pose)
                    if distance < distance_threshold:
                        break
                
                    matrix = utils.get_rotation_matrix(from_vec, to_vec)
                    rotation = R.from_matrix(matrix).inv()
                    
                    curr_i.children[0].last_rotation = curr_i.children[0].rotation
                    curr_i.children[0].rotation = curr_i.children[0].last_rotation * rotation
                    
            if curr_i.idx == 0:
                fk_direction_i = False
                    
            fk_direction_j = False
            for j, curr_j_idx in enumerate(path[:-1]):
                curr_j = joint_list[curr_j_idx]
                
                if curr_j.idx == 0:
                    fk_direction_j = True
                
                if fk_direction_j:
                    # Qi = Q_pi * R
                    # Pi = P_pi + Q_pi * l                    
                    if curr_j.idx != 0:
                        curr_j.orientation = curr_j.parent.orientation * curr_j.rotation
                        curr_j.position = curr_j.parent.position + curr_j.parent.orientation.apply(curr_j.translation)
                    else:
                        # Root
                        # curr_j.orientation = curr_j.orientation * (curr_j.last_rotation.inv() * curr_j.rotation)
                        # curr_j.orientation = curr_j.orientation
                        curr_j.orientation = curr_j.orientation * curr_j.rotation.inv()
                else:
                    # Q_pi = Qi * R(-1)
                    # P_pi = P_i - Q_pi * l
                    if curr_j.parent.idx != 0:
                        curr_j.parent.orientation = curr_j.orientation * curr_j.rotation.inv()
                        curr_j.parent.position = curr_j.position - curr_j.parent.orientation.apply(curr_j.translation)
                    else:
                        # Root
                        curr_j.parent.orientation = curr_j.orientation * curr_j.rotation.inv()
                        curr_j.parent.position = curr_j.position - curr_j.parent.orientation.apply(curr_j.translation)
                    
                if len(curr_j.children) > 1:
                    for un_ik_child in curr_j.children:
                        if un_ik_child.idx not in path:
                            joint_list[un_ik_child.idx].fk_by_offset()
                    
            last = joint_list[path[-1]]
            if last.parent != None:
                last.orientation = last.parent.orientation * last.rotation
                last.position = last.parent.position + last.parent.orientation.apply(last.translation) 
                
    joint_positions = []
    joint_orientations = []
    for i, curr_i in enumerate(joint_list):
        joint_positions.append(curr_i.position)
        joint_orientations.append(curr_i.orientation.as_quat())

    joint_positions = np.array(joint_positions)
    joint_orientations = np.array(joint_orientations)
        
    return joint_positions, joint_orientations

def part2_inverse_kinematics(meta_data, joint_positions, joint_orientations, relative_x, relative_z, target_height):
    """
    输入lWrist相对于RootJoint前进方向的xz偏移，以及目标高度，IK以外的部分与bvh一致
    """
    
    return joint_positions, joint_orientations

def bonus_inverse_kinematics(meta_data, joint_positions, joint_orientations, left_target_pose, right_target_pose):
    """
    输入左手和右手的目标位置，固定左脚，完成函数，计算逆运动学
    """
    
    return joint_positions, joint_orientations