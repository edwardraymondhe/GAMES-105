import numpy as np
from scipy.spatial.transform import Rotation as R
from Lab1_FK_answers import *

def rotation_matrix_reference(a, b):
    a=a/np.linalg.norm(a)
    b=b/np.linalg.norm(b)
    n = np.cross(a, b)
    # 旋转矩阵是正交矩阵，矩阵的每一行每一列的模，都为1；并且任意两个列向量或者任意两个行向量都是正交的。
    # n=n/np.linalg.norm(n)
    # 计算夹角
    cos_theta = np.dot(a, b)
    sin_theta = np.linalg.norm(n)
    theta = np.arctan2(sin_theta, cos_theta)
    # 构造旋转矩阵
    c = np.cos(theta)
    s = np.sin(theta)
    v = 1 - c
    rotation_matrix = np.array([[n[0]*n[0]*v+c, n[0]*n[1]*v-n[2]*s, n[0]*n[2]*v+n[1]*s],
                                 [n[0]*n[1]*v+n[2]*s, n[1]*n[1]*v+c, n[1]*n[2]*v-n[0]*s],
                                 [n[0]*n[2]*v-n[1]*s, n[1]*n[2]*v+n[0]*s, n[2]*n[2]*v+c]])
    return rotation_matrix


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
    # Fix naming issue
    target_pos = target_pose

    # Extract data from meta_data
    joint_name = meta_data.joint_name
    joint_parent = meta_data.joint_parent
    joint_initial_position = meta_data.joint_initial_position
    path, path_name, path1_raw, path2_raw = meta_data.get_path_from_root_to_end()
    
    # Root -> 腰 -> End
    # [0, 1, 2, 13, 15, 17, 19, 21]
    print(path)
    
    # ['RootJoint', 'pelvis_lowerback', 'lowerback_torso', 'lTorso_Clavicle', 'lShoulder', 'lElbow', 'lWrist', 'lWrist_end']
    print(path_name)

    # 手 -> 腰部上个节点
    # [21, 19, 17, 15, 13, 2, 1]
    # path1 = [item for item in path1_raw[::-1]]
    path1 = path1_raw.copy()
    path2 = path2_raw.copy()
    if path2_raw[-1] == 0:
        path1.append(0)
        path2.pop()

    print(path1)
    
    # 腰部
    # [0]
    print(path2)

    # 每个关节的世界坐标
    # print(joint_positions)

    # 每个关节的世界旋转
    # print(joint_orientations)

    # 计算关节在其父关节的局部旋转与局部坐标
    # FK:
    # for i in joint: (Root -> End) (Grand-grand-... parent -> Curr)
    #     p_i = i's parent joint
    #     Q_i = Q_pi * R_i (parent's orientation * joint's local rotation)
    #     x_i = x_pi + Q_pi * l_i (parent's position + parent's orientation * joint's local position)
    # Reverse FK:
    # for i in joint_reverse:
    #     p_i = current joint
    #     Q_i = current joint's orientation
    #     R_i = (parent's orientation)**-1 * Q_i
    #     l_i = (parent's orientation)**-1 * (current joint's position - parent's position)
    
    # Find 0 that minimizes [x - f(0) = 0]
    # Iterate 300 times
    
    for t in range(1):
        
        local_position = [(R.from_quat(joint_orientations[joint_parent[i]]).inv().apply(joint_positions[i] - joint_positions[joint_parent[i]])) for i in range(len(joint_name))]
        local_rotation = [(R.from_quat(joint_orientations[joint_parent[i]]).inv() * R.from_quat(joint_orientations[i])) for i in range(len(joint_name))]

        # Tweak from small route to large route
        # 1: Ignore end joint
        for j in range(0, len(path1) - 1):
            # Within this route, tweak each joint's orientation, for [joint -> end] to point to [joint -> x]
            start_path_idx = j
            end_path_idx = 0
            start_name = joint_name[path1[start_path_idx]]
            end_name = joint_name[path1[end_path_idx]]

            print(f"---- Route: {start_name} -> {end_name}")

            vector_curr_end = joint_positions[path1[end_path_idx]] - joint_positions[path1[start_path_idx]]
            vector_curr_target = target_pos                       - joint_positions[path1[start_path_idx]]

            target_end_rotation = R.from_matrix(get_rotation_matrix(vector_curr_end, vector_curr_target))

            for curr_path_idx in range(start_path_idx, end_path_idx-1, -1):
                curr_idx = path1[curr_path_idx]
                parent_idx = joint_parent[curr_idx]
                print(f"{joint_name[curr_idx]} <- {joint_name[parent_idx]}")

                # "*" operation in R = dot() method in ndarray matrix
                joint_orientations[curr_idx] = (target_end_rotation * R.from_quat(joint_orientations[curr_idx])).as_quat()
                joint_positions[curr_idx] = joint_positions[parent_idx] + R.from_quat(joint_orientations[parent_idx]).apply(local_position[curr_idx])

        # 获得除path以外的节点索引
        path_other = [x for x in range(len(joint_name)) if x not in path1]
        for curr_idx in path_other:
            parent_idx = joint_parent[curr_idx]
            target_orientation = (R.from_quat(joint_orientations[parent_idx]) * local_rotation[curr_idx]).as_quat()
            joint_orientations[curr_idx] = target_orientation
            joint_positions[curr_idx] = joint_positions[parent_idx] + R.from_quat(joint_orientations[parent_idx]).apply(local_position[curr_idx])

        end_pos = joint_positions[21]

        distance = end_pos - target_pos
        if np.linalg.norm(distance) <= 0.01:
            break

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