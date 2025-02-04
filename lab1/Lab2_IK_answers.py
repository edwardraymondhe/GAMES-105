import numpy as np
from scipy.spatial.transform import Rotation as R
from task2_inverse_kinematics import MetaData

import sys
sys.path.append("..//")
import utils

import torch

def tensor2numpy(data):
    return data.detach().numpy()

def tensor2quaternion(data):
    return R.from_matrix(data.detach().numpy()).as_quat()

def get_unit_matrix_tensor(matrix):
    # 计算均值和标准差
    mean = matrix.mean()
    std = matrix.std()
    # 标准化矩阵
    unit_matrix = (matrix - mean) / std
    return unit_matrix

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
    
    # With original joints
    # joint_positions, joint_orientations = ik_ccd_1(meta_data, joint_positions, joint_orientations, target_pose)
    
    joint_positions, joint_orientations = ik_jacobian(meta_data, joint_positions, joint_orientations, target_pose)
    
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
    iter_threshold = 1
    
    path_reverse = list(reversed(path))
    print(path_reverse)
    joint_list = utils.create_list(meta_data.joint_name, meta_data.joint_parent, joint_position=meta_data.joint_initial_position)
    joint_list[0].calculate_offset_by_position()
    
    end_idx = path_reverse[0]
    
    for _ in range(iter_threshold):
        
        fk_direction_i = True
        for i, curr_i_idx in enumerate(path_reverse[1:-1]):
            curr_i = joint_list[curr_i_idx]
            
            end_position = joint_list[end_idx].position
            
            if fk_direction_i:
                # Positive direction
                
                start_position = curr_i.position
                # 当前关节到末端的向量
                from_vec = end_position - start_position
                # 当前关节到目标位置的向量
                to_vec = target_pose - start_position
                
                # 计算距离，如果小于阈值，则停止迭代
                distance = np.linalg.norm(end_position - target_pose)
                if distance < distance_threshold:
                    break
                
                # 计算变换矩阵&旋转
                matrix_from2to = utils.get_rotation_matrix_by_vectors(from_vec, to_vec)
                q_from2to = R.from_matrix(matrix_from2to)
                
                # 存储老旋转
                curr_i.last_rotation = curr_i.rotation
                # 更新旋转
                curr_i.rotation = curr_i.last_rotation * q_from2to
            elif len(curr_i.children) == 1:
                # Negative direction, When get passes zero, flip the calculation
                
                # 当前关节的子关节
                start_position = curr_i.children[0].position
                # 当前关节的子关节到末端的向量
                from_vec = end_position - start_position
                # 当前关节的子关节到目标位置的向量
                to_vec = target_pose - start_position
                
                # 计算距离，如果小于阈值，则停止迭代
                distance = np.linalg.norm(end_position - target_pose)
                if distance < distance_threshold:
                    break
            
                # 计算变换矩阵&旋转
                matrix_from2to = utils.get_rotation_matrix_by_vectors(from_vec, to_vec)
                
                # 虽然迭代至当前关节，实则根据子关节的计算的转置，推导出当前关节的旋转
                q_from2to = R.from_matrix(matrix_from2to).inv()
                
                # 存储老旋转
                curr_i.children[0].last_rotation = curr_i.children[0].rotation
                # 更新旋转
                curr_i.children[0].rotation = curr_i.children[0].last_rotation * q_from2to
                    
            if curr_i.idx == 0:
                fk_direction_i = False
            
            # 根据关节i所更新的旋转，作FK
            fk_direction_j = False
            for j, curr_j_idx in enumerate(path[:-1]):
                curr_j = joint_list[curr_j_idx]
                
                if curr_j.idx == 0:
                    fk_direction_j = True
                
                if fk_direction_j and curr_j.idx != 0:
                    # Qi = Q_pi * R
                    # Pi = P_pi + Q_pi * l
                    # 从父关节推导出子关节的旋转位置
                    curr_j.orientation = curr_j.parent.orientation * curr_j.rotation
                    curr_j.position = curr_j.parent.position + curr_j.parent.orientation.apply(curr_j.translation)
                    
                elif len(curr_j.children) == 1:
                    # Q_pi = Qi * R(-1)
                    # P_pi = P_i - Q_pi * l
                    # 从子关节推导出父关节的旋转
                    parent = curr_j
                    child = curr_j.children[0]
                    parent.orientation = child.orientation * child.rotation.inv()
                    parent.position = child.position - parent.orientation.apply(child.translation)
                    
                    # 父关节为根关节
                    if curr_j.parent.idx == 0:
                        root = curr_j.parent
                        hip = curr_j
                        
                        root.orientation = hip.orientation * hip.rotation.inv() * root.rotation
                        root.position = hip.position + hip.orientation.apply(-hip.rotation.inv().apply(hip.translation))
                
                # 其他无需IK的关节作FK更新
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

def ik_jacobian(meta_data: MetaData, joint_positions: np.ndarray, joint_orientations: np.ndarray, target_pose: np.ndarray):
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
    # [F(0)] = J(T) * delta = [f(0)](T) * delta
    # target is to minimize "f(0) - x"
    # f(0) is end's position

    # Logic:
    # 1a. end's position is determined by its parent's position, parent's orientation, its offset
    # 1b. its parent's position and orientation are determined by its parent's position and orientation, and its offset
    # 2. therefore, once we have "required_grad=True", every operator applied on joint_rotations_t and joint_orientations_t can be tracked by pytorch (it's done by having a dag)
    # 3. after everything is being calculated & operated, can use pytorch's AutoGrad function
    # IMPORTANT: Only fk can get its recursive operators path, for the autograd to work in pytorch
    # 4a. target = (joint_positions_t[path[0]] - target_pose)
    # 4b. target.backward()
    # 4c. joint_orientations_t = joint_orientations_t - learning_rate * target.grad
    
    path, path_name, path1, path2 = meta_data.get_path_from_root_to_end()
    joint_name = meta_data.joint_name
    joint_parent = meta_data.joint_parent
    
    distance_threshold = 0.01
    
    # path_reverse = list(reversed(path))
    # [21, 19, 17, 15, 13, 2, 1, 0, 4, 6, 8, 10, 23]
    # ['lWrist_end', 'lWrist', 'lElbow', 'lShoulder', 'lTorso_Clavicle', 'lowerback_torso', 'pelvis_lowerback', 'RootJoint', 'lHip', 'lKnee', 'lAnkle', 'lToeJoint', 'lToeJoint_end']
    # print("Reverse")
    # print(path_reverse)
    # print("Normal")
    # print(path)
    # print(path_name)
    
    
    # Qi = Q_pi * R
    # Pi = P_pi + Q_pi * l
    # Q_pi = Qi * R(-1)
    # P_pi = P_i - Q_pi * l

    # - R = Q_pi(T) * Qi
    joint_rotations = [(R.from_quat(joint_orientations[joint_parent[i]]).inv() * R.from_quat(joint_orientations[i])).as_quat() if i != 0 else joint_orientations[0] for i in range(len(joint_orientations))]
    # - l = Q_pi(T) * (Pi - P_pi)
    joint_offsets = [R.from_quat(joint_orientations[joint_parent[i]]).inv().as_matrix() @ (joint_positions[i] - joint_positions[joint_parent[i]]) if i != 0 else joint_positions[0] for i in range(len(joint_positions))]
    
    # 将关节旋转和偏移转换为张量
    joint_orientations_t = [torch.tensor(R.from_quat(orientation).as_matrix(), requires_grad=True) for orientation in joint_orientations]
    joint_rotations_t = [torch.tensor(R.from_quat(rotation).as_matrix(), requires_grad=True) for rotation in joint_rotations]
    joint_positions_t = [torch.tensor(data) for data in joint_positions]
    joint_offsets_t = [torch.tensor(data) for data in joint_offsets]
    target_pos_t = torch.tensor(target_pose)
    
    iter_threshold = 500
    a = 0.01
    
    for _ in range(iter_threshold):
        
        # 在该路径上做所有的操作
        # 每个节点的tensor操作由上一个操作的tensor决定，因此有个DAG关系
        for i in range(len(path)):
            curr = path[i]
            # 根节点无需计算
            if i == 0:
                joint_orientations_t[curr] = joint_rotations_t[curr]
                joint_positions_t[curr] = joint_offsets_t[curr]
            else:
                prev = path[i-1]
                if prev == joint_parent[curr]:
                    # 上个节点是父节点
                    # ---   prev->curr    --->
                    parent = prev
                    child = curr
                    
                    # if fk_direction_j:
                    #     # Qi = Q_pi * R
                    #     # Pi = P_pi + Q_pi * l_i
                    joint_orientations_t[child] = joint_orientations_t[parent] @ joint_rotations_t[child]
                    joint_positions_t[child] = joint_positions_t[parent] + joint_orientations_t[parent] @ joint_offsets_t[child]
                else:
                    # 当前节点是父节点
                    # ---   prev<-curr    --->
                    parent = curr
                    child = prev
                    
                    # else:
                    #     # Q_pi = Qi * R(-1)
                    #     # P_pi = P_i - Q_pi * l_i
                    joint_orientations_t[parent] = joint_orientations_t[child] @ torch.transpose(joint_rotations_t[child],0,1)
                    joint_positions_t[parent] = joint_positions_t[child] - joint_orientations_t[parent] @ joint_offsets_t[child]
        
        # Avoids nan
        # 当前路径的末端节点是需要对齐的关节
        # target_function = torch.norm(joint_positions_t[path[-1]] - target_pos_t)
        target_function = 0.5 * torch.norm(joint_positions_t[path[-1]] - target_pos_t) ** 2
        # RuntimeError: grad can be implicitly created only for scalar outputs
        target_function.backward(torch.ones_like(target_function))
        
        for j in range(len(path)):
            if joint_rotations_t[j].grad != None:
                # r(i+1) = r(i) - a * grad(F(r(i))
                # grad(F(r(i)) = J(T) * δ
                # r(i+1) = r(i) - a * J(T) * δ
                # 这里没有显示地计算J(T)，而是直接用梯度来更新
                joint_rotations_t[j] = torch.tensor(joint_rotations_t[j] - a * joint_rotations_t[j].grad, requires_grad=True)
    
    joint_rotations = [tensor2quaternion(rotation_t) for rotation_t in joint_rotations_t]
        
    for i in range(len(path)):
        curr = path[i]
        if i == 0:
            joint_orientations[curr] = joint_rotations[curr]
            joint_positions[curr] = joint_offsets[curr]
        else:
            prev = path[i-1]
                
            if prev == joint_parent[curr]:
                # ---   prev->curr    --->
                parent = prev
                child = curr
                joint_orientations[child] = R.as_quat(R.from_quat(joint_orientations[parent]) * R.from_quat(joint_rotations[child]))
                joint_positions[child] = joint_positions[parent] + R.from_quat(joint_orientations[parent]).apply(joint_offsets_t[child])
            else:
                # ---   prev<-curr    --->
                parent = curr
                child = prev
                joint_orientations[parent] = R.as_quat(R.from_quat(joint_orientations[child]) * R.from_quat(joint_rotations[child]).inv())
                joint_positions[parent] = joint_positions[child] - R.from_quat(joint_orientations[parent]).apply(joint_offsets[child])
            # joint_positions_t[curr] = joint_positions_t[parent] + torch.tensor(R.from_quat(tensor2quaternion(joint_orientations_t[parent])).as_matrix()) @ joint_offsets_t[curr]
            
    # Make the rest to follow by fk
    for i in range(len(joint_name)):
        curr = i
        if i not in path:
            if i == 0:
                joint_orientations[curr] = joint_rotations[curr]
                joint_positions[curr] = joint_offsets[curr]
            else:
                parent = joint_parent[i]
                joint_orientations[curr] = R.as_quat(R.from_quat(joint_orientations[parent]) * R.from_quat(joint_rotations[curr]))
                joint_positions[curr] = joint_positions[parent] + R.from_quat(joint_orientations[parent]).apply(joint_offsets[curr])

    return joint_positions, joint_orientations

def part2_inverse_kinematics(meta_data, joint_positions, joint_orientations, relative_x, relative_z, target_height):
    """
    输入lWrist相对于RootJoint前进方向的xz偏移，以及目标高度，IK以外的部分与bvh一致
    """
    
    path, path_name, path1, path2 = meta_data.get_path_from_root_to_end()
    joint_name = meta_data.joint_name
    joint_parent = meta_data.joint_parent
    
    distance_threshold = 0.01
    
    # path_reverse = list(reversed(path))
    # [21, 19, 17, 15, 13, 2, 1, 0, 4, 6, 8, 10, 23]
    # ['lWrist_end', 'lWrist', 'lElbow', 'lShoulder', 'lTorso_Clavicle', 'lowerback_torso', 'pelvis_lowerback', 'RootJoint', 'lHip', 'lKnee', 'lAnkle', 'lToeJoint', 'lToeJoint_end']
    # print("Reverse")
    # print(path_reverse)
    # print("Normal")
    # print(path)
    # print(path_name)
     
    joint_rotations = [(R.from_quat(joint_orientations[joint_parent[i]]).inv() * R.from_quat(joint_orientations[i])).as_quat() if i != 0 else joint_orientations[0] for i in range(len(joint_orientations))]
    joint_offsets = [R.from_quat(joint_orientations[joint_parent[i]]).inv().as_matrix() @ (joint_positions[i] - joint_positions[joint_parent[i]]) if i != 0 else joint_positions[0] for i in range(len(joint_positions))]
    joint_orientations_t = [torch.tensor(R.from_quat(orientation).as_matrix(), requires_grad=True) for orientation in joint_orientations]
    joint_rotations_t = [torch.tensor(R.from_quat(rotation).as_matrix(), requires_grad=True) for rotation in joint_rotations]
    joint_positions_t = [torch.tensor(data) for data in joint_positions]
    joint_offsets_t = [torch.tensor(data) for data in joint_offsets]
    
    target_pos_t = torch.tensor([joint_positions[0][0] + relative_x, target_height, joint_positions[0][2] + relative_z])
    
    iter_threshold = 300
    learning_rate = 0.5
    
    root_parent = joint_parent[path[0]]
    
    for idx in path:
        joint_rotations_t[idx] = torch.tensor(R.from_quat([1.,0.,0.,0.]).as_matrix(), requires_grad=True, dtype=torch.float64)
    
    for _ in range(iter_threshold):
        
        for i in range(len(path)):
            # if fk_direction_j:
            #     # Qi = Q_pi * R
            #     # Pi = P_pi + Q_pi * l                    
            # else:
            #     # Q_pi = Qi * R(-1)
            #     # P_pi = P_i - Q_pi * l
            curr = path[i]
            if i == 0:
                joint_orientations_t[curr] = joint_rotations_t[curr]
                joint_positions_t[curr] = joint_positions_t[root_parent] + joint_orientations_t[root_parent] @ joint_offsets_t[curr]
            else:
                prev = path[i-1]
                
                if prev == joint_parent[curr]:
                    # ---   prev->curr    --->
                    parent = prev
                    child = curr
                    joint_orientations_t[child] = joint_orientations_t[parent] @ joint_rotations_t[child]
                    joint_positions_t[child] = joint_positions_t[parent] + joint_orientations_t[parent] @ joint_offsets_t[child]
                else:
                    # ---   prev<-curr    --->
                    parent = curr
                    child = prev
                    joint_orientations_t[parent] = joint_orientations_t[child] @ torch.transpose(joint_rotations_t[child],0,1)
                    joint_positions_t[parent] = joint_positions_t[child] - joint_orientations_t[parent] @ joint_offsets_t[child]
        
        # Avoids nan
        target_function = torch.norm(joint_positions_t[path[-1]] - target_pos_t)
        # RuntimeError: grad can be implicitly created only for scalar outputs
        target_function.backward(torch.ones_like(target_function))
        
        for j in path:
            if joint_rotations_t[j].grad != None:
                joint_rotations_t[j] = torch.tensor(joint_rotations_t[j] - learning_rate * joint_rotations_t[j].grad, requires_grad=True)
    
    joint_rotations = [tensor2quaternion(rotation_t) for rotation_t in joint_rotations_t]
        
    # for i in range(len(path)):
    #     curr = path[i]
    #     if i == 0:
    #         joint_orientations[curr] = joint_rotations[curr]
    #         joint_positions[curr] = joint_positions[root_parent] + R.from_quat(joint_orientations[root_parent]).apply(joint_offsets[curr])
    #     else:
    #         prev = path[i-1]
                
    #         if prev == joint_parent[curr]:
    #             # ---   prev->curr    --->
    #             parent = prev
    #             child = curr
    #             joint_orientations[child] = R.as_quat(R.from_quat(joint_orientations[parent]) * R.from_quat(joint_rotations[child]))
    #             joint_positions[child] = joint_positions[parent] + R.from_quat(joint_orientations[parent]).apply(joint_offsets_t[child])
    #         else:
    #             # ---   prev<-curr    --->
    #             parent = curr
    #             child = prev
    #             joint_orientations[parent] = R.as_quat(R.from_quat(joint_orientations[child]) * R.from_quat(joint_rotations[child]).inv())
    #             joint_positions[parent] = joint_positions[child] - R.from_quat(joint_orientations[parent]).apply(joint_offsets[child])
    #         # joint_positions_t[curr] = joint_positions_t[parent] + torch.tensor(R.from_quat(tensor2quaternion(joint_orientations_t[parent])).as_matrix()) @ joint_offsets_t[curr]
            
    # Make the rest to follow by fk
    for i in range(len(joint_name)):
        curr = i
        if i == 0:
            joint_orientations[curr] = joint_rotations[curr]
            joint_positions[curr] = joint_offsets[curr]
        else:
            parent = joint_parent[i]
            if i == path[0]:
                joint_orientations[curr] = joint_rotations[curr]
            else:
                joint_orientations[curr] = R.as_quat(R.from_quat(joint_orientations[parent]) * R.from_quat(joint_rotations[curr]))
            joint_positions[curr] = joint_positions[parent] + R.from_quat(joint_orientations[parent]).apply(joint_offsets[curr])
    
    
    return joint_positions, joint_orientations

def bonus_inverse_kinematics(meta_data, joint_positions, joint_orientations, left_target_pose, right_target_pose):
    """
    输入左手和右手的目标位置，固定左脚，完成函数，计算逆运动学
    """
    
    return joint_positions, joint_orientations