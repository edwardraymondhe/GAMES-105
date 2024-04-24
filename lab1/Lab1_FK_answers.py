import numpy as np
from scipy.spatial.transform import Rotation as R

#region Helper functions
def get_unit_vector(vector):
    # 计算向量的模（长度）
    magnitude = np.linalg.norm(vector)
    # 如果向量的模为0，则无法计算单位向量，返回原向量
    if magnitude == 0:
        return vector
    else:
        # 除以向量的模得到单位向量
        return vector / magnitude

def get_vector_angle(a, b):
    return np.arccos(np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b)))

def get_rotation_matrix(originVector, T_location):
    # T_location 是目标向量
    # T_location = np.array((1.0 , 0 ,.0))
    
    T_location_norm = get_unit_vector(T_location)
	
    # originVector是原始向量
    # originVector = np.array((.0 , .0 ,1.0))
    
    # @是向量点乘
    sita = np.arccos(np.dot(T_location_norm,originVector))
    n_vector = np.cross(T_location_norm, originVector)
    
    n_vector = get_unit_vector(n_vector)
    
    n_vector_invert = np.matrix((
        [0,-n_vector[2],n_vector[1]],
        [n_vector[2],0,-n_vector[0]],
        [-n_vector[1],n_vector[0],0]
    ))
    
    I = np.matrix((
        [1 ,  0 , 0],
        [0 ,  1 , 0],
        [0 ,  0 , 1]
    ))

    R_w2c = I + np.sin(sita)*n_vector_invert + n_vector_invert@(n_vector_invert)*(1-np.cos(sita))
    
    return R_w2c

import numpy as np
 
def list_equal_within_tolerance(list1, list2):
    array1 = np.array(list1)
    array2 = np.array(list2)
    # 使用allclose方法判断两数组是否在指定精度内相等
    return np.allclose(array1, array2, atol=0.001)

#endregion

class TreeNode:
    def __init__(self, index, name, offset):
        self.index = index
        self.name = name
        self.children = []
        self.offset = offset
        self.rotation = None
        self.translation = None
        self.orientation = None
        self.position = None
        self.retarget = R.identity()

def fk(node):
    for i in range(len(node.children)):
        child = node.children[i]
        child.orientation = node.orientation * child.rotation
        child.position = node.position + node.orientation.apply(child.translation)
        fk(child)

def get_child_node_recursive(node, child_name):
    """
    循环遍历查找指定节点
    """
    if node.name == child_name:
        return node
    
    for i in range(len(node.children)):
        result = get_child_node_recursive(node.children[i], child_name)
        if result != None:
            return result

def get_child_node(tree_node, child_name):
    """
    找该节点的子节点
    """
    for i in range(len(tree_node.children)):
        child = tree_node.children[i]
        if child.name == child_name:
            return child
    return None

def create_tree(joint_name, joint_parent, joint_offset):
    """请填写以下内容
    输入: 关节名字，父节点列表，偏移量列表
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数
        frame_id: int，需要返回的帧的索引
    输出:
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
        2. from_euler时注意使用大写的XYZ
    """

    # 初始化一个树状结构
    tree_nodes = []
    for i in range(len(joint_name)):
        tree_node = TreeNode(i, joint_name[i], joint_offset[i])
        tree_nodes.append(tree_node)

    # 设置树状关系
    for i in range(1, len(tree_nodes)):
        parent_idx = joint_parent[i]
        tree_node = tree_nodes[i]
        tree_nodes[parent_idx].children.append(tree_node)

    return tree_nodes

def retarget_node(node_a_prev, node_a, node_b):
    """
    从A节点重定向至B节点
    """
    
    # 计算重定向时的局部旋转矩阵
    # r = unit_vector(np.cross(node_a.offset, node_b.offset))
    # if np.linalg.norm(r) != 0:
    #     print("!= norm")
    #     q = np.insert(r, 0, 0)
    #     m = R.from_quat(q)
    #     # 重定向偏移量
    #     node_a.retarget = m
    #     print(m.as_euler('XYZ'))

    # if node_a.name == "lShoulder":
    #     node_a.retarget = R.from_euler('XYZ',[0,0,45], degrees=True)
    # elif node_a.name == "rShoulder":
    #     node_a.retarget = R.from_euler('XYZ',[0,0,-45], degrees=True)
    # else:
    #     node_a.retarget = R.identity()

    if node_a_prev != None and len(node_a.children) == 1:
        # A 相对旋转计算
        node_a_next = node_a.children[0]
        node_a_dir_1 = get_unit_vector(node_a.offset)
        node_a_dir_2 = get_unit_vector(node_a_next.offset)
        if list_equal_within_tolerance(node_a_dir_1, node_a_dir_2) == False:
            relative_rotation_euler_a = R.from_matrix(get_rotation_matrix(node_a_dir_1, node_a_dir_2)).as_euler('XYZ',degrees=True).tolist()
        else:
            relative_rotation_euler_a = [0,0,0]

        # B 相对旋转计算
        node_b_next = node_b.children[0]
        node_b_dir_1 = get_unit_vector(node_b.offset)
        node_b_dir_2 = get_unit_vector(node_b_next.offset)
        relative_rotation_euler_b = R.from_matrix(get_rotation_matrix(node_b_dir_1, node_b_dir_2)).as_euler('XYZ',degrees=True).tolist()

        # print(f"{node_a.name}, {node_a_dir_1.tolist()}, {node_a_dir_2.tolist()}, {relative_rotation_euler_a}, {relative_rotation_euler_b}")

        if list_equal_within_tolerance(relative_rotation_euler_a, relative_rotation_euler_b) == False:
            print(f"{node_a.name}, Not equal")
            rotation_bias_matrix = get_rotation_matrix(node_b_dir_2, node_a_dir_2)
            node_a.retarget = R.from_matrix(rotation_bias_matrix)
        else:
            node_a.retarget = R.identity()
    

    # 递归计算
    for i in range(len(node_a.children)):
        retarget_node(node_a, node_a.children[i], get_child_node_recursive(node_b, node_a.children[i].name))

def retarget_tree(tree_a, tree_b):
    """
    从A树重定向至B树
    """
    retarget_node(None, tree_a[0], tree_b[0])


def set_motion_data(tree_nodes, motion_data, frame_id):
    non_leaf_counter = 0
    for i in range(0, len(tree_nodes)):
        tree_node = tree_nodes[i]

        # 非叶节点
        if len(tree_node.children) != 0:
            # ROOT
            if non_leaf_counter == 0:
                rotation = R.from_euler('XYZ', motion_data[frame_id][3:6], degrees=True)
                translation = np.array(motion_data[frame_id][0:3])

                tree_node.rotation = rotation
                tree_node.translation = translation
                
                tree_node.orientation = rotation
                tree_node.position = translation

            # JOINT
            else:
                rotation = R.from_euler('XYZ', motion_data[frame_id][3 + non_leaf_counter * 3 : 6 + non_leaf_counter * 3], degrees=True)
                translation = tree_node.offset
                
                tree_node.rotation = rotation
                tree_node.translation = translation
            
            non_leaf_counter += 1

        # 叶节点
        else:
            tree_node.rotation = R.identity()
            tree_node.translation = tree_node.offset

    return tree_nodes

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
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()

        joint_index = -1
        joint_stack = []
        curr = ""
        root = ""

        for i in range(len(lines)):
            line = lines[i].lstrip()
            
            if line.startswith('{'):
                # 添加父节点索引
                if curr == root:
                    # 如果是根节点
                    joint_parent.append(-1)
                else:
                    # 如果不是根节点
                    joint_parent.append(joint_stack[-1])

                joint_index += 1
                joint_stack.append(joint_index)

                # 添加当前节点名字
                joint_name.append(curr)
            
            elif line.startswith('}'):
                joint_stack.pop()

            elif line.startswith('JOINT'):
                curr = line.split()[1]
            elif line.startswith('ROOT'):
                curr = line.split()[1]
                root = curr
            elif line.startswith('End'):
                curr = curr + '_end'

            elif line.startswith('OFFSET'):
                splits = line.split()[1:]
                splits_float = [float(x) for x in splits]
                joint_offset.append(splits_float)
            else:
                continue

    joint_offset = np.array(joint_offset)

    return joint_name, joint_parent, joint_offset


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

    # 初始化一个树状结构
    tree_nodes = create_tree(joint_name, joint_parent, joint_offset)

    # 应用动作数据
    set_motion_data(tree_nodes, motion_data, frame_id)

    # 计算前向动力学
    fk(tree_nodes[0])

    joint_positions = np.array([x.position for x in tree_nodes])
    joint_orientations = np.array([x.orientation.as_quat() for x in tree_nodes])
    
    return joint_positions, joint_orientations


def part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    用 T-pose的骨骼模型，A-pose的运动数据（转换至T-pose）
    输入: 两个bvh文件的路径
    输出: 
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    Tips:
        两个bvh的joint name顺序可能不一致哦(
        as_euler时也需要大写的XYZ
    """

    t_joint_name, t_joint_parent, t_joint_offset = part1_calculate_T_pose(T_pose_bvh_path)
    t_tree = create_tree(t_joint_name, t_joint_parent, t_joint_offset)

    a_joint_name, a_joint_parent, a_joint_offset = part1_calculate_T_pose(A_pose_bvh_path)
    a_motion_data = load_motion_data(A_pose_bvh_path)
    a_tree = create_tree(a_joint_name, a_joint_parent, a_joint_offset)

    retarget_tree(a_tree, t_tree)

    motion_data = []

    for n in range(len(a_motion_data)):
        set_motion_data(a_tree, a_motion_data, n)
        
        temp = []
        
        for i in range(len(t_joint_name)):
            # 根据T骨骼的顺序，获得A骨骼中对应的Rotation
            node = get_child_node_recursive(a_tree[0], t_joint_name[i])
            # 跳过叶节点
            if len(node.children) != 0:
                euler = (node.rotation * node.retarget.inv()).as_euler('XYZ', degrees=True)
                if i == 0:
                    for p in node.position:
                        temp.append(p)
                for e in euler:
                    temp.append(e)

        motion_data.append(temp)

    # return motion_data
    return np.array(motion_data)