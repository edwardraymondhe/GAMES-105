import numpy as np
from scipy.spatial.transform import Rotation as R

class TreeNode:
    def __init__(self, index, name):
        self.index = index
        self.name = name
        self.children = []
        self.rotation = None
        self.translation = None
        self.orientation = None
        self.position = None

def fk(tree_node):
    for i in range(len(tree_node.children)):
        child = tree_node.children[i]
        child.orientation = tree_node.orientation * child.rotation
        child.position = tree_node.position + tree_node.orientation.apply(child.translation)
        fk(child)

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
    tree_nodes = []
    for i in range(len(joint_name)):
        tree_node = TreeNode(i, joint_name[i])
        tree_nodes.append(tree_node)

    # 设置树状关系
    for i in range(1, len(tree_nodes)):
        parent_idx = joint_parent[i]
        tree_node = tree_nodes[i]
        tree_nodes[parent_idx].children.append(tree_node)

    # 设置节点的Rotation, Translation
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
                translation = joint_offset[i]
                
                tree_node.rotation = rotation
                tree_node.translation = translation
            
            non_leaf_counter += 1

        # 叶节点
        else:
            tree_node.rotation = R.identity()
            tree_node.translation = joint_offset[i]

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
    motion_data = None
    return motion_data
