import numpy as np
from scipy.spatial.transform import Rotation as R

class Joint:
    def __init__(self, idx, name):
        self.idx = idx
        self.name = name
        self.translation = None
        self.position = None
        self.rotation = R.identity()
        self.orientation = R.identity()
        self.children = []
        self.parent = None
        
        self.retarget = R.identity()
        
    def log(self):
        print(self.name)
        for i in range(len(self.children)):
            self.children[i].log()
            
    def fk_by_offset(self):
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
            self.children[i].fk_by_offset()
            
    def fk_by_orientation(self):
        # Non-root 
        if self.parent != None:
            # Qi = Q_pi * R
            # Pi = P_pi + Q_i * l
            self.rotation = self.parent.orientation.inv() * self.orientation
            self.position = self.parent.position + self.parent.orientation.apply(self.translation)
        else:        
            # Root
            self.rotation = self.orientation
            self.position = self.translation

        for i in range(len(self.children)):
            self.children[i].fk_by_orientation()
            
def create_list(joint_name, joint_parent, joint_offset = None, joint_position = None):
    joint_list = []
    for i in range(0, len(joint_parent)):
        pi = joint_parent[i]

        child = Joint(i, joint_name[i])
        if (joint_offset != None):
            child.translation = joint_offset[i]
        if (joint_position != None):
            child.position = joint_position[i]
            
        joint_list.append(child)
        
        if i != 0:
            parent = joint_list[pi]
            parent.children.append(child)
        else:
            parent = None
            
        child.parent = parent
        
    return joint_list

def get_joint_by_name(joint_list, joint_name):
    for i in range(len(joint_list)):
        if joint_list[i].name == joint_name:
            return joint_list[i]
            
def retarget_to_pose(joint_origin, joint_target):
    print(f"Retargeting {joint_origin.name} \n{joint_origin.orientation.as_matrix()}\n{joint_target.orientation.as_matrix()}")

    # Retarget to pose
    if joint_origin.parent != None:
        q_retarget = joint_target.orientation.inv() * joint_origin.parent.retarget * joint_origin.orientation
    else:
        q_retarget = joint_target.orientation.inv() * joint_origin.orientation
        
    joint_origin.retarget = q_retarget
    
    joint_origin_children = joint_origin.children
    for i in range(len(joint_origin_children)):
        retarget_to_pose(joint_origin_children[i], get_joint_by_name(joint_origin_children, joint_origin_children[i].name))
        
def parse_motion_data(joint_list, motion_data_curr, frame_id):
    # Parse Root translation
    joint_list[0].translation = motion_data_curr[0:3]

    # Parse Joint rotations
    read_idx = 0
    for curr_idx in range(0, len(joint_list)):
        curr = joint_list[curr_idx]
        r_r = R.identity()
        
        if len(curr.children) != 0:
            r_euler = motion_data_curr[3 + (read_idx * 3) : 3 + (read_idx * 3 + 3)]
            r_r = R.from_euler('XYZ', r_euler, degrees=True)
            read_idx += 1
        
        curr.rotation = r_r

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

def get_unit_vector(vec):
    magnitude = np.linalg.norm(vec)
    if magnitude == 0:
        return vec
    else:
        return vec / magnitude

def get_angle_between_unit_vectors(a, b):
    dot_product = np.dot(a, b)
    return np.arccos(dot_product)

def get_rotation_matrix(a, b):
    a = get_unit_vector(a)
    b = get_unit_vector(b)
    u = get_unit_vector(np.cross(a, b))
    
    theta = get_angle_between_unit_vectors(a, b)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    matrix = np.array(
            [[0, -u[2], u[1]],
            [u[2], 0, -u[0]],
            [-u[1], u[0], 0]]
        )
    
    r = R.identity().as_matrix() + sin_theta * matrix + (1 - cos_theta) * matrix * matrix
    
    return r

def list_equal_within_tolerance(list1, list2):
    array1 = np.array(list1)
    array2 = np.array(list2)
    # 使用allclose方法判断两数组是否在指定精度内相等
    return np.allclose(array1, array2, atol=0.001)