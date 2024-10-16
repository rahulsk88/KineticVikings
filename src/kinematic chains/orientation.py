import os

import json
import numpy as np



def load_json(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data ## I'll make this loop through but for now

def extract_joint_positions(json_data):
    positions = []
    for frame in json_data['tracking']:
        player_data = frame['data']['player']
        
        r_shoulder = np.array(player_data['R_SHOULDER']) ## TODO: Include an option to include this as a variable
        r_elbow = np.array(player_data['R_ELBOW'])
        r_wrist = np.array(player_data['R_WRIST'])

        positions.append((r_shoulder, r_elbow, r_wrist)) ## Should probably save this as a np.array but this works for now
    
    return positions

def calculate_limb_vectors(ith_position):
    r_shoulder, r_elbow, r_wrist = ith_position
    upper_arm_vector =  r_elbow - r_shoulder
    forearm_vector = r_wrist - r_elbow
    return upper_arm_vector, forearm_vector

def frame_to_rotation(positions):

    prev_upper_arm, prev_forearm = calculate_limb_vectors(positions[0])  #initing the prev variable
    prev_upper_arm, prev_forearm = prev_upper_arm / np.linalg.norm(prev_upper_arm), prev_forearm / np.linalg.norm(prev_forearm)

    rot_matrices = []
    for i in range(1, len(positions)):
        upper_arm, forearm = calculate_limb_vectors(positions[i])
        
        forearm = forearm / np.linalg.norm(forearm)
        upper_arm = upper_arm / np.linalg.norm(upper_arm)
        
        upper_arm_cross_prod = np.cross(prev_upper_arm, upper_arm )
        upper_arm_cos = np.dot(prev_upper_arm, upper_arm)
        upper_arm_sin = np.linalg.norm(upper_arm_cross_prod)

        upper_arm_kmat = np.array([[0, -upper_arm_cross_prod[2], upper_arm_cross_prod[1]], 
                        [upper_arm_cross_prod[2], 0, -upper_arm_cross_prod[0]], 
                        [-upper_arm_cross_prod[1], upper_arm_cross_prod[0], 0]])
        
        upper_arm_rotation_matrix = np.eye(3) + upper_arm_sin*upper_arm_kmat + (1-upper_arm_cos)*upper_arm_kmat.dot(upper_arm_kmat)

        forearm_cross_prod = np.cross(prev_forearm, forearm)
        forearm_cos = np.dot(prev_forearm, forearm)
        forearm_sin = np.linalg.norm(forearm_cross_prod)

        forearm_kmat = np.array([[0, -forearm_cross_prod[2], forearm_cross_prod[1]], 
                        [forearm_cross_prod[2], 0, -forearm_cross_prod[0]], 
                        [-forearm_cross_prod[1], forearm_cross_prod[0], 0]])
        
        forearm_rotation_matrix = np.eye(3) + forearm_sin*forearm_kmat + (1-forearm_cos)*forearm_kmat.dot(forearm_kmat)
        rot_matrices.append((upper_arm_rotation_matrix, forearm_rotation_matrix))

        prev_upper_arm, prev_forearm = upper_arm, forearm
        
    return rot_matrices

def json_to_rot_matrices(file_path):
    
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    positions = extract_joint_positions(data)
    rotation_matrices = frame_to_rotation(positions)
    
    return rotation_matrices

def convert_to_rot_matrices(folder_path):
    dict_output = {}

    for _, filename in enumerate(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, filename)
        dict_output[filename] = json_to_rot_matrices(file_path)
    
    return dict_output


    

