import math
import numpy as np
import torch
import torch.nn.functional as F

from models.cnn_multiframe import CNN_multiframe
from models.lstm import LSTM


ACTION_DICT = {
    0: 'walk',
    1: 'run',
    2: 'sit',
    3: 'greet (bow)',
    4: 'hug',
    5: 'falling down',
    6: 'Crossing arms',
    7: 'Jump',
    8: 'Climb stairs',
    9: 'Give directions',
    10: 'Sit-ups',
    11: 'Crawl',
    12: 'Taekwondo',
    13: 'Drinking a drink',
    14: 'Operating a cell phone',
    15: 'Talking on the phone',
    16: 'Carrying things',
    17: 'Smoking',
    18: 'Weightlifting (lifting objects)',
    19: 'Fencing (stabbing)'
}

ACTION_DICT_VER2 = {
    0 : 'walk',
    1 : 'run',
    2 : 'sit',
    3 : 'falling down',
    4 : 'walk',
    5 : 'crawl'
}

COCO_SKELETON = [
    [0,5], #nose-left_shoulder
    [0,6], #nose-right_shoulder
    [5,7], #left_shoulder-left_elbow
    [7,9], #left_elbow-left_wrist
    [6,8], #right_shoulder-right_elbow
    [8,10], #right_elbow-right_wrist
    [5,11], #left_shoulder-left_hip
    [6,12], #right_shoulder-right_hip
    [11,12], #left_hip-right_hip
    [11,13], #left_hip-left_knee
    [13,15], #left_knee-left_ankle
    [12,14], #right_hip-right_knee
    [14,16] #right_knee-right_ankle
]


def get_13_keypoint_from_coco(keypoint) :
        extracted_rows_1 = keypoint[:, 0, :]
        extracted_rows_1 = np.expand_dims(extracted_rows_1, axis=0)
        extracted_rows_2 = keypoint[:, 5:, :]

        new_keypoint = np.concatenate((extracted_rows_1, extracted_rows_2), axis=1)
        new_keypoint = torch.Tensor(new_keypoint)
        return new_keypoint


def subtract_vector(keypoint):
    skeleton_vector = np.zeros((1, len(COCO_SKELETON), 2))

    for i in range(len(COCO_SKELETON)):
        a_kpt, b_kpt = COCO_SKELETON[i][0], COCO_SKELETON[i][1]
        a_x, a_y = keypoint[a_kpt][0], keypoint[a_kpt][1]
        b_x, b_y = keypoint[b_kpt][0], keypoint[b_kpt][1]
        skeleton_vector[0][i] = np.array([a_x - b_x, a_y - b_y])

    return skeleton_vector


#keypoint 받아서 skeleton 행렬 반환
def lstm_subtract_vector(keypoint):
    skeleton_vector = np.zeros((1, 1, len(COCO_SKELETON)*2))
    vector_list = []
    for i in range(len(COCO_SKELETON)):
        a_kpt, b_kpt = COCO_SKELETON[i][0], COCO_SKELETON[i][1]
        a_x, a_y = keypoint[a_kpt][0], keypoint[a_kpt][1]
        b_x, b_y = keypoint[b_kpt][0], keypoint[b_kpt][1]
        vector_list.extend([a_x - b_x, a_y - b_y])
    skeleton_vector[0][0] = np.array(vector_list)  #(1, 1, 26)
    
    return skeleton_vector


def dotproduct(v1, v2):
    #a*c + b*d
    return sum((a*b) for a, b in zip(v1, v2))


def get_length(v):
    len = math.sqrt(dotproduct(v, v))
    return len


def get_angle(v1, v2):
    ang = dotproduct(v1, v2) / (get_length(v1) * get_length(v2))
    if ang < -1 :
        ang = -1
    elif ang > 1 :
        ang = 1
    
    angle = math.degrees(math.acos(ang))
    if np.isnan(angle) :
        angle = 0
    return angle


def get_cos_angle(keypoint):
    skeleton_vector = np.zeros((1, len(COCO_SKELETON), 3))
    for i in range(len(COCO_SKELETON)):
        a_kpt, b_kpt = COCO_SKELETON[i][0], COCO_SKELETON[i][1]
        a_x, a_y = keypoint[a_kpt][0], keypoint[a_kpt][1]
        b_x, b_y = keypoint[b_kpt][0], keypoint[b_kpt][1]
        angle = get_angle((a_x, a_y), (b_x, b_y))      
        skeleton_vector[0][i] = np.array([a_x-b_x, a_y-b_y, angle])       

    return skeleton_vector


#관절 좌표 + 벡터 차 
def get_all_featrues(keypoint):
    skeleton_vector = np.zeros((1, 1, len(COCO_SKELETON)*6))
    vector_list = []
    for i in range(len(COCO_SKELETON)):
        a_kpt, b_kpt = COCO_SKELETON[i][0], COCO_SKELETON[i][1]
        a_x, a_y = keypoint[a_kpt][0], keypoint[a_kpt][1]                                                     
        b_x, b_y = keypoint[b_kpt][0], keypoint[b_kpt][1]
        vector_list.extend([a_x, a_y, b_x, b_y, a_x-b_x, a_y-b_y])
    skeleton_vector[0][0] = np.array(vector_list) 
    return skeleton_vector


def inference_action(keypoint, model_type, input_type, window_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_type == 'cnn' and input_type == 'skeleton_sub' and window_size == 24:
        #class : 1: '걷기', 2: '달리기', 3: '앉기', 6: '쓰러짐', 9: '점프', 15: '포복'
        # path = './model/0725_model_scripted_e29_loss_0.13059952855110168.pt'  
        # path = './model/0725_2_model_scripted_e29_loss_0.024166611954569817.pt'

        path = './model/0725_3_model_scripted_e99_loss_2.6799294573720545e-05.pt'
        model = torch.jit.load(path)

    elif model_type == 'cnn' and input_type == 'skeleton_sub'  and window_size == 15:
        path = './model/0808_cnn_skeleton_sub_15_e99.pt'
        model = torch.load(path)

    elif model_type == 'cnn' and input_type == 'skeleton_sub' and window_size == 9 :
        path = './model/0731_cnn_skeleton_sub_9_scripted_e19_loss_0.099755.pt'
        model = torch.load(path)

    elif model_type == 'cnn' and input_type == 'skeleton_angle' and window_size == 15 :
        path = './model/0811_cnn_skeleton_sub_15_duplication_O_e32.pt'
        model = torch.load(path)

    elif model_type == 'cnn' and input_type == 'skeleton_angle' and window_size == 9 :
        path = './model/0809_cnn_skeleton_angle_9_duplication_O_e10.pt'
        model = torch.load(path)

    elif model_type == 'cnn' and input_type == 'key_skeleton_sub' and window_size == 9 :
        path = './model/0811_cnn_key_skeleton_sub_9_duplication_O_e5.pt'
        model = torch.load(path)

    elif model_type == 'lstm' and input_type == 'skeleton_sub'  and window_size == 24:
        path = './model/0726_lstm_model_e21_loss_0.6221391558647156.pt'
        model = torch.load(path)

    elif model_type == 'lstm' and input_type == 'skeleton_sub'  and window_size == 9:
        # path = './model/0809_lstm_skeleton_sub_9_d_X_e40_state_dict.pt'
        path = './model/0809_lstm_skeleton_sub_9_d_O_e17_state_dict.pt'

        model_args = {'mode' : 'multiframe', 
                      'window_size' : 9, 
                      'input_type' : 'skeleton_sub', 
                      'input_size' : 26,
                      'output_size'  : 6,
                      'hidden_dim' : 32,
                      'num_layers' :3,
                      'dropout' : 0.2}
        model = LSTM(**model_args)
        model.load_state_dict(torch.load(path))

    else :
        print("model input type parameter error")

    keypoint = torch.Tensor(keypoint)
    keypoint = keypoint.to(device)
    keypoint = F.normalize(keypoint, p=2, dim=1)

    model.to(device)
    model.eval()

    if model_type == 'cnn' :
        predictions = model(keypoint)
        
    # Get sequence predictions
    elif model_type == 'lstm' :
        h = model.init_hidden(1)
        h = tuple([e.data for e in h])
        predictions, (h_, c_) =  model(keypoint, h)
    
    if window_size == 9 or window_size == 15 :
        label = predictions.argmax().item()
        action = ACTION_DICT_VER2[label]
                                                                                                                                                                   
    else :
        label = predictions.argmax().item()
        action = ACTION_DICT[label]

    return action


def update_keypoint_each_id(keypoints_dict, object_ids, pose_results):
    for i, obj_id in enumerate(object_ids):
        if obj_id in keypoints_dict:
            keypoints_dict[obj_id].append(pose_results[i].preds)
        else:
            keypoints_dict[obj_id] = [pose_results[i].preds]



# #keypoint 받아서 skeleton 행렬 반환
# def subtract_vector_from_3dim( keypoint):

#     vector_list = np.empty(shape=(1, 13, 1))
#     for j in range(0, keypoint.shape[2], 2):
#         skeleton_vector = np.zeros((1, len(COCO_SKELETON), 2))

#         for i in range(len(COCO_SKELETON)):
#             a_kpt, b_kpt = COCO_SKELETON[i][0], COCO_SKELETON[i][1]
#             a_x, a_y = keypoint[0][a_kpt][j], keypoint[0][a_kpt][j+1]                                                     
#             b_x, b_y = keypoint[0][b_kpt][j], keypoint[0][b_kpt][j+1]
#             skeleton_vector[0][i] = np.array([a_x-b_x, a_y-b_y])
        
#         vector_list = np.concatenate((vector_list, skeleton_vector), axis = 2)
#     vector_list = np.delete(vector_list, 0, axis = 2)
    
#     return vector_list