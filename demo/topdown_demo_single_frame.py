# Copyright (c) OpenMMLab. All rights reserved.
import mimetypes
import os
import datetime
import time
from argparse import ArgumentParser

import cv2
import json_tricks as json
import mmcv
import mmengine
import numpy as np
import torch

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, split_instances
from mmpose.utils import adapt_mmdet_pipeline

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


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


def subtract_vector(keypoint, mode='single'):
    
    skeleton_vector = np.zeros((1, len(COCO_SKELETON), 2))
    if mode == 'single' :
        for i in range(len(COCO_SKELETON)):
            a_kpt, b_kpt = COCO_SKELETON[i][0], COCO_SKELETON[i][1]
            a_x, a_y = keypoint[0][a_kpt][0], keypoint[0][a_kpt][1]
            b_x, b_y = keypoint[0][b_kpt][0], keypoint[0][b_kpt][1]
            skeleton_vector[0][i] = np.array([a_x - b_x, a_y - b_y])

    elif mode == 'multi':
        for i in range(len(COCO_SKELETON)):
            a_kpt, b_kpt = COCO_SKELETON[i][0], COCO_SKELETON[i][1]
            a_x, a_y = keypoint[a_kpt][0], keypoint[a_kpt][1]
            b_x, b_y = keypoint[b_kpt][0], keypoint[b_kpt][1]
            skeleton_vector[0][i] = np.array([a_x - b_x, a_y - b_y])

    skeleton_vector = torch.Tensor(skeleton_vector)

    return skeleton_vector


def inference_action(keypoint, input_type):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if input_type == 'keypoint' :
        path = './model/0714_keypoint_model.pt'
        
    elif input_type == 'skeleton_matrix' :
        path = './model/0714_skeleton_matrix_model.pt'

    elif input_type == 'skeleton_sub' :
        path = './model/0715_skeleton_sub_model.pt'

    else :
        print("model input type parameter error")

    model = torch.load(path)
    model.to(device)
    model.eval()

    predictions = model(keypoint)
    label = predictions.argmax().item()
    action = ACTION_DICT[label]

    return action

def process_one_image(args,
                      img,
                      detector,
                      pose_estimator,
                      visualizer=None,
                      show_interval=0):
    """Visualize predicted keypoints (and heatmaps) of one image."""

    # predict bbox
    last_time = time.time()
    print( time.strftime('%c', time.localtime(time.time())))
    det_result = inference_detector(detector, img)
    pred_instance = det_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate(
        (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    bboxes = bboxes[np.logical_and(pred_instance.labels == args.det_cat_id,
                                   pred_instance.scores > args.bbox_thr)]
    bboxes = bboxes[nms(bboxes, args.nms_thr), :4]

    # predict keypoints
    pose_results = inference_topdown(pose_estimator, img, bboxes)
    data_samples = merge_data_samples(pose_results)
    fps = 1/(time.time()-last_time)
    print('FPS : {}'.format(fps))
    sample = data_samples.pred_instances.keypoints
    sample_len = len(sample)

    #predict action
    action = ""

    if sample_len ==1 :
        keypoint = subtract_vector(sample)

        action = inference_action(keypoint.to(device=args.device), 'skeleton_sub')
        # print(action)
        if bboxes.any():
            x,y,w,h = bboxes[0]
            x, y, w, h = int(x), int(y), int(w), int(h)
            img = cv2.putText(img, 'action: '+ action, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    else :
        for keypoint_i in sample :    
            keypoint = subtract_vector(keypoint_i, mode='multi')

            action = inference_action(keypoint.to(device=args.device), 'skeleton_sub')
            # print(action)
            x,y,w,h = bboxes[0]
            x, y, w, h = int(x), int(y), int(w), int(h)
            img = cv2.putText(img, 'action: '+ action, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # show the results
    if isinstance(img, str):
        img = mmcv.imread(img, channel_order='rgb')
    elif isinstance(img, np.ndarray):
        img = mmcv.bgr2rgb(img)

    img = cv2.putText(img, 'fps: '+ "%.2f"%(fps), (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    black_img = np.zeros_like(img, np.uint8)

    if visualizer is not None:
        visualizer.add_datasample(
            'result',
            img,
            #black_img,
            data_sample=data_samples,
            draw_gt=False,
            draw_heatmap=args.draw_heatmap,
            draw_bbox=args.draw_bbox,
            show_kpt_idx=args.show_kpt_idx,
            skeleton_style=args.skeleton_style,
            show=args.show,
            wait_time=show_interval,
            kpt_thr=args.kpt_thr)

    # if there is no instance detected, return None
    return data_samples.get('pred_instances', None)


def main():
    """Visualize the demo images.

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument(
        '--input', type=str, default='', help='Image/Video file')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    parser.add_argument(
        '--output-root',
        type=str,
        default='',
        help='root of the output img file. '
        'Default not saving the visualization images.')
    parser.add_argument(
        '--save-predictions',
        action='store_true',
        default=False,
        help='whether to save predicted results')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=0,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
    parser.add_argument(
        '--nms-thr',
        type=float,
        default=0.3,
        help='IoU threshold for bounding box NMS')
    parser.add_argument(
        '--kpt-thr',
        type=float,
        default=0.3,
        help='Visualizing keypoint thresholds')
    parser.add_argument(
        '--draw-heatmap',
        action='store_true',
        default=False,
        help='Draw heatmap predicted by the model')
    parser.add_argument(
        '--show-kpt-idx',
        action='store_true',
        default=False,
        help='Whether to show the index of keypoints')
    parser.add_argument(
        '--skeleton-style',
        default='mmpose',
        type=str,
        choices=['mmpose', 'openpose'],
        help='Skeleton style selection')
    parser.add_argument(
        '--radius',
        type=int,
        default=3,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')
    parser.add_argument(
        '--show-interval', type=int, default=0, help='Sleep seconds per frame')
    parser.add_argument(
        '--alpha', type=float, default=0.8, help='The transparency of bboxes')
    parser.add_argument(
        '--draw-bbox', action='store_true', help='Draw bboxes of instances')

    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parser.parse_args()

    assert args.show or (args.output_root != '')
    assert args.input != ''
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    output_file = None
    if args.output_root:
        mmengine.mkdir_or_exist(args.output_root)
        output_file = os.path.join(args.output_root,
                                   os.path.basename(args.input))
        if args.input == 'webcam':
            output_file += '.mp4'

    if args.save_predictions:
        assert args.output_root != ''
        args.pred_save_path = f'{args.output_root}/results_' \
            f'{os.path.splitext(os.path.basename(args.input))[0]}.json'

    # build detector
    detector = init_detector(
        args.det_config, args.det_checkpoint, device=args.device)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    # build pose estimator
    pose_estimator = init_pose_estimator(
        args.pose_config,
        args.pose_checkpoint,
        device=args.device,
        cfg_options=dict(
            model=dict(test_cfg=dict(output_heatmaps=args.draw_heatmap))))

    # build visualizer
    pose_estimator.cfg.visualizer.radius = args.radius
    pose_estimator.cfg.visualizer.alpha = args.alpha
    pose_estimator.cfg.visualizer.line_width = args.thickness
    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_pose_estimator
    visualizer.set_dataset_meta(
        pose_estimator.dataset_meta, skeleton_style=args.skeleton_style)

    if args.input == 'webcam':
        input_type = 'webcam'
    else:
        input_type = mimetypes.guess_type(args.input)[0].split('/')[0]

    if input_type == 'image':

        # inference
        pred_instances = process_one_image(args, args.input, detector,
                                           pose_estimator, visualizer)

        if args.save_predictions:
            pred_instances_list = split_instances(pred_instances)

        if output_file:
            img_vis = visualizer.get_image()
            mmcv.imwrite(mmcv.rgb2bgr(img_vis), output_file)

    elif input_type in ['webcam', 'video']:

        if args.input == 'webcam':
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(args.input)

        video_writer = None
        pred_instances_list = []
        frame_idx = 0

        while cap.isOpened():
            success, frame = cap.read()
            frame_idx +=1

            if not success:
                break

            # if frame_idx %2 ==0 :
            #     continue

            print("frame_idx : ", frame_idx)

            # topdown pose estimation
            pred_instances = process_one_image(args, frame, detector,
                                               pose_estimator, visualizer,
                                               0.001)
            
            if args.save_predictions:
                # save prediction results
                pred_instances_list.append(
                    dict(
                        frame_id=frame_idx,
                        instances=split_instances(pred_instances)))

            # output videos
            if output_file:
                frame_vis = visualizer.get_image()

                if video_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    # the size of the image with visualization may vary
                    # depending on the presence of heatmaps
                    video_writer = cv2.VideoWriter(
                        output_file,
                        fourcc,
                        25,  # saved fps
                        (frame_vis.shape[1], frame_vis.shape[0]))

                video_writer.write(mmcv.rgb2bgr(frame_vis))

            # press ESC to exit
            if cv2.waitKey(5) & 0xFF == 27:
                break

            time.sleep(args.show_interval)

        if video_writer:
            video_writer.release()

        cap.release()

    else:
        args.save_predictions = False
        raise ValueError(
            f'file {os.path.basename(args.input)} has invalid format.')

    if args.save_predictions:
        with open(args.pred_save_path, 'w') as f:
            json.dump(
                dict(
                    meta_info=pose_estimator.dataset_meta,
                    instance_info=pred_instances_list),
                f,
                indent='\t')
        print(f'predictions have been saved at {args.pred_save_path}')


if __name__ == '__main__':
    main()
