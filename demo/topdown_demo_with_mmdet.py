# Copyright (c) OpenMMLab. All rights reserved.
import mimetypes
import os
import time
from argparse import ArgumentParser

import cv2
import json_tricks as json
import mmcv
import mmengine
import numpy as np

from sort import *
from inference_action import *

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


def process_one_image(args,
                      img,
                      detector,
                      pose_estimator,
                      tracker,
                      keypoints_dict,
                      action_dict,
                      visualizer=None,
                      show_interval=0):
    """Visualize predicted keypoints (and heatmaps) of one image."""

    #action recognition model paragemeter 
    model_type = 'cnn'
    mode = 'multiframe'
    input_type = 'skeleton_sub'
    window_size = 9
    last_time = time.time()

    # predict bbox
    det_result = inference_detector(detector, img)
    pred_instance = det_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate(
        (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    bboxes = bboxes[np.logical_and(pred_instance.labels == args.det_cat_id,
                                   pred_instance.scores > args.bbox_thr)]
    bboxes = bboxes[nms(bboxes, args.nms_thr), :4]
    print("bboxes\n", bboxes)

    #track object
    mot_tracker = tracker
    detections = bboxes

    #update SORT
    track_bbs_ids = mot_tracker.update(detections) #[id, xx1, y1, xx2, yy2]

    # track_bbs_ids is a np array where each row contrains a valid bounding box and track_id (last column)
    for d in track_bbs_ids:
        # print('track_bbs : %d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(d[4],d[0],d[1],d[2]-d[0],d[3]-d[1]))
        print('track_bbs : id:{}, x:{}, y:{}, width:{}, height:{},1,-1,-1,-1'.format(int(d[4]),d[0],d[1],d[2]-d[0],d[3]-d[1]))

        #draw
        d = d.astype(np.int32)
        obj_id = int(d[4])
        img = cv2.rectangle(img, (d[0], d[1]), (d[2], d[3]),  (0, 255, 0), 2)
        img = cv2.putText(img,  str(d[4]), (d[0], d[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


        if input_type == 'skeleton_sub' :
            #차벡터 : (x, y)
            num_col_per_window = window_size * 2

        elif input_type == 'skeleton_angle' :
            #cos 각도 : (x, y, angle)
            num_col_per_window = window_size * 3

        elif input_type == 'key_skeleton_angle' :
            #cos 각도 : (x, y, angle)
            num_col_per_window = window_size * 3

        if model_type == 'cnn' :
            if obj_id in keypoints_dict and keypoints_dict[obj_id].shape[2] >=num_col_per_window :
                start = keypoints_dict[obj_id].shape[2] - num_col_per_window
                keypoint_for_action = keypoints_dict[obj_id][:, :, start : ]
                action = inference_action(keypoint_for_action, model_type=model_type, input_type=input_type, window_size=window_size )
                if action == "Climb stairs" or action == 'run':
                    action = "walk" 

                if obj_id in action_dict:
                    action_dict[obj_id].append(action)
                else:
                    action_dict[obj_id] = [""]
                    action_dict[obj_id].append(action)
        
                if action != action_dict[obj_id][-2] :

                    action = "" 
                    
                img = cv2.putText(img,  '{} '.format(action), (d[0]+20, d[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        elif model_type == 'lstm' :
            if obj_id in keypoints_dict and keypoints_dict[obj_id].shape[1] >= window_size  :
                start = keypoints_dict[obj_id].shape[1] - num_col_per_window
                keypoint_for_action = keypoints_dict[obj_id][:, start :, :  ]
                action = inference_action(keypoint_for_action, model_type=model_type, input_type=input_type, window_size=window_size )
                if action == "Climb stairs" :
                    action = "walk" 
                img = cv2.putText(img,  '{} '.format(action), (d[0]+20, d[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                img = cv2.putText(img,  '{} '.format(action), (25,80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    object_ids = track_bbs_ids[:, 4].astype(np.int32)


    # predict keypoints
    # pose_results = inference_topdown(pose_estimator, img, bboxes)
    pose_results = inference_topdown(pose_estimator, img, track_bbs_ids[:, :4])
    data_samples = merge_data_samples(pose_results)
    fps = 1/(time.time()-last_time)
    print('FPS : {}'.format(fps))
    # pose_results = data_samples.pred_instances.keypoints
    pred_instances =data_samples.get('pred_instances', None)


    #add keypoints coord
    for i, obj_id in enumerate(object_ids):
        if model_type == 'cnn':
            
            if input_type == 'skeleton_sub' :
                keypoint_13 = subtract_vector(pred_instances.keypoints[i]) #(1, 13, 2)
                
            elif input_type == 'skeleton_angle':
                keypoint_13 = get_cos_angle(pred_instances.keypoints[i])

            elif input_type == 'key_skeleton_sub' :
                keypoint_13 = get_all_featrues(pred_instances.keypoints[i])

            if np.any(keypoint_13 == 0.) :
                continue

            if obj_id in keypoints_dict:
                keypoints_dict[obj_id] = np.append(keypoints_dict[obj_id], keypoint_13, axis=2)
            else:
                keypoints_dict[obj_id] = keypoint_13
                
        elif model_type == 'lstm' : 
            keypoint_13 = lstm_subtract_vector(pred_instances.keypoints[i])  #(1, 26)

            if obj_id in keypoints_dict:
                keypoints_dict[obj_id] = np.append(keypoints_dict[obj_id], keypoint_13, axis=1)
            else:
                keypoints_dict[obj_id] = keypoint_13


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
    
        mot_tracker = Sort()
        keypoints_dict = {}
        action_dict = {}
        while cap.isOpened():
            success, frame = cap.read()
            frame_idx += 1

            if not success:
                break

            # if frame_idx %2 ==0 :
            #     continue

            print("frame_idx : ", frame_idx)
            
            # topdown pose estimation
            pred_instances = process_one_image(args, 
                                            frame, 
                                            detector, 
                                            pose_estimator,
                                            mot_tracker,
                                            keypoints_dict,
                                            action_dict,
                                            visualizer,
                                            0.001)
    
            if args.save_predictions:
                # save prediction results
                pred_instances_list.append(
                    dict(
                        frame_id=frame_idx,
                        instances=split_instances(pred_instances)))

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
