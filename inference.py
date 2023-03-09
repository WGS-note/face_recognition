# coding:utf-8
# @Email: wangguisen@infinities.com.cn
# @Time: 2023/3/6 21:21
# @File: inference.py
'''
inference
'''
import os
import joblib
import cv2
from PIL import Image
import numpy as np
from layers.functions.prior_box import PriorBox
from utilsRe.box_utils import decode, decode_landm
from utilsRe.nms.py_cpu_nms import py_cpu_nms
from mtcnn_pytorch.src.align_trans import get_reference_facial_points, warp_and_crop_face
from config import get_config
from utils import load_facebank, draw_box_name, prepare_facebank
import time

import torch

def read_img(filepath=None):
    flag = 1
    if filepath is None:
        flag = 0
    img = cv2.imread(filepath)

    return flag, img

def Mylign_multi(img, detRes, vis_thres):
    boxes = []
    landmarks = []
    for b in detRes:
        if b[4] < vis_thres:
            continue
        box = [b[0], b[1], b[2], b[3], b[4]]
        landmark = [b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12], b[13], b[14]]
        boxes.append(box)
        landmarks.append(landmark)

    # print('boxes   ', len(boxes))            # 1
    # print('landmarks   ', len(landmarks))    # 1

    faces = []
    refrence = get_reference_facial_points(default_square=True)
    for landmark in landmarks:
        facial5points = []
        ij = 0
        for j in range(5):
            l1 = [landmark[ij], landmark[ij + 1]]
            facial5points.append(l1)
            ij += 2
        # facial5points = [[landmark[j],landmark[j+1]] for j in range(5)]
        warped_face = warp_and_crop_face(np.array(img), facial5points, refrence, crop_size=(112, 112))
        faces.append(Image.fromarray(warped_face))
    return boxes, landmarks, faces

def detection(device, _t, net, cfg, resize, confidence_threshold, nms_threshold, vis_thres, learner, targets, args, names, filepath=None):
    '''

    :param device: device
    :param _t:
    :param net: RetinaFace
    :param cfg: RetinaFace backbone（Resnet50、mobilenet0.25 V1，详见：./data/config.py）
    :param resize: default 1
    :param confidence_threshold: 置信度阈值 default 1
    :param nms_threshold: default 0.4
    :param vis_thres: default 0.6
    :param learner: face model，用于提取脸部关键点特征，结果为 targets
    :param targets: 图库人脸特征 [n, embedding_size]，n=set(图库人脸个数)
    :param args:
    :param names: 图库人脸名称
    :param filepath: input images path
    :return:
    '''
    isSuccess, frame = read_img(filepath)
    # tmpF = frame.copy()

    # assert isSuccess == 0, '请传入正确的图片'
    if isSuccess == 0:
        raise ValueError('请传入正确的图片')
    else:
        image = Image.fromarray(frame)
        img = np.float32(frame)

        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])   # ？
        img -= (104, 117, 123)   # ?
        img = img.transpose(2, 0, 1)  # [c, h, w]
        img = torch.from_numpy(img).unsqueeze(0)  # [1， c, h, w]
        img = img.to(device)
        scale = scale.to(device)

        _t['forward_pass'].tic()
        loc, conf, landms = net(img)  # forward pass
        _t['forward_pass'].toc()

        _t['misc'].tic()
        priorbox = PriorBox(cfg, image_size=(im_height, im_width))   # 与Faster RCNN的anchor类似，目标的预设框
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize   # ？
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > confidence_threshold)[0]

        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, nms_threshold)

        dets = dets[keep, :]
        landms = landms[keep]

        dets = np.concatenate((dets, landms), axis=1)
        _t['misc'].toc()

        bboxes, landmarks, faces = Mylign_multi(image, dets, vis_thres)

        # print('---!!!---', len(faces))   # ?  1

        # bboxes = np.array(bboxes)
        # bboxes = bboxes[:, :-1]  # shape:[10,4], only keep 10 highest possibiity faces
        # bboxes = bboxes.astype(int)
        # bboxes = bboxes + [-1, -1, 1, 1]  # personal choice

        conf = get_config(False)
        # results, score = learner.infer(conf, faces, targets, args.tta)
        results, score, flag = learner.infer_multi(conf, faces, targets, args.tta)

        # print('---+++---', results, score, flag)

        # for idx, bbox in enumerate(bboxes):
        #     if args.score:
        #         frame = draw_box_name(bbox, names[results[idx] + 1] + '_{:.2f}'.format(score[idx]), frame)
        #     else:
        #         frame = draw_box_name(bbox, names[results[idx] + 1], frame)

        detection_id_dict = {}
        result_id = results.tolist()[0]
        result_name = names[result_id + 1]   # 最开始存了Unknown，所以要+1
        score = score.tolist()[0]

        # print('------', result_id, result_name, score)

        if flag == 1:
            detection_id_dict[result_name] = score
        else:
            detection_id_dict['Unknown, the most similar is [ ' + result_name + ' ]'] = score

    cv2.destroyAllWindows()

    return detection_id_dict

def face_recognition(filepath):

    dict1 = joblib.load('./dict1.pkl')

    detection_id_dict = detection(dict1['device'], dict1['_t'], dict1['net'], dict1['cfg'], dict1['resize'], dict1['confidence_threshold'], dict1['nms_threshold'], dict1['vis_thres'],
                                    dict1['learner'], dict1['targets'], dict1['args'], dict1['names'], filepath)

    return detection_id_dict


if __name__ == '__main__':
    '''
    docker run -d --gpus '"device=0"' \
       --rm -it --name face_recognition \
       --shm-size 15G \
       -v /data/wgs/face_recognition:/home \
       wgs-torch/face_recognition:1.0 \
       sh -c "python /home/inference.py 1>>/home/log/inference.log 2>>/home/log/inference.err"
    '''

    # os.chdir('./face_recognition')

    # filepath = r"./data/imgPath/lyf.jpg"
    # filepath = r"./data/imgPath/dlrb.jpg"
    # filepath = r"./data/imgPath/ym.jpg"
    # filepath = r"./data/imgPath/yq.jpg"
    # filepath = r"./data/imgPath/bailu.jpg"

    filepath = r"./data/imgPath/0390.png"
    # filepath = r"./data/imgPath/0010.png"

    # filepath = r"./data/imgPath/0003.png"
    # filepath = r"./data/imgPath/0880.png"
    # filepath = r"./data/imgPath/1850.png"
    # filepath = r"./data/imgPath/1857.png"

    start_time = time.time()

    detection_id_dict = face_recognition(filepath)

    print('detection_id_dict: ', detection_id_dict)

    end_time = time.time()
    print('Run Time: {:.0f}分 {:.0f}秒（{}）'.format((end_time - start_time) // 60, (end_time - start_time) % 60, (end_time - start_time)))
    print('----------')
    print('----------')
    print('----------')
    print()
    print()
    print()


