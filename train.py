# coding:utf-8
# @Email: wangguisen@infinities.com.cn
# @Time: 2023/3/6 21:04
# @File: train.py
'''
train
'''
import os
import time
import joblib
import argparse
from config import get_config  # 返回一个字典，里边包含模型的一些配置信息
from pathlib import Path
from data.config import cfg_mnet, cfg_re50
from models.retinaface import RetinaFace
from convert_to_onnx import load_model
from utilsRe.timer import Timer
from Learner import face_learner
from utils import prepare_facebank_retina, prepare_facebank_retina_coll, load_facebank, draw_box_name, prepare_facebank

import torch, gc
import torch.backends.cudnn as cudnn

def get_args():
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-s", "--save", help="whether save", action="store_true")
    parser.add_argument('-th', '--threshold', help='threshold to decide identical faces', default=1.54, type=float)
    parser.add_argument("-u", "--update", help="whether perform update the facebank", action="store_true")
    parser.add_argument("-tta", "--tta", help="whether test time augmentation", action="store_true")
    parser.add_argument("-c", "--score", help="whether show the confidence score", action="store_true")
    args = parser.parse_args()
    return args

def tain_facebank(args, conf, bank_name, rec_mobile=False):
    '''

    :param args:
    :param conf:
    :param bank_name:
    :param det_model_path:
    :param face_thres:
    :param rec_mobile:
    :param update:
    :return:
    '''
    conf.facebank_path = conf.data_path / Path(bank_name + '/')
    torch.set_grad_enabled(False)
    confidence_threshold = 0.1  # 置信度阈值
    nms_threshold = 0.4
    vis_thres = 0.6
    cpu = False

    if 'Resnet' in conf.det_model_path:
        cfg = cfg_re50
    else:
        cfg = cfg_mnet

    net = RetinaFace(cfg=cfg, phase='test')
    trained_model = conf.det_model_path
    net = load_model(net, trained_model, cpu)
    net.eval()
    print('Finished loading model!')
    # print(net)
    cudnn.benchmark = True

    net = net.to(conf.device)
    resize = 1
    _t = {'forward_pass': Timer(), 'misc': Timer()}
    # LC
    conf.use_mobilfacenet = rec_mobile
    learner = face_learner(conf, True)
    learner.threshold = conf.face_thres  # args.threshold
    print('-' * 50 + conf.device.type + '-' * 50)
    if cpu:
        learner.load_state(conf, 'cpu_final.pth', True, True, cpu="True")
    else:
        learner.load_state(conf, 'final.pth', True, True, cpu="False")
    learner.model.eval()
    print('learner loaded')

    if conf.update:
        if conf.use_coll_faces_flag:
            targets, names = prepare_facebank_retina_coll(conf, learner.model, net, tta=args.tta)
        else:
            targets, names = prepare_facebank_retina(conf, learner.model, net, tta=args.tta)
        print('facebank updated')
    else:
        targets, names = load_facebank(conf)
        print('facebank loaded')

    # print(targets.shape)
    # print(names.shape)

    return conf.device, _t, net, cfg, resize, confidence_threshold, nms_threshold, vis_thres, learner, targets, args, names

def app_run():
    # 清除pytorch无用缓存
    gc.collect()
    torch.cuda.empty_cache()

    args = get_args()
    conf = get_config(training=False)

    if conf.use_coll_faces_flag:
        bank_name = 'facebank_coll'
    else:
        bank_name = 'facebank'

    device, _t, net, cfg, resize, confidence_threshold, nms_threshold, vis_thres, learner, targets, args, names = tain_facebank(
        args, conf, bank_name, rec_mobile=False)

    dict1 = {"device": device, "_t": _t, "net": net, "cfg": cfg, "resize": resize, "confidence_threshold": confidence_threshold,
             "nms_threshold": nms_threshold, "vis_thres": vis_thres, "learner": learner, "targets": targets, "args": args, "names": names,
             "use_coll_faces_flag": conf.use_coll_faces_flag}

    joblib.dump(dict1, './dict1.pkl')

if __name__ == '__main__':
    '''
    docker run -d --gpus '"device=0"' \
       --rm -it --name face_recognition \
       --shm-size 15G \
       -v /data/wgs/face_recognition:/home \
       wgs-torch/face_recognition:1.0 \
       sh -c "python /home/train.py 1>>/home/log/train.log 2>>/home/log/train.err"
       
       
    加载 RetinaFace
    加载 learner
    prepare_facebank_retina：对齐人脸、保存图库人脸特征emb(人脸特征向量) [set(图库人脸个数), embedding_size]、人名称   选择最大脸并对齐后提取特征
    
    '''

    # os.chdir('./face_recognition')

    start_time = time.time()

    app_run()

    end_time = time.time()
    print('Run Time: {:.0f}分 {:.0f}秒（{}）'.format((end_time - start_time) // 60, (end_time - start_time) % 60, (end_time - start_time)))
    print('----------')
    print('----------')
    print('----------')
    print()
    print()
    print()



