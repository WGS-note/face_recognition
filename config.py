
from easydict import EasyDict as edict
from pathlib import Path
import torch
from torch.nn import CrossEntropyLoss
from torchvision import transforms as trans

def get_config(training = True):
    # 使用easydict创建一个数组
    conf = edict()

    conf.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # True：所有人脸一个文件件；False：一个人一个文件夹
    conf.use_coll_faces_flag = True
    # conf.use_coll_faces_flag = False

    conf.det_model_path = './weights/mobilenet0.25_Final.pth'
    # conf.det_model_path = './weights/Resnet50_Final.pth'

    conf.face_thres = 1.0  # 计算距离后对比人脸相似度阈值

    conf.update = True   # 开启提取图库人脸特征点

    # 数据路径
    conf.data_path = Path('data')
    conf.work_path = Path('work_space/')
    conf.model_path = conf.work_path/'models'
    conf.log_path = conf.work_path/'log'
    conf.save_path = conf.work_path/'save'

    # 输入大小 112,112
    conf.input_size = [112, 112]
    # embedding层大小：512层    提取人脸特征的维度
    conf.embedding_size = 512

    # 字面意思：使用移动的面部网络结构
    conf.use_mobilfacenet = False
    conf.net_depth = 50
    conf.drop_ratio = 0.6
    conf.net_mode = 'ir_se'  # or 'ir'
    conf.test_transform = trans.Compose([
                    trans.ToTensor(),
                    trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
    conf.data_mode = 'emore'
    conf.vgg_folder = conf.data_path/'faces_vgg_112x112'
    conf.ms1m_folder = conf.data_path/'faces_ms1m_112x112'
    conf.emore_folder = conf.data_path/'faces_emore'

    # 批量大小
    conf.batch_size = 50  # irse net depth 50
#   conf.batch_size = 200 # mobilefacenet

#--------------------Training Config ------------------------    
    if training:        
        conf.log_path = conf.work_path/'log'
        conf.save_path = conf.work_path/'save'
    #     conf.weight_decay = 5e-4
        conf.lr = 1e-3
        #conf.milestones = [12, 15, 18]
        conf.milestones = [20, 24, 28]
        conf.momentum = 0.9
        conf.pin_memory = True
#         conf.num_workers = 4 # when batchsize is 200
        conf.num_workers = 1  # 3
        conf.ce_loss = CrossEntropyLoss()
#--------------------Inference Config ------------------------
    else:
        conf.facebank_path = conf.data_path/'facebank/'
        conf.threshold = 1.5
        conf.face_limit = 10 
        # when inference, at maximum detect 10 faces in one image, my laptop is slow
        conf.min_face_size = 30 
        # the larger this value, the faster deduction, comes with tradeoff in small faces
    return conf