# 人脸识别 1:N

我公众号：WGS的学习笔记

![](./tmp.jpg)



基于 Facenet + Retinaface 的人脸识别；

+ RetinaFace（backbone 为 mobilenet0.25）；

+ Facenet（backbone 为 ir_se_50）；



## 程序入口

+ `./train.py`：【训练、提取人脸特征点】入口，保存 `./dict1.pkl` ，内存图库人脸特征向量；
  + `config.py` 里 use_coll_faces_flag 参数，True为所有图片共存一个文件夹，False为一个人一个文件夹；
+ `./inference.py`：【推理】入口；
  + 测试结果如下：

```
input: liuyifei, 识别成功：
detection_id_dict:  {'liuyifei': 0.5516928434371948}

input: bailu, 输入未知：
detection_id_dict:  {'Unknown, the most similar is [ liuyifei ]': 1.3141696453094482}

input: bailu, 输入未知：
detection_id_dict:  {'Unknown, the most similar is [ 0066 ]': 1.065873622894287}

input: 0390, 识别成功：
detection_id_dict:  {'0390': 0.11204338073730469}
```

> Tesla T4 
>
> 图库1000张图，提取Face embedding约2分钟；
>
> 识别一张图3s；



## 运行说明

外层包web程序：

+ 初始化 训练、提取人脸特征点，`train.py` 的 `app_run()`；
  + 图库更新需重新运行；
+ 人脸识别，`inference.py` 的 `face_recognition(filepath)`；

需要改动的参数 config.py：

+ use_coll_faces_flag，True为所有图片共存一个文件夹，False为一个人一个文件夹；

建议不动的参数：

+ net_mode：Facenet bottleneck，['ir', 'ir_se']；
+ det_model_path，RetinaFace backbone，['Resnet50', 'mobilenet0.25']；
+ face_thres：计算距离后对比人脸相似度阈值；



## 权重

链接: https://pan.baidu.com/s/1srBdBsxHl1bHVA7qwbMdhA?pwd=vegc 提取码: vegc 
--来自百度网盘超级会员v1的分享



## 基于styleGAN生成的一些人脸照片数据集

明星脸1万张：https://pan.baidu.com/s/1g5ASVZcRoYvClxqsQpShXQ（提取码：XVAL）

萌娃脸1万张：https://pan.baidu.com/s/1JfyZYyfGzdO6TgKzOuWa0Q（提取码：75AG）

黄种人脸5万张：https://pan.baidu.com/s/1X2RTqKKhG5mXx0d4HzfZLg（提取码：A01B）

网红脸1万张：https://pan.baidu.com/s/1Sn6j9g-8sddIvViGEawAWQ（提取码：3IQT）

超模脸1万张：https://pan.baidu.com/s/1G5lTsk1TJPZMCHqudQqqYg（提取码：2A5W）



