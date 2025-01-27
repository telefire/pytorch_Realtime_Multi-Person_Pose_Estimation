import os
import re
import sys
import torch
sys.path.append('.')
from collections import OrderedDict
from evaluate.coco_eval import run_eval
from lib.network.rtpose_vgg import get_model, use_vgg
from lib.network.openpose import OpenPose_Model, use_vgg
from torch import load

#Notice, if you using the 
with torch.autograd.no_grad():
    # this path is with respect to the root of the project

    '''
    weight_name = 'pose_model.pth'
    state_dict = torch.load(weight_name)['state_dict']
    
    new_state_dict = OrderedDict()
    for k,v in state_dict.items():
        name = k[6:]
        new_state_dict[name]=v
        
    model = get_model(trunk='vgg19')
    #model = openpose = OpenPose_Model(l2_stages=4, l1_stages=2, paf_out_channels=38, heat_out_channels=19)
    #model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(new_state_dict)
    model.eval()
    model.float()
    model = model.cuda()
    '''
    
    model = get_model('mobilenet')     

    raw_weight = torch.load("mobilenet_best_pose.pth")
    w = {}

    for k in raw_weight.keys():
        w[k[7:]] = raw_weight[k]


    #model.load_state_dict(torch.load("pose_model.pth"))
    model.load_state_dict(w)
    model.cuda()
    model.float()
    model.eval()


    # The choice of image preprocessing include: 'rtpose', 'inception', 'vgg' and 'ssd'.
    # If you use the converted model from caffe, it is 'rtpose' preprocess, the model trained in 
    # this repo used 'vgg' preprocess
    run_eval(image_dir= 'train/dataset/COCO/images/val2017', anno_file = 'train/dataset/COCO/annotations/person_keypoints_val2017.json', vis_dir = 'vis_val2017', model=model, preprocess='vgg')


