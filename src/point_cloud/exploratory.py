import sys
import importlib
from src.point_cloud.PointGPT.segmentation.models import pt
import torch
from src.point_cloud.PointGPT.segmentation.dataset import PartNormalDataset

model_name = 'PointGPT_S'
MODEL = pt
num_part = 50
npoint = 2048

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True

if model_name == 'PointGPT_S':
    classifier = MODEL.get_model(num_part, trans_dim=384, depth=12, drop_path_rate=0.1, num_heads=6, decoder_depth=4, group_size=32, num_group=128, prop_dim=1024, label_dim1=512, label_dim2=256, encoder_dims=384)
    classifier = classifier.cuda()
    criterion = MODEL.get_loss().cuda()
    classifier.apply(inplace_relu)


# try loading from a checkpoint
ckpt_path = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/PointGPT/pretrained.pth"
# ckpt_path = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/PointGPT/part_seg.pth"
classifier.load_model_from_ckpt(ckpt_path)


# try loading pointcloud data
root = "/media/nick/hdd02/Cole Trapnell's Lab Dropbox/Nick Lammers/Nick/pecfin_dynamics/PointGPT/data/ShapeNet55/"
TRAIN_DATASET = PartNormalDataset(
        root=root, npoints=npoint, split='trainval', normal_channel=False)

trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=16, shuffle=True)

test = next(iter(trainDataLoader))

points = test[0]
label = test[1]
target = test[2]

points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()

points = points.transpose(2, 1)

point_features = classifier.forward(points, label)
# point_features = classifier.extract_features(points)
# classifier.forward(points, label)
# seg_pred = classifier(points, to_categorical(label, num_classes=16))
#
# seg_pred = seg_pred.contiguous().view(-1, num_part)
# target = target.view(-1, 1)[:, 0]
# pred_choice = seg_pred.data.max(1)[1]