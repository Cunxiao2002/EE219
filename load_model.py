import torch
from torch.serialization import load
import torchvision.models as models

# pretrained=True使用预训练的模型
resnet18 = models.resnet18(pretrained=True)#创建实例，模型下载.Pth文件
model_path = '/home/ubuntu/lab/lab1/model/model.pth' 
model_data = torch.load(model_path)
resnet18.load_state_dict(model_data)
print(resnet18)