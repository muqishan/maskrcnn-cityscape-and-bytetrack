import cv2
from PIL import Image
from model.mark_R_CNN import MaskRcnnResnet50Fpn
import os
import torch
# from util_ import transforms
import torchvision.transforms as transforms
from util_ import visualize

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)
train_transforms = transforms.Compose([
    transforms.ToTensor(),  
    # transforms.Resize((1024, 1024)),
    # transforms.ConvertImageDtype(torch.float),  
    # transforms.RandomHorizontalFlip(0.5),  
])


def infer(img_path=None, image=None):
    '''

    :param img_path:
    :param img:  PIL.img
    :return: predictions, img:pil
    '''
    if img_path:
        img = Image.open(img_path).convert('RGB')
    else:
        TURN = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(TURN)
    inputs = train_transforms(img)
    inputs = inputs.to(device)
    model_path = 'net/Epoch_002_Box0.0150_Segm0.0228.pth'
    net = MaskRcnnResnet50Fpn()
    checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint)
    model = net.to(device)
    model.eval()
    predictions = model([inputs])
    return predictions, img


if __name__ == '__main__':
    r, img = infer(img_path='datasets/JPEGImages/00000.jpg')
    class_names = [
        '_background_',
        'bubble',
    ]
    visualize.display_instances(img, r[0]['boxes'], r[0]['masks'], r[0]['labels'], class_names, r[0]['scores'])