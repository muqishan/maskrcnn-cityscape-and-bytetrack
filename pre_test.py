import torchvision
from loadataset import BubbleDataset
import torch
from util_ import utils
from torchvision import transforms
from model.mark_R_CNN import MaskRcnnResnet50Fpn
from util_ import transforms as T

'''
采坑: 
png: png 3通道
mask: 单通道
'''


def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MaskRcnnResnet50Fpn()
    print('device in ', device)
    model.to(device)
    train_transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.RandomHorizontalFlip(0.5),
    ])

    dataset = BubbleDataset('datasets', get_transform(train=True))
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=1,
        collate_fn=utils.collate_fn)
    # For Training
    images, targets = next(iter(data_loader))
    images = list(image for image in images)
    images = [i.to(device) for i in images]
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    output = model(images, targets)  # Returns losses and detections
    # For inference
    model.eval()


    # model.to(device)

    x = [i.to(device) for i in [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]]
    print(x)
    predictions = model(x)  # Returns predictions

    print(predictions)
