import copy

from util_ import transforms, utils, engine
import torch
from model.mark_R_CNN import MaskRcnnResnet50Fpn
from loadataset import BubbleDataset
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# def evaluate(model, data_loader, device, num_classes):
#     model.eval()
#     confmat = utils.ConfusionMatrix(num_classes)
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     header = 'Test:'
#     with torch.no_grad():
#         for image, target in metric_logger.log_every(data_loader, 100, header):
#             image, target = image[0].to(device), target[0].to(device)
#             output = model(image)
#             output = output['out']
#
#             confmat.update(target.flatten(), output.argmax(1).flatten())
#
#         confmat.reduce_from_all_processes()
#
#     return confmat


if __name__ == '__main__':
    '##############  step-1 基础超参数 #################'
    LR = 0.005
    BATCH_SIZE = 1
    MAX_EPOCH = 100
    box_acc = 0
    segm_acc = 0
    '##############  step-2 创建模型 ################'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device in ', device)

    model = MaskRcnnResnet50Fpn()
    model.to(device)

    '##############  step-3 加载数据 ##################'

    '''   
    随机裁剪、resize、归一化、标准化等预处理在使用maskrcnn算法时，并不需要使用
    '''
    train_transforms = transforms.Compose([
        transforms.PILToTensor(),  # 张量化
        transforms.ConvertImageDtype(torch.float),  # dtype
        transforms.RandomHorizontalFlip(0.5),  # 随机翻转

    ])
    dataset = BubbleDataset('datasets', train_transforms)
    # indices = torch.randperm(len(dataset)).tolist()
    # dataset = torch.utils.data.Subset(dataset, indices[:-50])

    train_size = int(0.9 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1,
        collate_fn=utils.collate_fn)

    vaild_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1,
        collate_fn=utils.collate_fn)

    '##############  step-4 Training ##################'
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LR,
                                momentum=0.9, weight_decay=0.0005)


    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3, gamma=0.1)
    model_without_ddp = model
    for epoch in range(1, MAX_EPOCH + 1):
        # train for one epoch, printing every 10 iterations
        engine.train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=1)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        iou = engine.evaluate(model, vaild_loader, device=device)
        bbox_avg_acc = iou.coco_eval['bbox'].stats[0]
        segm_avg_acc = iou.coco_eval['segm'].stats[0]

        if bbox_avg_acc >= box_acc or segm_avg_acc >= segm_acc or epoch == MAX_EPOCH:
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts,
                       'net/Epoch_{:0>3d}_Box{:.4f}_Segm{:.4f}.pth'.format(epoch, bbox_avg_acc, segm_avg_acc))
            box_acc = bbox_avg_acc
            segm_acc = segm_avg_acc
