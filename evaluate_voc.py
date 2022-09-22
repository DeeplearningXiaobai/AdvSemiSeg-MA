import argparse
import numpy as np
import os
from packaging import version
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils import data, model_zoo
from model24.deeplab import Res_Deeplab
from dataset.voc_dataset import VOCDataSet
from PIL import Image
import time
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# IMG_MEAN = np.array((52.00698793,58.66876762,61.67891434), dtype=np.float32)
IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

MODEL = 'DeepLab'
DATA_DIRECTORY = './dataset/VOC2012'
DATA_LIST_PATH = './dataset/voc_list/val.txt'
IGNORE_LABEL = 255
NUM_CLASSES = 9
NUM_STEPS = 180  # Number of images in the validation set.
RESTORE_FROM = './AdvSemiSegVOC0.125-8d75b3f1.pth'
PRETRAINED_MODEL = None
SAVE_DIRECTORY = 'results'

pretrianed_models_dict = {'semi0.125': 'http://vllab1.ucmerced.edu/~whung/adv-semi-seg/AdvSemiSegVOC0.125-03c6f81c.pth',
                          'semi0.25': 'http://vllab1.ucmerced.edu/~whung/adv-semi-seg/AdvSemiSegVOC0.25-473f8a14.pth',
                          'semi0.5': 'http://vllab1.ucmerced.edu/~whung/adv-semi-seg/AdvSemiSegVOC0.5-acf6a654.pth',
                          'advFull': 'http://vllab1.ucmerced.edu/~whung/adv-semi-seg/AdvSegVOCFull-92fbc7ee.pth'}


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="VOC evaluation script")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab/DRN")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--pretrained-model", type=str, default=PRETRAINED_MODEL,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIRECTORY,
                        help="Directory to store results")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    return parser.parse_args()


class VOCColorize(object):
    def __init__(self, n=10):
        self.cmap = color_map(10)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.shape
        color_image = np.zeros((3, size[0], size[1]), dtype=np.uint8)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image)
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        # handle void
        mask = (255 == gray_image)
        color_image[0][mask] = color_image[1][mask] = color_image[2][mask] = 255

        return color_image


def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    cmap = np.array([[255, 0, 0], [0, 0, 243], [243, 113, 165], [194, 243, 0], [243, 0, 193],
                     [255, 159, 15], [13, 113, 243], [188, 75, 0],[0, 0, 0]])

    # for i in range(N):
    #     r = g = b = 0
    #     c = i
    #     for j in range(8):
    #         r = r | (bitget(c, 0) << 7 - j)
    #         g = g | (bitget(c, 1) << 7 - j)
    #         b = b | (bitget(c, 2) << 7 - j)
    #         c = c >> 3
    #
    #     cmap[i] = np.array([r, g, b])
    cmap = cmap / 255 if normalized else cmap
    print(cmap)
    return cmap


def get_iou(data_list, class_num, save_path=None):
    from multiprocessing import Pool
    from utils.metric import ConfusionMatrix

    ConfM = ConfusionMatrix(class_num)
    f = ConfM.generateM
    pool = Pool()
    m_list = pool.map(f, data_list)
    pool.close()
    pool.join()

    for m in m_list:
        ConfM.addM(m)

    aveJ, j_list, M = ConfM.jaccard()  # 去掉背景的平均IOU
    Recall = ConfM.recall()
    PA = ConfM.PA()
    MPA = ConfM.accuracy()
    F1 = (2 * PA * Recall) / (PA + Recall)
    FWIOU, FW = ConfM.FWIOU()
    classes = np.array(('background', '箱体', '圆柱直齿轮', '轴承', '轴承端盖',
                        '轴', '圆柱直齿齿轮轴', '圆锥斜齿轮', '轴套'))
    # classes = np.array(('background', 'xiangti', 'yuanzhuzhichilun', 'zhoucheng', 'zhoucduangai',
    #                     'zhou', 's', 'A', 'B', 'C', 'D', 'E',
    #                     'F', 'H', 'I','G', 'J'))
    print(j_list)
    print(FW)
    for i, iou in enumerate(j_list):
        print('class {:2d} {:12} IU {:.2f}'.format(i, classes[i], j_list[i]))
    print('meanIOU(remove the background): ' + str(aveJ))
    print('Recall: ' + str(Recall))
    print('PA: ' + str(PA))
    print('MPA: ' + str(MPA))
    print('F1: ' + str(F1))
    print('FWIOU: ' + str(FWIOU))
    if save_path:
        with open(save_path, 'w') as f:
            for i, iou in enumerate(j_list):
                f.write('class {:2d} {:12} IU {:.2f}'.format(i, classes[i], j_list[i]) + '\n')
            f.write('meanIOU(remove the background): ' + str(aveJ) + '\n')
            f.write('recall: ' + str(Recall) + '\n')
            f.write('PA: ' + str(PA) + '\n')
            f.write('MPA: ' + str(MPA) + '\n')
            f.write('F1: ' + str(F1) + '\n')
            f.write('FWIOU: ' + str(FWIOU) + '\n')


# def show_all(gt, pred):
#     import matplotlib.pyplot as plt
#     from matplotlib import colors
#     fig, axes = plt.subplots(1, 2)
#     ax1, ax2 = axes
#     colormap = [(0, 0, 0), (255, 0, 0), (0, 0, 243), (243, 113, 165), (194, 243, 0), (243, 0, 193), (255, 159, 15),
#                 (13, 113, 243), (188, 75, 0), (0, 188, 0), (100, 50, 100), (96, 128, 255), (0, 243, 97), (43, 0, 215),
#                 (96, 255, 223), (64, 191, 175), (239, 235, 143)]
#     cmap = colors.ListedColormap(colormap)
#     bounds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
#     norm = colors.BoundaryNorm(bounds, cmap.N)
#     ax1.set_title('gt')
#     ax1.imshow(gt, cmap=cmap, norm=norm)
#     ax2.set_title('pred')
#     ax2.imshow(pred, cmap=cmap, norm=norm)
#     plt.show()


def main():

    """Create the model and start the evaluation process."""
    args = get_arguments()

    gpu0 = args.gpu

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    model = Res_Deeplab(num_classes=args.num_classes)

    if args.pretrained_model != None:
        args.restore_from = pretrianed_models_dict[args.pretrained_model]

    if args.restore_from[:4] == 'http':
        saved_state_dict = model_zoo.load_url(args.restore_from)
    else:
        saved_state_dict = torch.load(args.restore_from, map_location='cuda:0')
    model.load_state_dict(saved_state_dict)

    model.eval()
    model.cuda(gpu0)
    start_time = time.time()
    testloader = data.DataLoader(
        VOCDataSet(args.data_dir, args.data_list, crop_size=(512, 512), mean=IMG_MEAN, scale=False, mirror=False),
        batch_size=1, shuffle=False, pin_memory=True)

    if version.parse(torch.__version__) >= version.parse('0.4.0'):
        interp = nn.Upsample(size=(512, 512), mode='bilinear', align_corners=True)
    else:
        interp = nn.Upsample(size=(512, 512), mode='bilinear')
    data_list = []

    colorize = VOCColorize()

    for index, batch in enumerate(testloader):
        if index % 100 == 0:
            print('%d processd' % (index))
        image, label, size, name = batch
        size = size[0].numpy()
        output = model(Variable(image, volatile=True).cuda(gpu0))
        output = interp(output).cpu().data[0].numpy()
        output = output[:, :size[0], :size[1]]
        gt = np.asarray(label[0].numpy()[:size[0], :size[1]], dtype=np.int)

        output = output.transpose(1, 2, 0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.int)

        filename = os.path.join(args.save_dir, '{}.png'.format(name[0]))
        color_file = Image.fromarray(colorize(output).transpose(1, 2, 0), 'RGB')
        color_file.save(filename)

        # show_all(gt, output)

        data_list.append([gt.flatten(), output.flatten()])
    end_time = time.time()
    filename = os.path.join(args.save_dir, 'result.txt')

    get_iou(data_list, args.num_classes, filename)

    print(f"the time is:{end_time-start_time}s")


if __name__ == '__main__':
    main()
