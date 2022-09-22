import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np

affine_par = True
'''
自动推断x-y维度的PyTorch的另一种实现
'''


class AddCoords(nn.Module):
    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)

        if self.with_r:
            rr = torch.sqrt(
                torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5,
                                                                                 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret


class CoordConv(nn.Module):
    def __init__(self, in_channels, out_channels, with_r=False, **kwargs):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        in_size = in_channels + 2
        if with_r:
            in_size += 1
        self.conv = nn.Conv2d(in_size, out_channels, **kwargs)

    def forward(self, x):
        ret = self.addcoords(x)
        ret = self.conv(ret)
        return ret


class conv_bn_relu(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(conv_bn_relu, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


def add_conv(in_ch, out_ch, ksize, stride, leaky=True):
    """
添加一个conv2d / batchnorm / leaky ReLU块。
参数:In_ch (int):卷积层输入通道数。Out_ch (int):卷积层输出通道数。Ksize (int):卷积层内核大小。Stride (int):卷积层的Stride。
返回:阶段(顺序):组成一个卷积块的顺序层。
    """
    stage = nn.Sequential()
    pad = (ksize - 1) // 2
    stage.add_module('conv', nn.Conv2d(in_channels=in_ch,
                                       out_channels=out_ch, kernel_size=ksize, stride=stride,
                                       padding=pad, bias=False))
    stage.add_module('batch_norm', nn.BatchNorm2d(out_ch))
    if leaky:
        stage.add_module('leaky', nn.LeakyReLU(0.1))
    else:
        stage.add_module('relu6', nn.ReLU6(inplace=True))
    return stage


class ASFF(nn.Module):
    def __init__(self, level, rfb=False):
        super(ASFF, self).__init__()
        self.level = level
        # 输入的三个特征层的channels, 根据实际修改
        # self.dim = [256, 128, 64]
        self.dim = [512, 256, 64]
        self.inter_dim = self.dim[self.level]
        # 每个层级三者输出通道数需要一致
        if level == 0:
            self.stride_level_1 = add_conv(self.dim[1], self.inter_dim, 3, 2)
            self.stride_level_2 = add_conv(self.dim[2], self.inter_dim, 3, 2)
            self.expand = add_conv(self.inter_dim, 512, 3, 1)
        elif level == 1:
            self.compress_level_0 = add_conv(self.dim[0], self.inter_dim, 1, 1)
            self.stride_level_2 = add_conv(self.dim[2], self.inter_dim, 3, 2)
            self.expand = add_conv(self.inter_dim, 256, 3, 1)
        elif level == 2:
            self.compress_level_0 = add_conv(self.dim[0], self.inter_dim, 1, 1)
            if self.dim[1] != self.dim[2]:
                self.compress_level_1 = add_conv(self.dim[1], self.inter_dim, 1, 1)
            self.expand = add_conv(self.inter_dim, 64, 3, 1)
        compress_c = 8 if rfb else 16  # 当添加rfb时，我们使用一半的通道数量来节省内存
        self.weight_level_0 = conv_bn_relu(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = conv_bn_relu(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = conv_bn_relu(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c * 3, 3, 1, 1, 0)

    # 尺度大小 level_0 < level_1 < level_2
    def forward(self, x_level_0, x_level_1, x_level_2):
        # Feature Resizing过程
        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_downsampled_inter = F.max_pool2d(x_level_2, 3, stride=2, padding=1)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)

        elif self.level == 1:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=2, mode='nearest')
            level_1_resized = x_level_1
            level_2_resized = self.stride_level_2(x_level_2)
        elif self.level == 2:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=4, mode='nearest')
            level_1_compressed = self.compress_level_1(x_level_1)
            level_1_resized = F.interpolate(level_1_compressed, scale_factor=2, mode='nearest')
            level_2_resized = x_level_2
        # 融合权重也是来自于网络学习
        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)  # alpha产生
        # 自适应融合
        fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] + \
                            level_1_resized * levels_weight[:, 1:2, :, :] + \
                            level_2_resized * levels_weight[:, 2:, :, :]
        out = self.expand(fused_out_reduced)
        return out


class RFB(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, visual=1):
        super(RFB, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 8
        # 分支0：1X1卷积+3X3卷积
        self.branch0 = nn.Sequential(conv_bn_relu(in_planes, 2 * inter_planes, 1, stride),
                                     conv_bn_relu(2 * inter_planes, 2 * inter_planes, 3, 1, visual, visual, relu=False))
        # 分支1：1X1卷积+3X3卷积+空洞卷积
        self.branch1 = nn.Sequential(conv_bn_relu(in_planes, inter_planes, 1, 1),
                                     conv_bn_relu(inter_planes, 2 * inter_planes, (3, 3), stride, (1, 1)),
                                     conv_bn_relu(2 * inter_planes, 2 * inter_planes, 3, 1, visual + 1, visual + 1,
                                                  relu=False))
        # 分支2：1X1卷积+3X3卷积*3代替5X5卷积+空洞卷积
        self.branch2 = nn.Sequential(conv_bn_relu(in_planes, inter_planes, 1, 1),
                                     conv_bn_relu(inter_planes, (inter_planes // 2) * 3, 3, 1, 1),
                                     conv_bn_relu((inter_planes // 2) * 3, 2 * inter_planes, 3, stride, 1),
                                     conv_bn_relu(2 * inter_planes, 2 * inter_planes, 3, 1, 2 * visual + 1,
                                                  2 * visual + 1, relu=False))
        # 分支3：1X1卷积+12的空洞卷积
        self.branch3 = nn.Sequential(conv_bn_relu(in_planes, inter_planes, 1, 1),
                                     conv_bn_relu(inter_planes, 2 * inter_planes, 3, 1, 18, 18,
                                                  relu=False))
        # 分支4：1X1卷积+18的空洞卷积
        self.branch4 = nn.Sequential(conv_bn_relu(in_planes, inter_planes, 1, 1),
                                     conv_bn_relu(inter_planes, 2 * inter_planes, 3, 1, 18, 18,
                                                  relu=False))
        self.ConvLinear = conv_bn_relu(12 * inter_planes, out_planes, 1, 1, relu=False)
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(in_planes, 2 * inter_planes, 1, 1)

    def forward(self, x):
        size = x.shape[2:]
        # 池化分支
        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image = F.upsample(image_features, size=size, mode='bilinear')
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        # 尺度融合
        out = torch.cat((image, x0, x1, x2, x3, x4), 1)
        # 1X1卷积
        out = self.ConvLinear(out)
        return out


def outS(i):
    i = int(i)
    i = (i + 1) / 2
    i = int(np.ceil((i + 1) / 2.0))
    i = (i + 1) / 2
    return i


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)  # change
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False

        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,  # change
                               padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        # self.layer5 = self._make_pred_layer(Classifier_Module, [6,12,18,24],[6,12,18,24],num_classes)
        self.layer5 = RFB(2048, 512)
        self.CoordConv1 = CoordConv(3, 3, with_r=False, kernel_size=3, stride=1, padding=1)
        self.ASFF0 = ASFF(0, rfb=True)
        self.ASFF1 = ASFF(1, rfb=True)
        self.ASFF2 = ASFF(2, rfb=True)
        self.conv1_1 = nn.Conv2d(768, 256, kernel_size=1, stride=1)
        self.conv1_2 = nn.Conv2d(320, 64, kernel_size=1, stride=1)
        self.CoordConv = nn.Conv2d(64, num_classes, kernel_size=3, stride=1, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #        for i in m.parameters():
        #            i.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def _make_pred_layer(self, block, dilation_series, padding_series, num_classes):
        return block(dilation_series, padding_series, num_classes)

    def forward(self, x):
        x = self.CoordConv1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x1 = self.relu(x)
        x = self.maxpool(x1)
        x2 = self.layer1(x)
        x = self.layer2(x2)
        x = self.layer3(x)
        x = self.layer4(x)
        x3 = self.layer5(x)
        x = self.ASFF0(x3, x2, x1)
        x = self.upsample(x)
        x2 = torch.cat([x2, x], 1)
        x2 = self.conv1_1(x2)
        x = self.ASFF1(x3, x2, x1)
        x = self.upsample(x)
        x1 = torch.cat([x1, x], 1)
        x1 = self.conv1_2(x1)
        x = self.ASFF2(x3, x2, x1)
        x = self.upsample(x)
        x = self.CoordConv(x)
        return x

    def get_1x_lr_params_NOscale(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = []

        b.append(self.conv1)
        b.append(self.bn1)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = [self.layer5, self.ASFF0, self.ASFF1, self.ASFF2, self.CoordConv,self.CoordConv1]
        for j in range(len(b)):
            for k in b[j].parameters():
                if k.requires_grad:
                    yield k

    def optim_parameters(self, args):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': args.learning_rate},
                {'params': self.get_10x_lr_params(), 'lr': 10 * args.learning_rate}]


def Res_Deeplab(num_classes=21):
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes)
    return model


if __name__ == '__main__':
    x = torch.rand(1, 3, 224, 224)
    model = Res_Deeplab(num_classes=9)
    print(model)
    model(x)
