import torch.nn as nn
import math
import torch


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def make_stages(cfg_dict):
    """Builds CPM stages from a dictionary
    Args:
        cfg_dict: a dictionary
    """
    layers = []
    for i in range(len(cfg_dict) - 1):
        one_ = cfg_dict[i]
        for k, v in one_.items():
            if 'pool' in k:
                layers += [nn.MaxPool2d(kernel_size=v[0], stride=v[1],
                                        padding=v[2])]
            else:
                conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                                   kernel_size=v[2], stride=v[3],
                                   padding=v[4])
                layers += [conv2d, nn.ReLU(inplace=True)]
    one_ = list(cfg_dict[-1].keys())
    k = one_[0]
    v = cfg_dict[-1][k]
    conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                       kernel_size=v[2], stride=v[3], padding=v[4])
    layers += [conv2d]
    return nn.Sequential(*layers)

blocks = {}

# Stage 1
blocks['block1_1'] = [{'conv5_1_CPM_L1': [128, 128, 3, 1, 1]},
                        {'conv5_2_CPM_L1': [128, 128, 3, 1, 1]},
                        {'conv5_3_CPM_L1': [128, 128, 3, 1, 1]},
                        {'conv5_4_CPM_L1': [128, 512, 1, 1, 0]},
                        {'conv5_5_CPM_L1': [512, 38, 1, 1, 0]}]

blocks['block1_2'] = [{'conv5_1_CPM_L2': [128, 128, 3, 1, 1]},
                        {'conv5_2_CPM_L2': [128, 128, 3, 1, 1]},
                        {'conv5_3_CPM_L2': [128, 128, 3, 1, 1]},
                        {'conv5_4_CPM_L2': [128, 512, 1, 1, 0]},
                        {'conv5_5_CPM_L2': [512, 19, 1, 1, 0]}]

# Stages 2 - 6
for i in range(2, 7):
    blocks['block%d_1' % i] = [
        {'Mconv1_stage%d_L1' % i: [185, 128, 7, 1, 3]},
        {'Mconv2_stage%d_L1' % i: [128, 128, 7, 1, 3]},
        {'Mconv3_stage%d_L1' % i: [128, 128, 7, 1, 3]},
        {'Mconv4_stage%d_L1' % i: [128, 128, 7, 1, 3]},
        {'Mconv5_stage%d_L1' % i: [128, 128, 7, 1, 3]},
        {'Mconv6_stage%d_L1' % i: [128, 128, 1, 1, 0]},
        {'Mconv7_stage%d_L1' % i: [128, 38, 1, 1, 0]}
    ]

    blocks['block%d_2' % i] = [
        {'Mconv1_stage%d_L2' % i: [185, 128, 7, 1, 3]},
        {'Mconv2_stage%d_L2' % i: [128, 128, 7, 1, 3]},
        {'Mconv3_stage%d_L2' % i: [128, 128, 7, 1, 3]},
        {'Mconv4_stage%d_L2' % i: [128, 128, 7, 1, 3]},
        {'Mconv5_stage%d_L2' % i: [128, 128, 7, 1, 3]},
        {'Mconv6_stage%d_L2' % i: [128, 128, 1, 1, 0]},
        {'Mconv7_stage%d_L2' % i: [128, 19, 1, 1, 0]}
    ]

model_dict = {}
for k, v in blocks.items():
    model_dict[k] = make_stages(list(v))

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

def get_model():

    class MobileNetV2(nn.Module):
        def __init__(self, model_dict, n_class=1000, input_size=368, width_mult=1.):
            super(MobileNetV2, self).__init__()
            block = InvertedResidual
            input_channel = 32
            last_channel = 128
            interverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

            # building first layer
            #assert input_size % 32 == 0
            input_channel = int(input_channel * width_mult)
            self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
            self.features = [conv_bn(3, input_channel, 2)]
            # building inverted residual blocks
            for t, c, n, s in interverted_residual_setting:
                output_channel = int(c * width_mult)
                for i in range(n):
                    if i == 0:
                        self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                    else:
                        self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                    input_channel = output_channel
            # building last several layers
            self.features.append(conv_1x1_bn(input_channel, self.last_channel))
            # make it nn.Sequential
            self.features = nn.Sequential(*self.features)

            # building classifier
            self.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(self.last_channel, n_class),
            )

            self.model1_1 = model_dict['block1_1']
            self.model2_1 = model_dict['block2_1']
            self.model3_1 = model_dict['block3_1']
            self.model4_1 = model_dict['block4_1']
            self.model5_1 = model_dict['block5_1']
            self.model6_1 = model_dict['block6_1']

            self.model1_2 = model_dict['block1_2']
            self.model2_2 = model_dict['block2_2']
            self.model3_2 = model_dict['block3_2']
            self.model4_2 = model_dict['block4_2']
            self.model5_2 = model_dict['block5_2']
            self.model6_2 = model_dict['block6_2']

            self._initialize_weights()


        def forward(self, x):
            saved_for_loss = []
            x = self.features(x)
            out1 = x

            out1_1 = self.model1_1(out1)
            out1_2 = self.model1_2(out1)
            out2 = torch.cat([out1_1, out1_2, out1], 1)
            saved_for_loss.append(out1_1)
            saved_for_loss.append(out1_2)

            out2_1 = self.model2_1(out2)
            out2_2 = self.model2_2(out2)
            out3 = torch.cat([out2_1, out2_2, out1], 1)
            saved_for_loss.append(out2_1)
            saved_for_loss.append(out2_2)

            out3_1 = self.model3_1(out3)
            out3_2 = self.model3_2(out3)
            out4 = torch.cat([out3_1, out3_2, out1], 1)
            saved_for_loss.append(out3_1)
            saved_for_loss.append(out3_2)

            out4_1 = self.model4_1(out4)
            out4_2 = self.model4_2(out4)
            out5 = torch.cat([out4_1, out4_2, out1], 1)
            saved_for_loss.append(out4_1)
            saved_for_loss.append(out4_2)

            out5_1 = self.model5_1(out5)
            out5_2 = self.model5_2(out5)
            out6 = torch.cat([out5_1, out5_2, out1], 1)
            saved_for_loss.append(out5_1)
            saved_for_loss.append(out5_2)

            out6_1 = self.model6_1(out6)
            out6_2 = self.model6_2(out6)
            saved_for_loss.append(out6_1)
            saved_for_loss.append(out6_2)

            return (out6_1, out6_2), saved_for_loss


        def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    n = m.weight.size(1)
                    m.weight.data.normal_(0, 0.01)
                    m.bias.data.zero_()

    model = MobileNetV2(model_dict)
    return model
