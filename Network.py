import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------
# 1. 论文自定义激活函数
# --------------------------
class HardSwish(nn.Module):
    """MobileNetV3专用激活函数: h-swish(x) = x * h-sigmoid(x)"""
    def __init__(self, inplace=True):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.

class HardSigmoid(nn.Module):
    """轻量化sigmoid: h-sigmoid(x) = relu6(x+3)/6"""
    def __init__(self, inplace=True):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.

# --------------------------
# 2. SE注意力模块（轻量化版）
# --------------------------
# 流程：全局平均池化 → 1×1 卷积降维 → ReLU → 1×1 卷积升维 → HardSigmoid → 通道加权
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        # 通道压缩，最小通道数为4
        mid_channels = max(in_channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, 1, bias=True),
            HardSigmoid()
        )

    def forward(self, x):
        scale = self.avg_pool(x)
        scale = self.fc(scale)
        return x * scale  # 通道加权

# --------------------------
# 3. 核心模块：Bneck（倒残差结构）
# --------------------------
class Bneck(nn.Module):
    def __init__(
        self,
        in_channels,    # 输入通道
        out_channels,   # 输出通道
        expand_channels,# 升维通道数
        kernel_size,    # 深度卷积核 3/5
        stride,         # 步幅 1/2
        use_se,         # 是否使用SE
        act             # 激活函数: RE=ReLU, HS=HardSwish
    ):
        super().__init__()
        self.act = nn.ReLU(inplace=True) if act == 'RE' else HardSwish(inplace=True)
        # 残差连接条件：步幅=1 且 输入输出通道相同
        self.use_res_connect = (stride == 1 and in_channels == out_channels)

        layers = []
        # 1x1升维卷积
        if expand_channels != in_channels:
            layers.extend([
                nn.Conv2d(in_channels, expand_channels, 1, bias=False),
                nn.BatchNorm2d(expand_channels),
                self.act
            ])
        # 深度卷积（分组卷积，groups=输入通道）
        layers.extend([
            nn.Conv2d(expand_channels, expand_channels, kernel_size,
                      stride, kernel_size//2, groups=expand_channels, bias=False),
            nn.BatchNorm2d(expand_channels),
            self.act
        ])
        # SE模块
        if use_se:
            layers.append(SEBlock(expand_channels))
        # 1x1降维卷积（无激活）
        layers.extend([
            nn.Conv2d(expand_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        if self.use_res_connect:
            out = out + x  # 残差相加
        return out

# --------------------------
# 4. MobileNetV3 主网络
# --------------------------
class MobileNetV3(nn.Module):
    def __init__(self, mode='large', num_classes=1000, dropout=0.2):
        super().__init__()
        self.mode = mode.lower()
        assert self.mode in ['large', 'small'], "模式仅支持 large / small"

        #  Stem 初始卷积层（统一输入3通道→16通道）
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            HardSwish(inplace=True)
        )

        # --------------------------
        # 论文官方配置表
        # 格式：[输入, 升维, 输出, 核, 步幅, SE, 激活]
        # --------------------------
        if self.mode == 'large':
            self.bneck_cfg = [
                [16,16,16,3,1,False,'RE'],[16,64,24,3,2,False,'RE'],
                [24,72,24,3,1,False,'RE'],[24,72,40,5,2,True,'RE'],
                [40,120,40,5,1,True,'RE'],[40,120,40,5,1,True,'RE'],
                [40,240,80,3,2,False,'HS'],[80,200,80,3,1,False,'HS'],
                [80,184,80,3,1,False,'HS'],[80,184,80,3,1,False,'HS'],
                [80,480,112,3,1,True,'HS'],[112,672,112,3,1,True,'HS'],
                [112,672,160,5,2,True,'HS'],[160,960,160,5,1,True,'HS'],
                [160,960,160,5,1,True,'HS'],
            ]
            self.last_ch = 960  # 最终卷积输出通道
        else:
            self.bneck_cfg = [
                [16,16,16,3,2,True,'RE'],[16,72,24,3,2,False,'RE'],
                [24,88,24,3,1,False,'RE'],[24,96,40,5,2,True,'HS'],
                [40,240,40,5,1,True,'HS'],[40,240,40,5,1,True,'HS'],
                [40,120,48,5,1,True,'HS'],[48,144,48,5,1,True,'HS'],
                [48,288,96,5,2,True,'HS'],[96,576,96,5,1,True,'HS'],
                [96,576,96,5,1,True,'HS'],
            ]
            self.last_ch = 576

        # 堆叠Bneck模块
        self.bneck_layers = self._make_layers()

        # 尾部卷积层
        self.last_conv = nn.Sequential(
            nn.Conv2d(self.bneck_cfg[-1][2], self.last_ch, 1, bias=False),
            nn.BatchNorm2d(self.last_ch),
            HardSwish(inplace=True)
        )

        # 分类头
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(self.last_ch, 1280),
            nn.BatchNorm1d(1280),
            HardSwish(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1280, num_classes)
        )

        # 权重初始化
        self._init_weights()

    def _make_layers(self):
        layers = []
        for in_c, expand_c, out_c, k, s, se, act in self.bneck_cfg:
            layers.append(Bneck(in_c, out_c, expand_c, k, s, se, act))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.bneck_layers(x)
        x = self.last_conv(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# --------------------------
# 快捷调用函数
# --------------------------
def mobilenet_v3_large(num_classes=1000, dropout=0.2):
    return MobileNetV3(mode='large', num_classes=num_classes, dropout=dropout)

def mobilenet_v3_small(num_classes=1000, dropout=0.2):
    return MobileNetV3(mode='small', num_classes=num_classes, dropout=dropout)

# --------------------------
# 测试代码
# --------------------------
if __name__ == '__main__':
    # 初始化模型
    model_large = mobilenet_v3_large(num_classes=1000)
    model_small = mobilenet_v3_small(num_classes=1000)

    # 模拟输入：batch=2, 3通道, 224x224（论文标准输入）
    dummy_input = torch.randn(2, 3, 224, 224)

    # 前向推理
    out_large = model_large(dummy_input)
    out_small = model_small(dummy_input)

    print(f"MobileNetV3-Large 输出形状: {out_large.shape}")
    print(f"MobileNetV3-Small 输出形状: {out_small.shape}")

    # 打印参数量
    print(f"Large参数量: {sum(p.numel() for p in model_large.parameters())/1e6:.2f}M")
    print(f"Small参数量: {sum(p.numel() for p in model_small.parameters())/1e6:.2f}M")