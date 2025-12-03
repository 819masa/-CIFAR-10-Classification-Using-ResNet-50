# --- 5. モデル定義 (ResNet50) ---
class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.convs = nn.Sequential(*[nn.Conv2d(in_planes, planes, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(planes),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
                                     nn.BatchNorm2d(planes),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(self.expansion * planes)])

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.convs(x)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, n_class=10, n_blocks=[3, 4, 6, 3]):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False) # CIFAR用にkernelサイズ調整済み
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.res1 = self._make_layer(64, n_blocks[0], stride=1)
        self.res2 = self._make_layer(128, n_blocks[1], stride=2)
        self.res3 = self._make_layer(256, n_blocks[2], stride=2)
        self.res4 = self._make_layer(512, n_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, n_class)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(BottleNeck(self.in_planes, planes, stride))
            self.in_planes = planes * BottleNeck.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        h = self.relu(self.bn1(self.conv1(x)))
        h = self.res1(h)
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)
        h = self.avgpool(h)
        h = torch.flatten(h, 1)
        h = self.fc(h)
        return h

class ResNet50(ResNet):
    def __init__(self, n_class=10):
        super(ResNet50, self).__init__(n_class, n_blocks=[3, 4, 6, 3])


# --- ヘルパー関数 (これがあると便利) ---
def get_model(device):
    """
    他のファイルから呼び出すための関数
    """
    model = ResNet50(n_class=10)
    return model.to(device)

