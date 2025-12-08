import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGate(nn.Module):
    def __init__(self, x_channels, g_channels, inter_channels=None):
        super().__init__()
        if inter_channels is None:
            inter_channels = max(
                x_channels // 4, 1
            )

        self.theta = nn.Conv2d(x_channels, inter_channels, kernel_size=1, bias=False)
        self.psi = nn.Conv2d(g_channels, inter_channels, kernel_size=1, bias=False)
        self.phi = nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        theta_x = self.theta(x)
        psi_g = self.psi(g)

        if psi_g.shape[2:] != theta_x.shape[2:]:
            psi_g = F.interpolate(psi_g, size=theta_x.shape[2:], mode="bilinear")

        f = self.relu(theta_x + psi_g)
        score = self.phi(f)
        att_map = torch.sigmoid(score)

        return x * att_map, att_map


def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class AG_CNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=3, dropout_p=0.3):
        super().__init__()

        self.conv1 = conv_block(in_channels, 6)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = conv_block(6, 16)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = conv_block(16, 32)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = conv_block(32, 64)


        self.att1 = AttentionGate(32, 64, inter_channels=8)
        self.att2 = AttentionGate(16, 64, inter_channels=4)
        self.att3 = AttentionGate(6, 64, inter_channels=2)

        self.fc = nn.Sequential(
            nn.Linear(32 + 16 + 6, 64),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x1 = self.conv1(x)  # (B,6,H,W)
        p1 = self.pool1(x1)

        x2 = self.conv2(p1)  # (B,16,H,W)
        p2 = self.pool2(x2)

        x3 = self.conv3(p2)  # (B,32,H,W)
        p3 = self.pool3(x3)

        g = self.conv4(p3)  # (B,64,H,W)

        out1, _ = self.att1(x3, g)
        out2, _ = self.att2(x2, g)
        out3, _ = self.att3(x1, g)


        f1 = out1.flatten(2).sum(2)
        f2 = out2.flatten(2).sum(2)
        f3 = out3.flatten(2).sum(2)

        features = torch.cat([f1, f2, f3], dim=1)

        logits = self.fc(features)
        return logits


if __name__ == "__main__":
    input_tensor = torch.randn((8, 3, 50, 60))
    model = AG_CNN(num_classes=3)
    out = model(input_tensor)
    print("Output shape:", out.shape)
