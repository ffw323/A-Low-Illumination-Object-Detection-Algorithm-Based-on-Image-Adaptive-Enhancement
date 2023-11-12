import torch


class VisionEncoder(torch.nn.Module):
    def __init__(self, encoder_output_dim=128):
        super(VisionEncoder, self).__init__()
        # conv_1
        self.conv_1 = torch.nn.Sequential(torch.nn.Conv2d(3, 64, kernel_size=3, stride=1),
                                          torch.nn.ReLU(True))
        self.max_pool_1 = torch.nn.AvgPool2d((3, 3), (2, 2))

        # conv_2
        self.conv_2 = torch.nn.Sequential(torch.nn.Conv2d(64, 128, kernel_size=3, stride=1),
                                          torch.nn.ReLU(True))
        self.max_pool_2 = torch.nn.AvgPool2d((3, 3), (2, 2))
        # conv_3
        self.conv_3 = torch.nn.Sequential(torch.nn.Conv2d(128, 256, kernel_size=3, stride=1),
                                          torch.nn.ReLU(True))
        self.adp_pool_3 = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.linear_proj_5 = torch.nn.Sequential(torch.nn.Linear(256, encoder_output_dim),
                                                 torch.nn.ReLU(True))

    def forward(self, x):
        out_x = self.conv_1(x)
        max_pool_1 = self.max_pool_1(out_x)

        out_x = self.conv_2(max_pool_1)
        max_pool_2 = self.max_pool_2(out_x)

        out_x = self.conv_3(max_pool_2)
        adp_pool_3 = self.adp_pool_3(out_x)

        linear_proj_5 = self.linear_proj_5(adp_pool_3.view(adp_pool_3.shape[0], -1))

        return linear_proj_5


if __name__ == '__main__':
    img = torch.randn(4, 3, 448, 448).cuda()
    encoder = VisionEncoder().cuda()
    print('output shape:', encoder(img).shape)  # output should be [4,256]
