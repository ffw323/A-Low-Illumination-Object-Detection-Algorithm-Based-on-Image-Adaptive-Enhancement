import math
import torch
import torchvision
from models.cnn import VisionEncoder

class dDIP(torch.nn.Module):
    def __init__(self,
                 encoder_output_dim: int = 256,
                 num_of_gates: int = 7):
        super(dDIP, self).__init__()
        # Encoder Model
        self.encoder = VisionEncoder(encoder_output_dim=encoder_output_dim)
        # Changed 4096 --> 256 dimension

        # Gating Module
        self.gate_module = torch.nn.Sequential(torch.nn.Linear(encoder_output_dim, num_of_gates, bias=True))

        # White-Balance Module
        self.wb_module = torch.nn.Sequential(torch.nn.Linear(encoder_output_dim, 3, bias=True))

        # Gamma Module
        self.gamma_module = torch.nn.Sequential(torch.nn.Linear(encoder_output_dim, 1, bias=True))

        # Sharpning Module
        self.gaussian_blur = torchvision.transforms.GaussianBlur(13, sigma=(0.1, 5.0))
        self.sharpning_module = torch.nn.Sequential(torch.nn.Linear(encoder_output_dim, 1, bias=True))


        # Contrast Module
        self.contrast_module = torch.nn.Sequential(torch.nn.Linear(encoder_output_dim, 1, bias=True))

        # Contrast Module
        self.tone_module = torch.nn.Sequential(torch.nn.Linear(encoder_output_dim, 8, bias=True))

    def rgb2lum(self, img: torch.tensor):

        img = 0.27 * img[:, 0, :, :] + 0.67 * img[:, 1, :, :] + 0.06 * img[:, 2, :, :]
        return img

    def lerp(self, a: int, b: int, l: torch.tensor):
        return (1 - l.unsqueeze(2).unsqueeze(3)) * a + l.unsqueeze(2).unsqueeze(3) * b

    def dark_channel(self, x: torch.tensor):

        z = x.min(dim=1)[0].unsqueeze(1)
        return z

    def atmospheric_light(self, x: torch.tensor, dark: torch.tensor, top_k: int = 1000):

        h, w = x.shape[2], x.shape[3]
        imsz = h * w
        numpx = int(max(math.floor(imsz / top_k), 1))
        darkvec = dark.reshape(x.shape[0], imsz, 1)
        imvec = x.reshape(x.shape[0], 3, imsz).transpose(1, 2)
        indices = darkvec.argsort(1)
        indices = indices[:, imsz - numpx:imsz]
        atmsum = torch.zeros([x.shape[0], 1, 3]).cuda()
        for b in range(x.shape[0]):
            for ind in range(1, numpx):
                atmsum[b, :, :] = atmsum[b, :, :] + imvec[b, indices[b, ind], :]
        a = atmsum / numpx
        a = a.squeeze(1).unsqueeze(2).unsqueeze(3)
        return a

    def blur(self, x: torch.tensor):

        return self.gaussian_blur(x)

    def white_balance(self, x: torch.tensor, latent_out: torch.tensor, wb_gate: torch.tensor):

        log_wb_range = 0.5
        wb = self.wb_module(latent_out)
        wb = torch.exp(self.tanh_range(wb, -log_wb_range, log_wb_range))

        color_scaling = 1. / (1e-5 + 0.27 * wb[:, 0] + 0.67 * wb[:, 1] +
                              0.06 * wb[:, 2])
        wb = color_scaling.unsqueeze(1) * wb
        wb_out = wb.unsqueeze(2).unsqueeze(3) * x
        wb_out = (wb_out - wb_out.min()) / (wb_out.max() - wb_out.min())
        wb_out = wb_gate.unsqueeze(1).unsqueeze(2).unsqueeze(3) * wb_out
        return wb_out

    def tanh01(self, x: torch.tensor):

        return torch.tanh(x) * 0.5 + 0.5

    def tanh_range(self, x: torch.tensor, left: float, right: float):

        return self.tanh01(x) * (right - left) + left

    def gamma_balance(self, x: torch.tensor, latent_out: torch.tensor, gamma_gate: torch.tensor):

        log_gamma = torch.log(torch.tensor(2.5))
        gamma = self.gamma_module(latent_out).unsqueeze(2).unsqueeze(3)
        gamma = torch.exp(self.tanh_range(gamma, -log_gamma, log_gamma))
        g = torch.pow(torch.maximum(x, torch.tensor(1e-4)), gamma)
        g = (g - g.min()) / (g.max() - g.min())
        g = g * gamma_gate.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        return g

    def sharpning(self, x: torch.tensor, latent_out: torch.tensor, sharpning_gate: torch.tensor):

        out_x = self.blur(x)
        y = self.sharpning_module(latent_out).unsqueeze(2).unsqueeze(3)
        y = self.tanh_range(y, torch.tensor(0.1), torch.tensor(1.))
        s = x + (y * (x - out_x))
        s = (s - s.min()) / (s.max() - s.min())
        s = s * (sharpning_gate.unsqueeze(1).unsqueeze(2).unsqueeze(3))
        return s

    def contrast(self, x: torch.tensor, latent_out: torch.tensor, contrast_gate: torch.tensor):

        alpha = torch.tanh(self.contrast_module(latent_out))
        luminance = torch.minimum(torch.maximum(self.rgb2lum(x), torch.tensor(0.0)), torch.tensor(1.0)).unsqueeze(1)
        contrast_lum = -torch.cos(math.pi * luminance) * 0.5 + 0.5
        contrast_image = x / (luminance + 1e-6) * contrast_lum
        contrast_image = self.lerp(x, contrast_image, alpha)
        contrast_image = (contrast_image - contrast_image.min()) / (contrast_image.max() - contrast_image.min())
        contrast_image = contrast_image * contrast_gate.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        return contrast_image


    def forward(self, x: torch.Tensor):
        latent_out = self.encoder(x)
        gate = self.tanh_range(self.gate_module(latent_out), 0.01, 1.0)
        wb_out = self.white_balance(x, latent_out, gate[:, 0])
        torch.save(wb_out, './wb.pt')
        gamma_out = self.gamma_balance(x, latent_out, gate[:, 1])
        torch.save(gamma_out, './gamma.pt')
        sharping_out = self.sharpning(x, latent_out, gate[:, 3])
        torch.save(sharping_out, './sharping.pt')
        #fog_out = self.defog(x, latent_out, gate[:, 4])
        contrast_out = self.contrast(x, latent_out, gate[:, 4])
        torch.save(contrast_out, './contrast.pt')
        x = wb_out + gamma_out + sharping_out + contrast_out
        torch.save(x, './x.pt')
        x = (x - x.min()) / (x.max() - x.min())
        return x


