import torch
import torch.nn as nn
import pennylane as qml
from einops import rearrange

dev = qml.device("default.qubit", wires=4)

def QNN(embedding, p, cp):
    """
    embedding (array[float], batch x dim_embedding): the input data embedding
    p  (array[float], 3 x num_layer x num_qubit): the quantum training parameters
    cp (array[float], 1 x num_layer x num_qubit/2)
    """
    measure_set = [0, 1]
    group = [0, 1, 2, 3]

    embedding = embedding.unsqueeze(1) if len(embedding.shape) == 1 else embedding

    # group the channels
    qml.AngleEmbedding(embedding, wires=range(4), rotation="Y")

    qml.AngleEmbedding(p[0, 0, :], wires=range(4), rotation="Y")

    qml.IsingXX(cp[0, group[0]], wires=[group[0], group[1]])
    qml.IsingXX(cp[0, group[1]], wires=[group[2], group[3]])

    qml.AngleEmbedding(p[1, 0, :], wires=range(4), rotation="X")

    qml.IsingXX(cp[0, group[2]], wires=[group[1], group[2]])
    qml.IsingXX(cp[0, group[3]], wires=[group[0], group[3]])

    qml.AngleEmbedding(p[2, 0, :], wires=range(4), rotation="Y")

    qml.MultiControlledX(
        control_wires=[group[0], group[1]], wires=group[2], control_values="10"
    )
    qml.MultiControlledX(
        control_wires=[group[1], group[2]], wires=group[3], control_values="10"
    )
    qml.MultiControlledX(
        control_wires=[group[2], group[3]], wires=group[0], control_values="10"
    )
    qml.MultiControlledX(
        control_wires=[group[3], group[0]], wires=group[1], control_values="10"
    )
    exp_vals_z = [qml.expval(qml.PauliZ(w)) for w in measure_set]
    return exp_vals_z


class Model(nn.Module):

    def __init__(self, in_channel=4, out_channel=8):
        super(Model, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        # define qnode with its parameters
        torch.manual_seed(3)
        self.p = nn.Parameter(torch.rand((3, 1, 4)) * torch.pi, True)  
        self.cp = nn.Parameter(torch.zeros((1, 4)) * torch.pi, True) 
        self.qnn = qml.QNode(QNN, dev, "torch", diff_method="backprop")

        # module
        self.down1 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channel,  # 4
                out_channels=self.in_channel * 2,  # 8
                kernel_size=3,
                stride=1,
                padding=1,
            ),  
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.in_channel * 2, self.in_channel * 2, 3),  
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.in_channel * 2, self.in_channel * 2, 3),  
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2, stride=2), 
            nn.Conv2d(self.in_channel * 2, self.in_channel * 2, 3),  
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.in_channel * 2, self.in_channel * 2, 3),  
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.in_channel * 2, self.in_channel * 2, 3),  
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2, stride=2), 
            nn.Conv2d(self.in_channel * 2, self.in_channel * 2, 3),  
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.in_channel * 2, self.in_channel * 2, 3),  
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.in_channel * 2, self.in_channel * 2, 3),  
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(
                self.in_channel * 1,
                self.in_channel * 2,
                kernel_size=3,
                stride=1,
            ),  
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(
                self.in_channel * 2, self.in_channel * 2, kernel_size=3, stride=1
            ),  
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(
                self.in_channel * 2, self.in_channel * 2, kernel_size=3, stride=1
            ),  
            nn.LeakyReLU(0.2),
            nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            ), 
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(
                self.in_channel * 2 + self.in_channel * 2,
                self.in_channel * 2,
                kernel_size=3,
                stride=1,
            ),  
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(
                self.in_channel * 2, self.in_channel * 2, kernel_size=3, stride=1
            ), 
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(
                self.in_channel * 2, self.in_channel * 2, kernel_size=3, stride=1
            ),  
            nn.LeakyReLU(0.2),
            nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            ),  
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(
                self.in_channel * 2 + self.in_channel * 2,
                self.in_channel * 2,
                kernel_size=3,
                stride=1,
            ), 
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(
                self.in_channel * 2, self.out_channel-self.in_channel, kernel_size=3, stride=1
            ),  
            nn.LeakyReLU(0.2),
        )
        self.linear = nn.Sequential(nn.Linear(8, 4))
        self.sig = nn.Sequential(nn.Sigmoid())

    def mp_qnn(self, e, p, cp):
        tem = self.qnn(e, p, cp)
        return tem

    def forward(self, x):
        xb, xc, xh, xw = x.shape
        x1 = self.down1(x)  
        x2 = self.down2(x1)
        x3 = self.down3(x2)  

        # =========================Quantum===============================
        b, c, h, w = x3.shape
        num_qnn_groups = self.in_channel * 2 // 4
        xq = rearrange(x3, "b (c c1) h w -> (b c h w) c1 ", c1 = self.in_channel * 2)
    
        res_list = []
        for i in range(num_qnn_groups):
            start = i * 4
            end = (i + 1) * 4

            res_i = self.mp_qnn(
                xq[:, start:end],         
                self.p[:, :, 0:4],        
                self.cp[:, 0:4]           
            )
            
            res_i = torch.stack(res_i).type(torch.FloatTensor).permute(1, 0)
            res_list.append(res_i)

        res = torch.cat(res_list, dim=1)
        xq = rearrange(res, "(b c1 h w) c -> b (c1 c) h w", b=b, c1=c // (self.in_channel * 2), h=h, w=w)
        # ===============================================================

        x_up3 = self.up3(xq)
        xx = torch.cat([x_up3, x2], dim=1)
        x_up2 = self.up2(xx)
        xx = torch.cat([x_up2, x1], dim=1)
        x_generated = self.up1(xx)

        generated_bands = x_generated.shape[1]
        total_bands = generated_bands + self.in_channel
        group_size = generated_bands // self.in_channel  

        x_sum_bands = x_generated.view(x_generated.shape[0], self.in_channel, group_size, x_generated.shape[2], x_generated.shape[3]).sum(dim=2)
        
        x_remain = x - x_sum_bands

        x_out = torch.empty(1, total_bands, xh, xw, requires_grad=True)
        
        x_temp = x_out.clone()
        step = total_bands //  self.in_channel
        remain_idx = torch.arange(step-1, total_bands, step=step)[:self.in_channel]
        generated_idx = [i for i in range(total_bands) if i not in remain_idx]
        
        x_temp[:, remain_idx, :, :] = x_remain
        x_temp[:, generated_idx, :, :] = x_generated

        x_out = x_temp
        x_out = torch.clamp(x_out, min=0).cuda()

        return x_out

if __name__ == "__main__":
    print("here")
