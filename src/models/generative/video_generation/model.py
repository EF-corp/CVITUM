import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import (Tuple, 
                    Union)

import numpy as np
import math


class ConvLSTMCell(nn.Module):

    def __init__(self, 
                 in_channels:int, 
                 out_channels:int, 
    
                 kernel_size:Tuple[int], 
                 padding:int, 
                 activation:str, 
                 frame_size:Tuple[int],
                 **kwargs) -> None:

        super().__init__()  

        if activation == "tanh":
            self.activation = torch.tanh 
        elif activation == "relu":
            self.activation = torch.relu
        

        self.conv = nn.Conv2d(
            in_channels=in_channels+out_channels, 
            out_channels=4*out_channels, 
            kernel_size=kernel_size, 
            padding=padding)           


        self.W_ci = nn.Parameter(torch.Tensor(out_channels, *frame_size))
        self.W_co = nn.Parameter(torch.Tensor(out_channels, *frame_size))
        self.W_cf = nn.Parameter(torch.Tensor(out_channels, *frame_size))

    def forward(self, X, H_prev, C_prev):

        conv_output = self.conv(torch.cat([X, H_prev], dim=1))


        i_conv, f_conv, C_conv, o_conv = torch.chunk(conv_output, chunks=4, dim=1)

        input_gate = torch.sigmoid(i_conv + self.W_ci * C_prev )
        forget_gate = torch.sigmoid(f_conv + self.W_cf * C_prev )


        C = forget_gate*C_prev + input_gate * self.activation(C_conv)

        output_gate = torch.sigmoid(o_conv + self.W_co * C )


        H = output_gate * self.activation(C)

        return H, C
    

class ConvLSTM(nn.Module):

    def __init__(self, 
                 in_channels:int, 
                 out_channels:int, 
                 kernel_size:Tuple[int], 
                 padding:int, 
                 activation:str, 
                 frame_size:Tuple[int],
                 device:Union[str, torch.device],
                 **kwargs) -> None:

        super().__init__()

        self.out_channels = out_channels
        self.device = device

        self.convLSTMcell = ConvLSTMCell(in_channels=in_channels, 
                                         out_channels=out_channels, 
                                         kernel_size=kernel_size, 
                                         padding=padding, 
                                         activation=activation, 
                                         frame_size=frame_size)

    def forward(self, X):

        batch_size, _, seq_len, height, width = X.size()


        output = torch.zeros(batch_size, self.out_channels, seq_len, 
        height, width, device=self.device)
        

        H = torch.zeros(batch_size, self.out_channels, 
        height, width, device=self.device)


        C = torch.zeros(batch_size,self.out_channels, 
        height, width, device=self.device)


        for time_step in range(seq_len):

            H, C = self.convLSTMcell(X[:,:,time_step], H, C)

            output[:,:,time_step] = H

        return output

class ConvLSTMSeq2Seq(nn.Module):

    def __init__(self, 
                 num_channels:int, 
                 num_kernels:int, 
                 kernel_size:Tuple[int], 
                 padding:int, 
                 activation:str, 
                 frame_size:Tuple[int], 
                 num_layers:int,
                 device:Union[str, torch.device],
                 **kwargs) -> None:

        super().__init__()

        self.sequential = nn.Sequential()


        self.sequential.add_module(
            "convlstm1", ConvLSTM(
                in_channels=num_channels, 
                out_channels=num_kernels,
                kernel_size=kernel_size, 
                padding=padding, 
                activation=activation, 
                frame_size=frame_size,
                device=device)
        )

        self.sequential.add_module(
            "batchnorm1", nn.BatchNorm3d(num_features=num_kernels)
        ) 


        for l in range(2, num_layers+1):

            self.sequential.add_module(
                f"convlstm{l}", ConvLSTM(
                    in_channels=num_kernels, 
                    out_channels=num_kernels,
                    kernel_size=kernel_size, 
                    padding=padding, 
                    activation=activation, 
                    frame_size=frame_size,
                    device=device)
                )
                
            self.sequential.add_module(
                f"batchnorm{l}", nn.BatchNorm3d(num_features=num_kernels)
                ) 

        self.conv = nn.Conv2d(
            in_channels=num_kernels, 
            out_channels=num_channels,
            kernel_size=kernel_size, 
            padding=padding)

    def forward(self, X):

        output = self.sequential(X)

        output = self.conv(output[:,:,-1])
        
        return nn.Sigmoid()(output)



class ConvLSTMCellV2(nn.Module):

    def __init__(self, 
                 input_dim:int, 
                 hidden_dim:int, 
                 kernel_size:Tuple[int], 
                 bias:bool) -> None:

        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):

        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


    def init_hidden(self, batch_size, image_size):

        height, width = image_size

        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))
    

class ConvLSTMSeq2SeqV2(nn.Module):
    def __init__(self, 
                 nf:int, 
                 in_channels:int) -> None:
        super().__init__()

        self.encoder_1_convlstm = ConvLSTMCell(input_dim=in_channels,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.encoder_2_convlstm = ConvLSTMCell(input_dim=nf,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.decoder_1_convlstm = ConvLSTMCell(input_dim=nf,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.decoder_2_convlstm = ConvLSTMCell(input_dim=nf,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.decoder_CNN = nn.Conv3d(in_channels=nf,
                                     out_channels=1,
                                     kernel_size=(1, 3, 3),
                                     padding=(0, 1, 1))


    def autoencoder(self, x, seq_len, future_step, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4) -> torch.tensor:

        outputs = []

        # encoder
        for t in range(seq_len):
            h_t, c_t = self.encoder_1_convlstm(input_tensor=x[:, t, :, :],
                                               cur_state=[h_t, c_t])
            h_t2, c_t2 = self.encoder_2_convlstm(input_tensor=h_t,
                                                 cur_state=[h_t2, c_t2])

        encoder_vector = h_t2

        #decoder
        for t in range(future_step):
            h_t3, c_t3 = self.decoder_1_convlstm(input_tensor=encoder_vector,
                                                 cur_state=[h_t3, c_t3])  
            h_t4, c_t4 = self.decoder_2_convlstm(input_tensor=h_t3,
                                                 cur_state=[h_t4, c_t4])  
            encoder_vector = h_t4
            outputs += [h_t4] 

        outputs = torch.stack(outputs, 1)
        outputs = outputs.permute(0, 2, 1, 3, 4)
        outputs = self.decoder_CNN(outputs)
        outputs = torch.nn.Sigmoid()(outputs)

        return outputs

    def forward(self, x, future_seq=0, hidden_state=None):

        b, seq_len, _, h, w = x.size()


        h_t, c_t = self.encoder_1_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t2, c_t2 = self.encoder_2_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t3, c_t3 = self.decoder_1_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t4, c_t4 = self.decoder_2_convlstm.init_hidden(batch_size=b, image_size=(h, w))


        outputs = self.autoencoder(x, seq_len, future_seq, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4)

        return outputs