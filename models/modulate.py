
import torch
import torch.nn as nn

class CondBN(nn.Module):
    def __init__(self, encoder, context, module):
        super(CondBN, self).__init__()

        self.encoder = encoder
        self.module = module
        self.context = context
        
    def reset_context(self, N):
        params = self.context.detach() * 0
        # import pdb; pdb.set_trace()
        if params.shape[0] != N:
            params = torch.Tensor(N, params.shape[1]).cuda()
        params.requires_grad_()
        
        self.context = params

    def forward(self, x, z=None):
        if self.module is not None:
            x = self.module(x)
        B, C = x.shape[:2]

        if z is None:
            z = self.context

        h = self.encoder(z)

        assert h.shape[1] == 2 * C, 'need scale and bias parameters per channel'

        x = x * h[:, :C] + h[:, C:]

        return x

    def update_context(self, loss):
        ctx_grad = torch.autograd.grad(loss, self.context, create_graph=True, allow_unused=True)[0]
        self.context = self.context - 0.1 * ctx_grad


class CondConv(nn.Conv2d):
    def __init__(self, encoder, *args, **kwargs):
        super(CondConv, self).__init__(*args, **kwargs)

        self.encoder = encoder
    
    def forward(self, input, z):
        x = self.conv2d_forward(input, self.weight)
        B, C, H, W = x.shape
        h = self.encoder(z)
        assert h.shape[1] == 2 * C, 'need scale and bias parameters per channel'

        # here, h should be B x C x 1 x 1
        # or                B x C x H x W

        x = x * h[:, :C] + h[:, C:]
        
        return x
