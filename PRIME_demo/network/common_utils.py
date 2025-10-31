import torch
from einops import rearrange

def down_sample(x, D):
    b, _, h, w = x.shape
    D = D.repeat(b, 1, 1)
    x_d = D @ rearrange(x, "b c h w -> b c (h w)")
    x_d = rearrange(x_d, "b c (h w) -> b c h w", h=h, w=w)
    return x_d

def spectral_TV(x):
    diff = torch.abs(x[:, :-1, :, :] - x[:, 1:, :, :])
    loss = torch.sum(diff)
    return loss

def get_params(opt_over, net, net_input, downsampler=None):
    '''Returns parameters that we want to optimize over.

    Args:
        opt_over: comma separated list, e.g. "net,input" or "net"
        net: network
        net_input: torch.Tensor that stores input `z`
    '''
    opt_over_list = opt_over.split(',')
    params = []
    
    for opt in opt_over_list:
    
        if opt == 'net':
            params += [x for x in net.parameters() ]
        elif  opt=='down':
            assert downsampler is not None
            params = [x for x in downsampler.parameters()]
        elif opt == 'input':
            net_input.requires_grad = True
            params += [net_input]
        else:
            assert False, 'what is it?'
            
    return params

def optimize(optimizer_type, parameters, closure, LR, num_iter):
    """Runs optimization loop.

    Args:
        optimizer_type: 'LBFGS' of 'adam'
        parameters: list of Tensors to optimize over
        closure: function, that returns loss variable
        LR: learning rate
        num_iter: number of iterations 
    """
    if optimizer_type == 'LBFGS':
        # Do several steps with adam first
        optimizer = torch.optim.Adam(parameters, lr=0.001)
        for j in range(100):
            optimizer.zero_grad()
            closure()
            optimizer.step()

        print('Starting optimization with LBFGS')        
        def closure2():
            optimizer.zero_grad()
            return closure()
        optimizer = torch.optim.LBFGS(parameters, max_iter=num_iter, lr=LR, tolerance_grad=-1, tolerance_change=-1)
        optimizer.step(closure2)

    elif optimizer_type == 'adam':
        optimizer = torch.optim.Adam(parameters, lr=LR)
        for j in range(num_iter):
            optimizer.zero_grad()
            closure()
            optimizer.step()

    else:
        assert False