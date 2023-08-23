import torch 

class ThirdOrderLegendrePolynomial(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, _x):
        ctx.save_for_backward(_x)
        return 0.5*(5*_x**3 - 3*_x)

    @staticmethod
    def backward(ctx, grad_output):
        _x, = ctx.saved_tensors
        return grad_output * 0.5 * (15*_x**2 - 3)


class FifthOrderLegendrePolynomial(torch.autograd.Function):

    @staticmethod
    def forward(ctx, _x):
        ctx.save_for_backward(_x)
        return 0.125*(63*_x**5 - 70*_x**3 + 15*_x)

    @staticmethod
    def backward(ctx, gard_output):
        _x, = ctx.saved_tensors
        return gard_output * 0.125*(315*_x**4 - 210*_x**2 + 15)