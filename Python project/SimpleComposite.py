import torch


def f(x, y):
    x = torch.tensor(x, requires_grad=True)
    y = torch.tensor(y, requires_grad=True)

    f_1 = torch.sin(x * y)
    f_2 = torch.log(x * y)

    f_1.backward()

    return (f_1, f_2, x.grad, y.grad)
