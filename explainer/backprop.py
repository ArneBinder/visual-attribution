import numpy as np
from torch.autograd import Variable, Function
import torch
import types


def default_backwards_function(output, index):
    if index is None:
        index = output.data.max(1)[1]

    grad_out = output.data.clone()
    grad_out.fill_(0.0)
    grad_out.scatter_(1, index.unsqueeze(0).t(), 1.0)
    output.backward(grad_out)


class VanillaGradExplainer(object):
    """
     # Parameters

    model: The `torch.nn.Module` to calculate explanations for.
    grad_out_getter: function that takes the model output and an index and should call backward to produce gradients
        at the inputs. See default_backwards_function for an example.

    """
    def __init__(self, model, backward_function=None):
        self.model = model
        self.backward_function = backward_function or default_backwards_function

    def _backprop(self, args, kwargs, explain_index=None):
        for arg in tuple(args) + tuple(kwargs.values()):
            if isinstance(arg, torch.Tensor):
                try:
                    arg.requires_grad_()
                except RuntimeError:
                    pass
        output = self.model.forward(*args, **kwargs)
        self.backward_function(output=output, index=explain_index)

        #return inp.grad.data
        return [arg.grad.data if isinstance(arg, torch.Tensor) and arg.grad is not None else None for i, arg in enumerate(args)], \
               {arg_name: arg.grad.data for arg_name, arg in kwargs.items() if isinstance(arg, torch.Tensor) and arg.grad is not None}

    def explain(self, *args, explain_index=None, **kwargs):
        res = self._backprop(args, kwargs, explain_index)
        return _select_result(*res)


def _prod(list_or_dict1, list_or_dict2):
    res = {}
    keys = list_or_dict1.keys() if isinstance(list_or_dict1, dict) else range(len(list_or_dict1))
    for k in keys:
        res[k] = list_or_dict1[k] * list_or_dict2[k]
    return res


def _select_result(args, kwargs):
    if len(args) == 0 and len(kwargs) == 0:
        return None
    if len(args) == 0:
        return kwargs
    if len(kwargs) == 0:
        return args
    return args, kwargs


class GradxInputExplainer(VanillaGradExplainer):
    def __init__(self, model, backward_function=None):
        super(GradxInputExplainer, self).__init__(model, backward_function)

    def explain(self, *args, explain_index=None, **kwargs):
        arg_grads, kwarg_grads = self._backprop(args, kwargs, explain_index)
        #return inp.data * grad
        return _select_result(_prod(arg_grads, args), _prod(kwarg_grads, kwargs))


class SaliencyExplainer(VanillaGradExplainer):
    def __init__(self, model, backward_function=None):
        super(SaliencyExplainer, self).__init__(model, backward_function)

    def explain(self, *args, explain_index=None, **kwargs):
        arg_grads, kwarg_grads = self._backprop(args, kwargs, explain_index)
        #return grad.abs()
        return _select_result([v.abs() if v is not None else None for v in arg_grads], {k: v.abs() for k, v in kwarg_grads.items()})


class IntegrateGradExplainer(VanillaGradExplainer):
    def __init__(self, model, steps=100):
        super(IntegrateGradExplainer, self).__init__(model)
        raise NotImplementedError('IntegrateGradExplainer not yet adapted for multi input / output')
        self.steps = steps

    def explain(self, args, kwargs, explain_index=None):
        grad = 0
        inp_data = inp.data.clone()

        for alpha in np.arange(1 / self.steps, 1.0, 1 / self.steps):
            new_inp = Variable(inp_data * alpha, requires_grad=True)
            g = self._backprop(new_inp, explain_index)
            grad += g

        return grad * inp_data / self.steps


class DeconvExplainer(VanillaGradExplainer):
    def __init__(self, model):
        super(DeconvExplainer, self).__init__(model)
        raise NotImplementedError('DeconvExplainer not yet adapted for multi input / output')
        self._override_backward()

    def _override_backward(self):
        class _ReLU(Function):
            @staticmethod
            def forward(ctx, input):
                output = torch.clamp(input, min=0)
                return output

            @staticmethod
            def backward(ctx, grad_output):
                grad_inp = torch.clamp(grad_output, min=0)
                return grad_inp

        def new_forward(self, x):
            return _ReLU.apply(x)

        def replace(m):
            if m.__class__.__name__ == 'ReLU':
                m.forward = types.MethodType(new_forward, m)

        self.model.apply(replace)


class GuidedBackpropExplainer(VanillaGradExplainer):
    def __init__(self, model):
        super(GuidedBackpropExplainer, self).__init__(model)
        raise NotImplementedError('GuidedBackpropExplainer not yet adapted for multi input / output')
        self._override_backward()

    def _override_backward(self):
        class _ReLU(Function):
            @staticmethod
            def forward(ctx, input):
                output = torch.clamp(input, min=0)
                ctx.save_for_backward(output)
                return output

            @staticmethod
            def backward(ctx, grad_output):
                output, = ctx.saved_tensors
                mask1 = (output > 0).float()
                mask2 = (grad_output.data > 0).float()
                grad_inp = mask1 * mask2 * grad_output.data
                grad_output.data.copy_(grad_inp)
                return grad_output

        def new_forward(self, x):
            return _ReLU.apply(x)

        def replace(m):
            if m.__class__.__name__ == 'ReLU':
                m.forward = types.MethodType(new_forward, m)

        self.model.apply(replace)


# modified from https://github.com/PAIR-code/saliency/blob/master/saliency/base.py#L80
class SmoothGradExplainer(object):
    def __init__(self, base_explainer, stdev_spread=0.15,
                nsamples=25, magnitude=True):
        self.base_explainer = base_explainer
        self.stdev_spread = stdev_spread
        self.nsamples = nsamples
        self.magnitude = magnitude
        raise NotImplementedError('SmoothGradExplainer not yet adapted for multi input / output')

    def explain(self, inp, ind=None):
        stdev = self.stdev_spread * (inp.data.max() - inp.data.min())

        total_gradients = 0
        origin_inp_data = inp.data.clone()

        for i in range(self.nsamples):
            noise = torch.randn(inp.size()).cuda() * stdev
            inp.data.copy_(noise + origin_inp_data)
            grad = self.base_explainer.explain(inp, ind)

            if self.magnitude:
                total_gradients += grad ** 2
            else:
                total_gradients += grad

        return total_gradients / self.nsamples
