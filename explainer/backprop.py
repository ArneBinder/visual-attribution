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
        for i in range(len(args)):
            if isinstance(args[i], torch.Tensor) and args[i].dtype == torch.float:
                args[i] = Variable(args[i], requires_grad=True)
        for k in kwargs.keys():
            if isinstance(kwargs[k], torch.Tensor) and kwargs[k].dtype == torch.float:
                kwargs[k] = Variable(kwargs[k], requires_grad=True)

        output = self.model.forward(*args, **kwargs)
        self.backward_function(output=output, index=explain_index)

        #return inp.grad.data
        return [arg.grad.data if isinstance(arg, torch.Tensor) and arg.grad is not None else None for i, arg in enumerate(args)], \
               {arg_name: arg.grad.data for arg_name, arg in kwargs.items() if isinstance(arg, torch.Tensor) and arg.grad is not None}

    def explain(self, *args, explain_index=None, **kwargs):
        res = self._backprop(args, kwargs, explain_index)
        return _select_result(*res, args=args, kwargs=kwargs)


def _prod(list_or_dict1, list_or_dict2):
    res = {}
    keys = list_or_dict1.keys() if isinstance(list_or_dict1, dict) else range(len(list_or_dict1))
    for k in keys:
        res[k] = list_or_dict1[k] * list_or_dict2[k]
    return res


def _select_result(args_grads, kwargs_grads, args, kwargs):
    if len(args) == 0 and len(kwargs) == 0:
        return None
    if len(args) == 0:
        return kwargs_grads
    if len(kwargs) == 0:
        return args_grads
    return args_grads, kwargs_grads


class GradxInputExplainer(VanillaGradExplainer):
    def __init__(self, model, **kwargs):
        super(GradxInputExplainer, self).__init__(model, **kwargs)

    def explain(self, *args, explain_index=None, **kwargs):
        arg_grads, kwarg_grads = self._backprop(args, kwargs, explain_index)
        #return inp.data * grad
        return _select_result(_prod(arg_grads, args), _prod(kwarg_grads, kwargs), args=args, kwargs=kwargs)


class SaliencyExplainer(VanillaGradExplainer):
    def __init__(self, model, **kwargs):
        super(SaliencyExplainer, self).__init__(model, **kwargs)

    def explain(self, *args, explain_index=None, **kwargs):
        arg_grads, kwarg_grads = self._backprop(args, kwargs, explain_index)
        #return grad.abs()
        return _select_result([v.abs() if v is not None else None for v in arg_grads],
                              {k: v.abs() for k, v in kwarg_grads.items()},
                              args=args, kwargs=kwargs)


class IntegrateGradExplainer(VanillaGradExplainer):
    def __init__(self, model, steps=100, args_indices=None, kwargs_keys=None, **kwargs):
        super(IntegrateGradExplainer, self).__init__(model, **kwargs)
        self.steps = steps
        self.args_indices = args_indices
        self.kwargs_keys = kwargs_keys

    def explain(self, *args, explain_index=None, **kwargs):
        #grad = 0
        args_grads = None
        kwargs_grads = None
        #inp_data = inp.data.clone()

        # if args_indices / kwargs_keys are note provided, compute explanations for a all FloatTensor inputs
        if self.args_indices is None:
            args_indices = [idx for idx, arg in enumerate(args) if isinstance(arg, torch.Tensor) and arg.dtype == torch.float]
        else:
            args_indices = self.args_indices
        if self.kwargs_keys is None:
            kwargs_keys = [k for k, arg in kwargs.items() if isinstance(arg, torch.Tensor) and arg.dtype == torch.float]
        else:
            kwargs_keys = self.kwargs_keys

        args = [arg.data.clone() if idx in args_indices else arg for idx, arg in enumerate(args)]
        kwargs = {k: arg.data.clone() if k in kwargs_keys else arg for k, arg in kwargs.items()}

        for alpha in np.arange(1 / self.steps, 1.0, 1 / self.steps):
            #new_inp = Variable(inp_data * alpha, requires_grad=True)
            new_args = [arg * alpha if idx in args_indices else arg for idx, arg in enumerate(args)]
            new_kwargs = {k: arg * alpha if k in kwargs_keys else arg for k, arg in kwargs.items()}
            new_args_grads, new_kwargs_grads = self._backprop(new_args, new_kwargs, explain_index=explain_index)
            #grad += g
            if args_grads is None:
                args_grads = new_args_grads
            else:
                for idx, grad in enumerate(new_args_grads):
                    if grad is not None:
                        args_grads[idx] *= grad
            if kwargs_grads is None:
                kwargs_grads = new_kwargs_grads
            else:
                for k, grad in new_kwargs_grads.items():
                    kwargs_grads[k] += grad

        #return grad * inp_data / self.steps
        args_grads = [grad * args[idx] / self.steps if grad is not None else None for idx, grad in enumerate(args_grads)]
        kwargs_grads = {k: grad * kwargs[k] / self.steps for k, grad in kwargs_grads.items()}
        return _select_result(args_grads, kwargs_grads, args=args, kwargs=kwargs)


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
