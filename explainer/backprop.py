import numpy as np
from torch.autograd import Variable, Function
import torch
import types


class VanillaGradExplainer(object):
    def __init__(self, model, explain_input_getter, explain_output_getter=None):
        self.model = model
        # explanations are calculated with respect to this input
        self.explain_input_getter = explain_input_getter
        self.explain_output_getter = explain_output_getter or (lambda x: x)

    def _backprop(self, ind, **inputs):
        self.input_explain = self.explain_input_getter(inputs, requires_grad=True)
        #inputs[self.explain_input_setter] = Variable(inputs[self.explain_input_setter], requires_grad=True)
        #input_explain = inputs[self.explain_input_setter]

        output_original = self.model(**inputs)
        output = self.explain_output_getter(output_original)

        grad_out = output.data.clone()
        grad_out.fill_(0.0)
        if ind is None:
            ind = output.data.max(dim=-1)[1]
            ind = ind.unsqueeze(-1)
            grad_out.scatter_(-1, ind, 1.0)
        else:
            # if ind is a tuple, the second element is the token index
            if isinstance(ind, tuple):
                if len(ind) == 2:
                    _grad_out = torch.zeros_like(output.data)
                    _grad_out.scatter_(-1, ind[0].unsqueeze(-1), 1.0)
                    grad_out[:, ind[1]] = _grad_out[:, ind[1]]
                else:
                    raise NotImplementedError('index with len(shape) > 2 not implemented')
            else:
                ind = ind.unsqueeze(-1)
                grad_out.scatter_(-1, ind, 1.0)

        output.backward(grad_out)
        return self.input_explain.grad.data, output_original

    def explain(self, ind=None, **inputs):
        return self._backprop(ind, **inputs)


class GradxInputExplainer(VanillaGradExplainer):
    def __init__(self, model, **kwargs):
        super(GradxInputExplainer, self).__init__(model, **kwargs)

    def explain(self, ind=None, **inputs):
        grad, out = self._backprop(ind, **inputs)
        return self.input_explain.data * grad, out


class SaliencyExplainer(VanillaGradExplainer):
    def __init__(self, model, **kwargs):
        super(SaliencyExplainer, self).__init__(model, **kwargs)

    def explain(self, ind=None, **inputs):
        grad, out = self._backprop(ind=ind, **inputs)
        return grad.abs(), out


class IntegrateGradExplainer(VanillaGradExplainer):
    def __init__(self, model, steps=100, **kwargs):
        super(IntegrateGradExplainer, self).__init__(model, **kwargs)
        self.steps = steps

    def explain(self, ind=None, **inputs):
        grad = 0
        #inp_data = inputs[self.explain_input_setter].data.clone()
        inp_data = self.explain_input_getter(inputs).data.clone()

        for alpha in np.arange(1 / self.steps, 1.0, 1 / self.steps):
            new_inp = Variable(inp_data * alpha, requires_grad=True)
            inputs[self.explain_input_getter] = new_inp
            g, out = self._backprop(ind=ind, **inputs)
            grad += g

        return grad * inp_data / self.steps, out


class DeconvExplainer(VanillaGradExplainer):
    def __init__(self, model):
        raise NotImplementedError('adapt changes from VanillaGradExplainer (see GradxInputExplainer)')
        super(DeconvExplainer, self).__init__(model)
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
        raise NotImplementedError('adapt changes from VanillaGradExplainer (see GradxInputExplainer)')
        super(GuidedBackpropExplainer, self).__init__(model)
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

    def explain(self,  ind=None, **inputs):
        inp = inputs[self.base_explainer.explain_input]
        stdev = self.stdev_spread * (inp.data.max() - inp.data.min())

        total_gradients = 0
        origin_inp_data = inp.data.clone()

        for i in range(self.nsamples):
            noise = torch.randn(inp.size())
            if torch.cuda.is_available():
                noise = noise.cuda()
            noise = noise * stdev
            inp.data.copy_(noise + origin_inp_data)
            inputs[self.base_explainer.explain_input] = inp
            grad, out = self.base_explainer.explain(ind=ind, **inputs)

            if self.magnitude:
                total_gradients += grad ** 2
            else:
                total_gradients += grad

        return total_gradients / self.nsamples, None