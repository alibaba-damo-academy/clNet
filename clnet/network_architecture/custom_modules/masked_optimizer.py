import torch.optim as optim
import torch

class MaskedOptimizer(optim.Optimizer):
    def __init__(self, optimizer, masks=None):
        self.optimizer = optimizer
        self.masks = masks

    def step(self):
        if self.masks is not None:
            for group in self.optimizer.param_groups:
                for i, p in enumerate(group['params']):
                    if p.grad is not None:
                        p.grad.data *= self.masks[i]  # Apply mask to gradients
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()


class MaskedAdamW(optim.AdamW):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, amsgrad=False, mask=None):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)

        # Initialize mask to ones if None is provided
        if mask is not None:
            self.masks = [m.to(p.device) if not m.is_cuda else m for group, m in zip(self.param_groups, mask) for p in group['params']]
        else:
            self.masks = None

    def step(self, masks=None, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()
        if masks is not None:
            for group, mask in zip(self.param_groups, masks):
                for p, m in zip(group['params'], mask):
                    if p.grad is None:
                        continue

                    grad = p.grad.data
                    if grad.is_sparse:
                        raise RuntimeError('MaskedAdamW does not support sparse gradients')

                    # Apply mask
                    grad.mul_(m)

        # Call the step function of parent AdamW class to perform the actual update
        super().step()

        return loss


class MaskedAdam(optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, amsgrad=False, mask=None):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)

        # Initialize mask to ones if None is provided
        if mask is not None:
            self.masks = [m.to(p.device) if not m.is_cuda else m for group, m in zip(self.param_groups, mask) for p in group['params']]
        else:
            self.masks = None

    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()
        if self.masks is not None:
            for group, mask in zip(self.param_groups, self.masks):
                for p, m in zip(group['params'], mask):
                    if p.grad is None:
                        continue

                    grad = p.grad.data
                    if grad.is_sparse:
                        raise RuntimeError('MaskedAdamW does not support sparse gradients')

                    # Apply mask
                    grad.mul_(m)

        # Call the step function of parent AdamW class to perform the actual update
        super().step()

        return loss


class MaskedSGD(optim.SGD):
    def __init__(self, params, lr=0.01, momentum=0, dampening=0, weight_decay=0, nesterov=False, mask=None):
        super().__init__(params, lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)

        # Initialize mask to ones if None is provided
        if mask is not None:
            self.masks = [m.to(p.device) if not m.is_cuda else m for group, m in zip(self.param_groups, mask) for p in group['params']]
        else:
            self.masks = None

    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()
        if self.masks is not None:
            for group, mask in zip(self.param_groups, self.masks):
                for p, m in zip(group['params'], mask):
                    if p.grad is None:
                        continue

                    grad = p.grad.data
                    if grad.is_sparse:
                        raise RuntimeError('MaskedAdamW does not support sparse gradients')

                    # Apply mask
                    grad.mul_(m)

        # Call the step function of parent AdamW class to perform the actual update
        super().step()

        return loss
