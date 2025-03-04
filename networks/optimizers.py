import torch
from torch.optim import Optimizer

class PIOptimizer(Optimizer):
    def __init__(self, params, kp=0.1, ki=0.01):
        """PI Optimizer.
        Args:
            params: iterable of parameters to optimize
            kp (float): Proportional gain (like learning rate)
            ki (float): Integral gain
        """
        defaults = dict(kp=kp, ki=ki)
        super(PIOptimizer, self).__init__(params, defaults)
        # Initialize integral term for each parameter
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['integral'] = torch.zeros_like(p.data)

    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            kp = group['kp']
            ki = group['ki']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                
                # Update integral term
                state['integral'].add_(grad)
                
                # PI update: - (Kp * grad + Ki * integral)
                update = kp * grad + ki * state['integral']
                p.data.add_(-update)

        return loss