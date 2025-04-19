import torch
from typing import Callable, Tuple, List
from abc import ABC, abstractmethod

class FixpointSolver(ABC):
    """
    Base class for fixed-point solvers.
    Used to find equilibrium states in DFC and EFC.
    """
    def __init__(self, max_iter: int = 50, error_threshold: float = 1e-4, threshold_metric: str = "l2"):
        """
        Initialize the solver.
        
        Args:
            max_iter: Maximum number of iterations
            error_threshold: Convergence threshold
            threshold_metric: Metric for convergence ('l2' or 'l_inf')
        """
        self.max_iter = max_iter
        self.error_threshold = error_threshold
        self.threshold_metric = threshold_metric
        
    def get_relative_error(self, x: torch.Tensor, x_next: torch.Tensor) -> float:
        """
        Compute relative error between current and next state.
        
        Args:
            x: Current state tensor
            x_next: Next state tensor
            
        Returns:
            float: Relative error
        """
        if self.threshold_metric == "l2":
            return (x - x_next).norm(p=2, dim=tuple(range(1, len(x.shape)))).div(
                1e-5 + x.norm(p=2, dim=tuple(range(1, len(x.shape))))
            ).max().item()
        elif self.threshold_metric == "l_inf":
            return (x - x_next).abs().div(1e-5 + x.abs()).max().item()
        else:
            raise ValueError(f"Unknown threshold metric: {self.threshold_metric}")
    
    def __call__(self, dynamic_function: Callable[[torch.Tensor], torch.Tensor], 
                initial_state: torch.Tensor) -> Tuple[torch.Tensor, List[float]]:
        """
        Find fixed point of the dynamic function.
        
        Args:
            dynamic_function: Function that computes the next state given the current state
            initial_state: Initial state tensor
            
        Returns:
            Tuple[torch.Tensor, List[float]]: Fixed point and convergence history
        """
        return self._solve(dynamic_function, initial_state)
    
    @abstractmethod
    def _solve(self, dynamic_function: Callable[[torch.Tensor], torch.Tensor], 
              initial_state: torch.Tensor) -> Tuple[torch.Tensor, List[float]]:
        """
        Implementation of the solver.
        To be overridden by specific solver algorithms.
        """
        pass


class VanillaSolver(FixpointSolver):
    """Simple fixed-point iteration solver."""
    
    def _solve(self, dynamic_function: Callable[[torch.Tensor], torch.Tensor], 
              initial_state: torch.Tensor) -> Tuple[torch.Tensor, List[float]]:
        """
        Solve using simple fixed-point iteration.
        
        Args:
            dynamic_function: Function that computes the next state given the current state
            initial_state: Initial state tensor
            
        Returns:
            Tuple[torch.Tensor, List[float]]: Fixed point and convergence history
        """
        state = initial_state
        convergence_history = []
        
        for _ in range(self.max_iter):
            next_state = dynamic_function(state)
            error = self.get_relative_error(state, next_state)
            convergence_history.append(error)
            
            state = next_state
            
            if error < self.error_threshold:
                break
                
        return state, convergence_history


class AndersonSolver(FixpointSolver):
    """
    Anderson acceleration for fixed point problems.
    Faster convergence than vanilla fixed-point iteration.
    """
    def __init__(self, m: int = 5, lam: float = 1e-4, beta: float = 1.0, 
                max_iter: int = 50, error_threshold: float = 1e-4):
        """
        Initialize Anderson acceleration solver.
        
        Args:
            m: Number of past iterations to use
            lam: Regularization parameter
            beta: Mixing parameter
            max_iter: Maximum number of iterations
            error_threshold: Convergence threshold
        """
        super().__init__(max_iter, error_threshold)
        self.m = m
        self.lam = lam
        self.beta = beta
        
    def _solve(self, dynamic_function: Callable[[torch.Tensor], torch.Tensor], 
              initial_state: torch.Tensor) -> Tuple[torch.Tensor, List[float]]:
        """
        Solve using Anderson acceleration.
        
        Args:
            dynamic_function: Function that computes the next state given the current state
            initial_state: Initial state tensor
            
        Returns:
            Tuple[torch.Tensor, List[float]]: Fixed point and convergence history
        """
        bsz, d = initial_state.shape
        X = torch.zeros(bsz, self.m, d, dtype=initial_state.dtype, device=initial_state.device)
        F = torch.zeros(bsz, self.m, d, dtype=initial_state.dtype, device=initial_state.device)
        
        # Initial states
        X[:, 0], F[:, 0] = initial_state, dynamic_function(initial_state)
        X[:, 1], F[:, 1] = F[:, 0], dynamic_function(F[:, 0])
        
        # Setup for least squares solution
        H = torch.zeros(bsz, self.m + 1, self.m + 1, dtype=initial_state.dtype, device=initial_state.device)
        H[:, 0, 1:] = H[:, 1:, 0] = 1
        y = torch.zeros(bsz, self.m + 1, 1, dtype=initial_state.dtype, device=initial_state.device)
        y[:, 0] = 1
        
        convergence_history = []
        
        for k in range(2, self.max_iter):
            n = min(k, self.m)
            G = F[:, :n] - X[:, :n]
            
            # Update H matrix for least squares
            H[:, 1:n + 1, 1:n + 1] = torch.bmm(G, G.transpose(1, 2)) + self.lam * \
                                     torch.eye(n, dtype=initial_state.dtype, device=initial_state.device)[None]
            
            # Solve least squares problem
            alpha = torch.solve(y[:, :n + 1], H[:, :n + 1, :n + 1])[0][:, 1:n + 1, 0]
            
            # Update state
            X[:, k % self.m] = self.beta * (alpha[:, None] @ F[:, :n])[:, 0] + \
                              (1 - self.beta) * (alpha[:, None] @ X[:, :n])[:, 0]
            F[:, k % self.m] = dynamic_function(X[:, k % self.m])
            
            # Check convergence
            error = self.get_relative_error(F[:, k % self.m], X[:, k % self.m])
            convergence_history.append(error)
            
            if error < self.error_threshold:
                break
                
        return X[:, k % self.m], convergence_history


def get_solver(config):
    """
    Factory function to create a solver from configuration.
    
    Args:
        config: Configuration object with solver settings
        
    Returns:
        FixpointSolver: Configured solver
    """
    solver_type = getattr(config, 'solver', 'vanilla')
    
    if solver_type == 'vanilla':
        return VanillaSolver(
            max_iter=getattr(config, 'solver_max_iter', 50),
            error_threshold=getattr(config, 'solver_error_threshold', 1e-4),
            threshold_metric=getattr(config, 'solver_threshold_metric', 'l2')
        )
    elif solver_type == 'anderson':
        return AndersonSolver(
            m=getattr(config, 'solver_m', 5),
            lam=getattr(config, 'solver_lam', 1e-4),
            beta=getattr(config, 'solver_beta', 1.0),
            max_iter=getattr(config, 'solver_max_iter', 50),
            error_threshold=getattr(config, 'solver_error_threshold', 1e-4)
        )
    else:
        raise ValueError(f"Unknown solver type: {solver_type}")