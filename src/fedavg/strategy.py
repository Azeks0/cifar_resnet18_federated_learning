from __future__ import annotations

import flwr as fl
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
from typing import Callable, Dict, List, Optional, Tuple, Union

class CustomFedAvg(fl.server.strategy.Strategy):
    """A custom FedAvg implementation from scratch."""
    
    def __init__(self, initial_parameters: Parameters, evaluate_fn: Optional[Callable] = None):
        super().__init__()
        self.initial_parameters = initial_parameters
        self.evaluate_fn = evaluate_fn

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        return self.initial_parameters

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        # Sample all available clients
        clients = client_manager.sample(num_clients=client_manager.num_available(), min_num_clients=1)
        config = {} # Empty config for fit
        fit_ins = FitIns(parameters, config)
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        # Implementation of FedAvg from scratch
        # 1. Calculate total number of examples across all clients
        total_examples = sum([fit_res.num_examples for _, fit_res in results])
        
        # 2. Extract weights and compute weighted average
        weighted_weights = []
        for _, fit_res in results:
            weights = parameters_to_ndarrays(fit_res.parameters)
            num_examples = fit_res.num_examples
            weight_factor = num_examples / total_examples
            weighted_weights.append([layer * weight_factor for layer in weights])
            
        # Sum layers across all clients
        aggregated_weights = list()
        for layer_updates in zip(*weighted_weights):
            aggregated_weights.append(sum(layer_updates))
            
        return ndarrays_to_parameters(aggregated_weights), {}

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        # We perform centralized evaluation via evaluate_fn, so no client evaluation is strictly necessary.
        # But we can ask all clients to evaluate just to conform.
        clients = client_manager.sample(num_clients=client_manager.num_available(), min_num_clients=1)
        eval_ins = EvaluateIns(parameters, {})
        return [(client, eval_ins) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        if not results:
            return None, {}
            
        # Weighted average of client evaluation losses and accuracies
        total_examples = sum([eval_res.num_examples for _, eval_res in results])
        weighted_loss = sum([eval_res.loss * eval_res.num_examples for _, eval_res in results]) / total_examples
        
        weighted_acc = sum([eval_res.metrics["accuracy"] * eval_res.num_examples for _, eval_res in results]) / total_examples
        
        return weighted_loss, {"accuracy": weighted_acc}

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        if self.evaluate_fn is None:
            return None
            
        weights = parameters_to_ndarrays(parameters)
        return self.evaluate_fn(server_round, weights, {})
