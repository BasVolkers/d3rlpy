from typing import Any, Dict, Optional
import wandb
from d3rlpy.logging.logger import LoggerAdapter, LoggerAdapterFactory

__all__ = ["WandBLoggerAdapter", "WandBLoggerAdapterFactory"]

class WandBLoggerAdapter(LoggerAdapter):
    def __init__(self, project_name: str, experiment_name: str, config: dict, **kwargs) -> None:
        wandb.init(
            project=project_name,
            config=config,
            name=experiment_name,
            **kwargs
        )

    def write_params(self, params: Dict[str, Any]) -> None:
        wandb.config.update(params)

    def write_metric(self, epoch: int, step: int, name: str, value: float) -> None:
        wandb.log({name: value}, step=step)

class WandBLoggerAdapterFactory(LoggerAdapterFactory):
    r"""WandBLoggerAdapter class.

    This class instantiates ``WandBLoggerAdapter`` object.

    Args:
        project_name (str): Name of the WandB project.
        config (dict): Dictionary of hyperparameters and run metadata.
    """
    def __init__(self, project_name, config, run_name, **wandb_kwargs) -> None:
        super().__init__()
        self._project_name = project_name
        self._config = config
        self._run_name = run_name
        self._wandb_kwargs = wandb_kwargs

    def create(self, experiment_name: str) -> WandBLoggerAdapter:
        return WandBLoggerAdapter(self._project_name, self._run_name or experiment_name, self._config, **self._wandb_kwargs)