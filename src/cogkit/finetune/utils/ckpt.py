from pathlib import Path

import torch.distributed as dist
from safetensors.torch import save_file
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict, StateDictOptions
from torch.distributed.checkpoint.stateful import Stateful

from cogkit.utils.lora import save_lora

from .dist import is_main_process
from .io import check_path


def save_state_dict(
    state_dict: dict, save_dir: str, fname: str, metadata: dict = None, lora: bool = False
) -> None:
    if is_main_process():
        if lora:
            save_lora(state_dict, save_dir)
        else:
            save_file(state_dict, save_dir / fname, metadata)

    dist.barrier()


def get_global_step(ckpt_path: str | Path) -> int:
    ckpt_path = Path(ckpt_path)
    check_path(ckpt_path, must_exists=True, must_dir=True)

    try:
        global_step = int(ckpt_path.name.split("-")[1])
    except IndexError:
        raise ValueError(f"Checkpoint path '{ckpt_path}' is not in the correct format.")

    return global_step


class AppState(Stateful):
    """This is a useful wrapper for checkpointing the Application State. Since this object is compliant
    with the Stateful protocol, DCP will automatically call state_dict/load_stat_dict as needed in the
    dcp.save/load APIs.

    Note: We take advantage of this wrapper to hande calling distributed state dict methods on the model
    and optimizer.

    For more details, please refer to: https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html
    """

    def __init__(self, model, optimizer=None, lora: bool = False):
        self.model = model
        self.optimizer = optimizer
        self.lora = lora

    def state_dict(self):
        # this line automatically manages FSDP FQN's, as well as sets the default state dict type to FSDP.SHARDED_STATE_DICT
        model_state_dict, optimizer_state_dict = get_state_dict(self.model, self.optimizer)
        if self.lora:
            from peft import get_peft_model_state_dict

            model_state_dict = get_peft_model_state_dict(self.model)

        return {"model": model_state_dict, "optim": optimizer_state_dict}

    def load_state_dict(self, state_dict):
        # sets our state dicts on the model and optimizer, now that we've loaded
        if self.lora:
            from peft.utils.save_and_load import _insert_adapter_name_into_state_dict
            from cogkit.utils.lora import _ADAPTER_NAME
            from peft.utils.constants import PEFT_TYPE_TO_PREFIX_MAPPING

            state_dict["model"] = _insert_adapter_name_into_state_dict(
                state_dict["model"],
                adapter_name=_ADAPTER_NAME,
                parameter_prefix=PEFT_TYPE_TO_PREFIX_MAPPING[
                    self.model.peft_config[_ADAPTER_NAME].peft_type
                ],
            )

        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optim"],
            options=StateDictOptions(strict=False),
        )
