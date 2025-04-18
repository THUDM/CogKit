# -*- coding: utf-8 -*-


from cogkit.finetune import register

from .lora_trainer_packing import Cogview4LoraPackingTrainer


class Cogview4SFTPackingTrainer(Cogview4LoraPackingTrainer):
    pass


register("cogview4-6b", "sft-packing", Cogview4SFTPackingTrainer)
