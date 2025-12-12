# Author: Haohe Liu
# Email: haoheliu@gmail.com
# Date: 11 Feb 2023

import sys

sys.path.append("src")
import shutil
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"

import argparse
import yaml
import torch

from tqdm import tqdm
from pytorch_lightning.strategies.ddp import DDPStrategy
from audioldm_train.utilities.data.dataset import AudioDataset

from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback
from audioldm_train.utilities.tools import (
    get_restore_step,
    copy_test_subset_data,
)
from audioldm_train.utilities.model_util import instantiate_from_config
import logging

logging.basicConfig(level=logging.WARNING)


def print_on_rank0(msg):
    if torch.distributed.get_rank() == 0:
        print(msg)


def main(configs, config_yaml_path, exp_group_name, exp_name, perform_validation):
    if "seed" in configs.keys():
        seed_everything(configs["seed"])
    else:
        print("SEED EVERYTHING TO 0")
        seed_everything(0)

    if "precision" in configs.keys():
        torch.set_float32_matmul_precision(
            configs["precision"]
        )  # highest, high, medium

    log_path = configs["log_directory"]
    batch_size = configs["model"]["params"]["batchsize"]

    if "dataloader_add_ons" in configs["data"].keys():
        dataloader_add_ons = configs["data"]["dataloader_add_ons"]
    else:
        dataloader_add_ons = []

    dataset = AudioDataset(configs, split="train", add_ons=dataloader_add_ons)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=16,
        pin_memory=True,
        shuffle=True,
    )

    print(
        "The length of the dataset is %s, the length of the dataloader is %s, the batchsize is %s"
        % (len(dataset), len(loader), batch_size)
    )

    val_dataset = AudioDataset(configs, split="test", add_ons=dataloader_add_ons)

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
    )

    # Copy test data
    test_data_subset_folder = os.path.join(
        os.path.dirname(configs["log_directory"]),
        "testset_data",
        val_dataset.dataset_name,
    )
    os.makedirs(test_data_subset_folder, exist_ok=True)
    copy_test_subset_data(val_dataset.data, test_data_subset_folder)

    try:
        config_reload_from_ckpt = configs["reload_from_ckpt"]
    except:
        config_reload_from_ckpt = None

    try:
        limit_val_batches = configs["step"]["limit_val_batches"]
    except:
        limit_val_batches = None

    validation_every_n_epochs = configs["step"]["validation_every_n_epochs"]
    save_checkpoint_every_n_steps = configs["step"]["save_checkpoint_every_n_steps"]
    max_steps = configs["step"]["max_steps"]
    save_top_k = configs["step"]["save_top_k"]

    checkpoint_path = os.path.join(log_path, exp_group_name, exp_name, "checkpoints")

    wandb_path = os.path.join(log_path, exp_group_name, exp_name)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        monitor="global_step",
        mode="max",
        filename="checkpoint-fad-{val/frechet_inception_distance:.2f}-global_step={global_step:.0f}",
        every_n_train_steps=save_checkpoint_every_n_steps,
        save_top_k=save_top_k,
        auto_insert_metric_name=False,
        save_last=False,
    )

    os.makedirs(checkpoint_path, exist_ok=True)
    shutil.copy(config_yaml_path, wandb_path)

    is_external_checkpoints = False
    if len(os.listdir(checkpoint_path)) > 0:
        print("Load checkpoint from path: %s" % checkpoint_path)
        restore_step, n_step = get_restore_step(checkpoint_path)
        resume_from_checkpoint = os.path.join(checkpoint_path, restore_step)
        print("Resume from checkpoint", resume_from_checkpoint)
    elif config_reload_from_ckpt is not None:
        resume_from_checkpoint = config_reload_from_ckpt
        is_external_checkpoints = True
        print("Reload ckpt specified in the config file %s" % resume_from_checkpoint)
    else:
        print("Train from scratch")
        resume_from_checkpoint = None

    devices = torch.cuda.device_count()

    latent_diffusion = instantiate_from_config(configs["model"])
    latent_diffusion.set_log_dir(log_path, exp_group_name, exp_name)

        # ---------------------------------------------------------
    # DEBUG: Check that LoRALinear layers are actually in model
    # ---------------------------------------------------------
    try:
        print("\n=== Checking Attention Q/K/V Types ===")
        block = latent_diffusion.model.diffusion_model.middle_block[1].transformer_blocks[0].attn1
        print("Q:", type(block.to_q))
        print("K:", type(block.to_k))
        print("V:", type(block.to_v))
        print("======================================\n")
    except Exception as e:
        print("Could not inspect attention block:", e)

    # freeze all the non lora parameters
    for name, param in latent_diffusion.named_parameters():
        # train only LoRA parameters
        if "lora" not in name.lower():
            param.requires_grad = False
        else:
            print("[LoRA Trainable]", name)

    # print("\n=== Checking LoRA parameters ===")
    # lora_count = 0
    # for n, p in latent_diffusion.named_parameters():
    #     if "lora" in n.lower():
    #         print(n, "requires_grad =", p.requires_grad)
    #         lora_count += 1
    #     else:
    #         p.requires_grad = False
    # print("Total LoRA params found:", lora_count)
    total_trainable = sum(p.numel() for p in latent_diffusion.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in latent_diffusion.parameters())
    print(f"Trainable params: {total_trainable}/{total_params}")
    print("================================\n")

    # print("=== Model State Dict Keys (partial) ===")
    # sd = latent_diffusion.state_dict()
    # for k in sd.keys():
    #     if "to_q" in k or "to_k" in k or "to_v" in k:
    #         print(k)
    # print("=== END ===")

    # disable EMA
    latent_diffusion.use_ema = False

    # # freeze all the non lora parameters
    # for name, param in latent_diffusion.named_parameters():
    #     # train only LoRA parameters
    #     if "lora" not in name.lower():
    #         param.requires_grad = False
    #     else:
    #         print("[LoRA Trainable]", name)

    # print how many params are trainable
    trainable = sum(p.numel() for p in latent_diffusion.parameters() if p.requires_grad)
    total = sum(p.numel() for p in latent_diffusion.parameters())
    print(f"Trainable parameters (LoRA only): {trainable}/{total} ({100*trainable/total:.4f}%)")

    wandb_logger = WandbLogger(
        save_dir=wandb_path,
        project=configs["project"],
        config=configs,
        name="%s/%s" % (exp_group_name, exp_name),
    )

    latent_diffusion.test_data_subset_path = test_data_subset_folder

    print("==> Save checkpoint every %s steps" % save_checkpoint_every_n_steps)
    print("==> Perform validation every %s epochs" % validation_every_n_epochs)

    trainer = Trainer(
        accelerator="gpu",
        devices=devices,
        logger=wandb_logger,
        max_steps=max_steps,
        num_sanity_val_steps=1,
        limit_val_batches=limit_val_batches,
        check_val_every_n_epoch=validation_every_n_epochs,
        strategy=DDPStrategy(find_unused_parameters=False),
        #callbacks=[checkpoint_callback, LoRA_GradMonitor(), LoRA_WeightDiffMonitor()],
        callbacks=[checkpoint_callback, LoRA_GradMonitor()],
    )

    if is_external_checkpoints:
        if resume_from_checkpoint is not None:
            ckpt = torch.load(resume_from_checkpoint)["state_dict"]

            # key_not_in_model_state_dict = []
            # size_mismatch_keys = []
            state_dict = latent_diffusion.state_dict()

            # ------------------------------------------------------------------
            # 1) Remap QKV weights to LoRA base weights
            #    from:  ...attn?.to_q.weight
            #    to:    ...attn?.to_q.base.weight
            # ------------------------------------------------------------------
            remapped = {}

            def maybe_remap_qkv(orig_suffix, new_suffix):
                """
                Map keys like:
                  model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn1.to_q.weight
                to:
                  model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn1.to_q.base.weight
                """
                for key, value in list(ckpt.items()):
                    if key.endswith(orig_suffix):
                        new_key = key.replace(orig_suffix, new_suffix)
                        if new_key in state_dict and state_dict[new_key].shape == value.shape:
                            remapped[new_key] = value
                            del ckpt[key]

            # Map all Q/K/V
            maybe_remap_qkv(".to_q.weight", ".to_q.base.weight")
            maybe_remap_qkv(".to_k.weight", ".to_k.base.weight")
            maybe_remap_qkv(".to_v.weight", ".to_v.base.weight")

            # Add remapped entries back into checkpoint dict
            ckpt.update(remapped)

            # ------------------------------------------------------------------
            # 2) Drop EMA weights (we disabled EMA for training anyway)
            # ------------------------------------------------------------------
            for key in list(ckpt.keys()):
                if key.startswith("model_ema."):
                    del ckpt[key]

             # ------------------------------------------------------------------
            # 3) Filter keys that don't exist or have mismatched size
            # ------------------------------------------------------------------
            from tqdm import tqdm

            key_not_in_model_state_dict = []
            size_mismatch_keys = []

            print("Filtering key for reloading:", resume_from_checkpoint)
            print(
                "State dict key size:",
                len(list(state_dict.keys())),
                len(list(ckpt.keys())),
            )
            for key in tqdm(list(ckpt.keys())):
                if key not in state_dict.keys():
                    key_not_in_model_state_dict.append(key)
                    del ckpt[key]
                    continue
                if state_dict[key].size() != ckpt[key].size():
                    size_mismatch_keys.append(key)
                    del ckpt[key]

            latent_diffusion.load_state_dict(ckpt, strict=False)

            # --- quick sanity check: is base attention non-random now? ---
            any_q_key = [
                k for k in latent_diffusion.state_dict().keys()
                if "attn1.to_q.base.weight" in k
            ][0]
            print("Sample attn1.to_q.base.weight norm:",
                  latent_diffusion.state_dict()[any_q_key].norm().item())
            # print("Filtering key for reloading:", resume_from_checkpoint)
            # print(
            #     "State dict key size:",
            #     len(list(state_dict.keys())),
            #     len(list(ckpt.keys())),
            # )


            # for key in tqdm(list(ckpt.keys())):
            #     if key not in state_dict.keys():
            #         key_not_in_model_state_dict.append(key)
            #         del ckpt[key]
            #         continue
            #     if state_dict[key].size() != ckpt[key].size():
            #         del ckpt[key]
            #         size_mismatch_keys.append(key)

            # if(len(key_not_in_model_state_dict) != 0 or len(size_mismatch_keys) != 0):
            # print("⛳", end=" ")

            # print("==> Warning: The following key in the checkpoint is not presented in the model:", key_not_in_model_state_dict)
            # print("==> Warning: These keys have different size between checkpoint and current model: ", size_mismatch_keys)

            latent_diffusion.load_state_dict(ckpt, strict=False)

        # if(perform_validation):
        #     trainer.validate(latent_diffusion, val_loader)

        trainer.fit(latent_diffusion, loader, val_loader)
    # else:
    #     trainer.fit(
    #         latent_diffusion, loader, val_loader, ckpt_path=resume_from_checkpoint
    #     )

    else:
    # For LoRA finetuning, treat base checkpoint like "external" weights.
    # Do NOT resume optimizer state from old runs.
        if resume_from_checkpoint is not None:
            ckpt = torch.load(resume_from_checkpoint)["state_dict"]
            # (optionally apply same remapping logic here as in is_external_checkpoints)
            latent_diffusion.load_state_dict(ckpt, strict=False)

        trainer.fit(latent_diffusion, loader, val_loader)

    # after training finishes, save LoRA weights only
    save_path = os.path.join(checkpoint_path, "lora_only.pt")
    latent_diffusion.save_lora(save_path)
    print(f"[LoRA Saved] -> {save_path}")

def print_lora_gradients(model):
    print("\n=== LoRA Grad Check (detailed) ===")
    for name, param in model.named_parameters():
        if "lora" in name:
            if param.grad is None:
                print(f"[NO GRAD] {name}")
            else:
                g = param.grad
                print(
                    f"[GRAD OK] {name} | "
                    f"norm={g.norm().item():.2e}, "
                    f"max={g.abs().max().item():.2e}, "
                    f"min={g.min().item():.2e}"
                )
            break  # just one for sanity
    print("==================================\n")

def print_lora_weight_stats(model):
    print("\n=== LoRA Weight Stats ===")
    for name, param in model.named_parameters():
        if "lora" in name.lower():
            print(
                f"{name}: mean={param.data.mean().item():.9f}, "
                f"std={param.data.std().item():.9f}"
            )
    print("=========================\n")


class LoRA_GradMonitor(Callback):
    def __init__(self, max_batches_per_epoch_to_print: int = 5):
        super().__init__()
        self.max_batches_per_epoch_to_print = max_batches_per_epoch_to_print

    def on_train_epoch_start(self, trainer, pl_module):
        self.batches_this_epoch = 0

    def on_after_backward(self, trainer, pl_module):
        if self.batches_this_epoch < self.max_batches_per_epoch_to_print:
            print_lora_gradients(pl_module)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.batches_this_epoch < self.max_batches_per_epoch_to_print:
            print_lora_weight_stats(pl_module)
        self.batches_this_epoch += 1

# class LoRA_WeightDiffMonitor(Callback):
#     def __init__(self):
#         super().__init__()
#         self.prev = {}

#     def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
#         print("\n=== LoRA Weight Δ (per batch) ===")
#         with torch.no_grad():
#             for name, p in pl_module.named_parameters():
#                 if "lora" in name.lower():
#                     cur = p.detach().cpu()
#                     if name in self.prev:
#                         diff = (cur - self.prev[name]).abs().max().item()
#                         print(f"{name}: max |Δw| = {diff:.2e}")
#                     self.prev[name] = cur.clone()
#         print("=================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_yaml",
        type=str,
        required=False,
        help="path to config .yaml file",
    )

    parser.add_argument(
        "--reload_from_ckpt",
        type=str,
        required=False,
        default=None,
        help="path to pretrained checkpoint",
    )

    parser.add_argument("--val", action="store_true")

    args = parser.parse_args()

    perform_validation = args.val

    assert torch.cuda.is_available(), "CUDA is not available"

    config_yaml = args.config_yaml

    exp_name = os.path.basename(config_yaml.split(".")[0])
    exp_group_name = os.path.basename(os.path.dirname(config_yaml))

    config_yaml_path = os.path.join(config_yaml)
    config_yaml = yaml.load(open(config_yaml_path, "r"), Loader=yaml.FullLoader)

    if args.reload_from_ckpt is not None:
        config_yaml["reload_from_ckpt"] = args.reload_from_ckpt

    if perform_validation:
        config_yaml["model"]["params"]["cond_stage_config"][
            "crossattn_audiomae_generated"
        ]["params"]["use_gt_mae_output"] = False
        config_yaml["step"]["limit_val_batches"] = None

    main(config_yaml, config_yaml_path, exp_group_name, exp_name, perform_validation)
