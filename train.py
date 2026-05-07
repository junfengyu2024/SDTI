#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 Yixiao Chen, Shikun Li, Ruoyu Li, Zhe Li, Junliang Xing
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0
"""
Training script: multi‑targeted adversarial attack via DDIM inversion + diffusion with LoRA.
First half of generation steps are frozen (unconditional), second half are fine‑tuned
with adversarial loss on a classifier.
"""

import os
import json
import math
import random
import argparse

import safetensors
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
from accelerate.utils import set_seed
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from PIL import Image, ImageFile
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    StableDiffusionPipeline,
    UNet2DConditionModel,
    DDIMScheduler,
    DDIMInverseScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import convert_state_dict_to_diffusers, convert_all_state_dict_to_peft

from utils import get_classes, normalize, unnormalize_ddpm, budget, load_single_classifier


class ImageFolderWithPath(torchvision.datasets.ImageFolder):
    """Dataset returning (image, path) instead of (image, label)."""
    def __getitem__(self, index):
        path, _ = self.samples[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi‑targeted adversarial attack via DDIM + LoRA"
    )
    # Paths
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained diffusion model (tokenizer, text_encoder, vae, unet, scheduler).",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="output",
        help="Root output directory.",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Directory containing the training images.",
    )
    parser.add_argument(
        "--label_to_caption_path",
        type=str,
        default="imagenet_class_index.json",
        help="JSON mapping from ImageNet class id to [wnid, caption].",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Path to pre‑trained LoRA weights (safetensors).",
    )

    # Training
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=2e-05)
    parser.add_argument("--dataloader_num_workers", type=int, default=8)

    # Optimizer
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Attack
    parser.add_argument("--eps", type=int, default=16, help="Perturbation budget in 0‑255 scale.")
    parser.add_argument("--label_flag", type=str, default="N8", help="Target label set identifier.")
    parser.add_argument("--model_type", type=str, default="incv3", help="Classifier architecture.")
    parser.add_argument("--attack_mode", type=str, default="multi_targeted", help="Attack mode.")
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank.")
    parser.add_argument("--s_for", type=int, default=6, help="Number of inverse DDIM steps.")
    parser.add_argument("--s_gen", type=int, default=6, help="Number of generative DDIM steps.")
    parser.add_argument("--num_timesteps", type=int, default=20, help="Total DDIM timesteps.")

    # Loss
    parser.add_argument("--lambda_adv", type=float, default=1.0, help="Weight of adversarial loss.")
    parser.add_argument("--freeze_ratio", type=float, default=0.5, help="Ratio of frozen generation steps.")

    return parser.parse_args()


def main():
    args = parse_args()

    torch.set_float32_matmul_precision("high")

    # ======================== Device ========================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.float32

    # ======================== Output directory ========================
    output_dir = os.path.join(args.output_root, args.attack_mode, args.model_type)
    eps = args.eps / 255.0
    label_set = get_classes(args.label_flag)

    if args.seed is not None:
        set_seed(args.seed)
    os.makedirs(output_dir, exist_ok=True)

    # ======================== Load diffusion models ========================
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder"
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae"
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet"
    )
    unet_origin = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet"
    )

    ddim_scheduler = DDIMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    ddim_inv_scheduler = DDIMInverseScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    ddim_inv_scheduler.set_timesteps(args.num_timesteps, device=device)
    timesteps_inverse = ddim_inv_scheduler.timesteps[: args.s_for]

    ddim_scheduler.set_timesteps(args.num_timesteps, device=device)
    timesteps = ddim_scheduler.timesteps[-args.s_gen :]

    assert len(timesteps) >= 2, "s_gen must be at least 2 to split frozen/trainable steps"

    freeze_steps = max(1, int(len(timesteps) * args.freeze_ratio))
    freeze_steps = min(freeze_steps, len(timesteps) - 1)

    timesteps_frozen = timesteps[:freeze_steps]   # unconditional, frozen
    timesteps_train = timesteps[freeze_steps:]    # fine‑tuned with target caption

    print(f"Total generation steps: {len(timesteps)}")
    print(f"Frozen steps: {len(timesteps_frozen)} | Trainable steps: {len(timesteps_train)}")

    # ======================== Load classifier (single, via utils) ========================
    classifier, input_size = load_single_classifier(args.model_type, device, weight_dtype)
    criterion = nn.CrossEntropyLoss()

    # ======================== Freeze components, move to device ========================
    def freeze_and_move(model):
        model.requires_grad_(False)
        return model.to(device, dtype=weight_dtype)

    vae = freeze_and_move(vae)
    text_encoder = freeze_and_move(text_encoder)
    unet_origin = freeze_and_move(unet_origin)
    # classifier is already frozen by load_single_classifier
    unet = unet.to(device, dtype=weight_dtype)

    # ======================== LoRA configuration ========================
    unet_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0", "add_k_proj", "add_v_proj"],
    )
    unet.add_adapter(unet_lora_config)

    # ======================== Load pre‑trained LoRA weights ========================
    if args.lora_path is not None and os.path.exists(args.lora_path):
        print(f"Loading LoRA weights from: {args.lora_path}")
        lora_state_dict = safetensors.torch.load_file(args.lora_path)
        unet_lora_state_dict = {
            k[len("unet."):]: v for k, v in lora_state_dict.items()
            if k.startswith("unet.")
        }
        unet_lora_state_dict = convert_all_state_dict_to_peft(unet_lora_state_dict)
        set_peft_model_state_dict(unet, unet_lora_state_dict)
        print("LoRA weights loaded successfully.")

    lora_layers = [p for p in unet.parameters() if p.requires_grad]

    # ======================== Optimizer ========================
    optimizer = torch.optim.AdamW(
        lora_layers,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    def tokenize_captions(captions):
        return tokenizer(
            captions,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids

    # ======================== Data preprocessing ========================
    scale_size = input_size
    unet_in_transform = transforms.Resize(352)   # fixed for latent space
    unet_out_transform = transforms.Resize(scale_size)

    train_transforms = transforms.Compose([
        transforms.Resize(scale_size),
        transforms.CenterCrop(scale_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    # ======================== Dataset ========================
    full_dataset = ImageFolderWithPath(args.dataset_dir, train_transforms)
    print(f"Dataset size: {len(full_dataset)} images")

    with open(args.label_to_caption_path, "r") as f:
        label_to_caption = json.load(f)

    wnid_to_label = {v[0]: int(k) for k, v in label_to_caption.items()}
    label_candidates = [int(x.item()) if torch.is_tensor(x) else int(x) for x in label_set]

    def get_true_label_from_path(img_path):
        file_name = os.path.basename(img_path)
        wnid_prefix = file_name.split("_")[0]
        if wnid_prefix not in wnid_to_label:
            raise ValueError(f"Cannot infer true label from: {img_path}")
        return wnid_to_label[wnid_prefix]

    def get_target_label_and_caption(img_path):
        true_label = get_true_label_from_path(img_path)
        # Multi‑targeted: any label except the true one
        valid_candidates = [x for x in label_candidates if x != true_label]
        if not valid_candidates:
            raise ValueError(f"No valid target label for true label {true_label}")
        label = random.choice(valid_candidates)
        caption = f"a photo of {label_to_caption[str(label)][1]}"
        return label, caption

    def collate_fn(examples):
        pixel_values = torch.stack([x[0] for x in examples]).float()
        img_paths = [x[1] for x in examples]
        labels, captions = zip(
            *[get_target_label_and_caption(img_path) for img_path in img_paths]
        )
        return {
            "imgs": pixel_values,
            "captions": tokenize_captions(captions),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    train_dataloader = torch.utils.data.DataLoader(
        full_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # ======================== Learning rate scheduler ========================
    num_update_steps_per_epoch = len(train_dataloader) * len(timesteps_train)
    total_training_steps = num_update_steps_per_epoch * args.num_train_epochs
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=total_training_steps,
    )

    # ======================== Training loop ========================
    for epoch in range(args.num_train_epochs):
        unet.train()
        epoch_progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch}",
            unit="it",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )

        for step, batch in enumerate(epoch_progress_bar):
            imgs = batch["imgs"].to(dtype=weight_dtype, device=device)
            cap = batch["captions"].to(device)
            label = batch["labels"].long().to(device)

            with torch.no_grad():
                latents = (
                    vae.encode(unet_in_transform(imgs)).latent_dist.sample()
                    * vae.config.scaling_factor
                )
                enc = text_encoder(cap, return_dict=False)[0]
                enc_empty = text_encoder(
                    tokenize_captions([""] * imgs.size(0)).to(device),
                    return_dict=False,
                )[0]

                # Inverse DDIM to the attack starting point
                for t in timesteps_inverse:
                    pred = unet_origin(
                        latents, t, encoder_hidden_states=enc_empty, return_dict=False
                    )[0]
                    latents = ddim_inv_scheduler.step(pred, t, latents, return_dict=False)[0]

                # Frozen generation steps (unconditional)
                for t in timesteps_frozen:
                    pred = unet_origin(
                        latents, t, encoder_hidden_states=enc_empty, return_dict=False
                    )[0]
                    latents, _ = ddim_scheduler.step(pred, t, latents, return_dict=False)
                    latents = latents.detach()

            # Trainable generation steps: one optimizer step per timestep
            for t in timesteps_train:
                optimizer.zero_grad()

                pred = unet(latents, t, encoder_hidden_states=enc.detach(), return_dict=False)[0]
                latents, pred_original_sample = ddim_scheduler.step(
                    pred, t, latents, return_dict=False
                )

                adv_img = vae.decode(
                    pred_original_sample / vae.config.scaling_factor, return_dict=False
                )[0]
                adv_img = budget(
                    unet_out_transform(unnormalize_ddpm(adv_img)),
                    unnormalize_ddpm(imgs),
                    eps,
                    device,
                    attack_mode=args.attack_mode,
                )

                # Forward through classifier
                out = classifier(normalize(adv_img))
                # For HuggingFace models, logits are stored in .logits
                if hasattr(out, "logits"):
                    out = out.logits

                adv_loss = criterion(out, label)
                step_loss = args.lambda_adv * adv_loss

                step_loss.backward()
                torch.nn.utils.clip_grad_norm_(lora_layers, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()

                current_lr = lr_scheduler.get_last_lr()[0]
                epoch_progress_bar.set_postfix(
                    adv_loss=f"{adv_loss.item():.3f}",
                    lr=f"{current_lr:.1e}",
                )

                # Detach latents to break gradient flow between steps
                latents = latents.detach()

        epoch_progress_bar.close()

    # ======================== Save final LoRA weights ========================
    print("Training finished. Saving final model.")
    final_path = os.path.join(output_dir, "final_lora")
    lora_state = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    StableDiffusionPipeline.save_lora_weights(
        final_path,
        unet_lora_layers=lora_state,
        safe_serialization=True,
    )
    print(f"Final LoRA weights saved to {final_path}")


if __name__ == "__main__":
    main()