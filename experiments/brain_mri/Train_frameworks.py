import argparse
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger, WandbLogger
import torch

from dataset import SliceDataModule
from FastMRI_model_stage1 import Stage1RVQVAE
from FastMRI_model_stage2 import FactorizedMaskGIT


def parse_args():
    parser = argparse.ArgumentParser(description="Train RVQ-VAE (Stage1) or Factorized MaskGIT (Stage2)")
    parser.add_argument("--stage", choices=["stage1", "stage2"], required=False, help="Explicit stage selector")
    parser.add_argument("--stage1", dest="stage1_flag", action="store_true", help="Shorthand for --stage stage1")
    parser.add_argument("--stage2", dest="stage2_flag", action="store_true", help="Shorthand for --stage stage2")
    parser.add_argument("--train-dir", default="/home/mluser1/Musti_Anomaly_Detection/FastMRI_Sample_Work/Training_samples_FastMRI_IXI")
    parser.add_argument("--val-dir", default="/home/mluser1/Musti_Anomaly_Detection/FastMRI_Sample_Work/Validation_samples_FastMRI")
    parser.add_argument("--file-ext", type=str, default=".npz", help="Slice file extension (e.g., .npz)")
    parser.add_argument("--augment", dest="augment", action="store_true", help="Enable data augmentation for both Stage1 and Stage2")
    parser.add_argument("--no-augment", dest="augment", action="store_false", help="Disable data augmentation for both Stage1 and Stage2")
    parser.set_defaults(augment=True)
    parser.add_argument("--batch-size", type=int, default=158) #stage 2 = 158, stage 1 = 192
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--precision", default=32, type=int)
    parser.add_argument("--biomedclip-model-name", type=str, default="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    parser.add_argument("--biomedclip-open-clip-model", type=str, default=None)
    parser.add_argument("--biomedclip-open-clip-pretrained", type=str, default=None)
    
    parser.add_argument("--stage1-ckpt", type=str, default="/home/mluser1/Musti_Anomaly_Detection/FastMRI_Sample_Work/FastMRI_IXI_Augmented_lightningCheckpoints/FastMRI_stage1-epoch=099-val/loss=0.0891.ckpt", help="Checkpoint path for Stage1 (needed for Stage2 training)")
    parser.add_argument("--pretrained-stage1-ckpt", type=str, default=None, help="Optional Stage1 checkpoint to initialize weights for fine-tuning")
    parser.add_argument("--pretrained-stage2-ckpt", type=str, default=None, help="Optional Stage2 checkpoint to initialize MaskGIT weights for fine-tuning")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--wandb-project", type=str, default="RVQ-MaskGIT-FastMRI-IXI", help="Weights & Biases project name; set empty to disable")
    parser.add_argument("--wandb-entity", type=str, default=None, help="Weights & Biases entity (team)")
    parser.add_argument("--wandb-run-name", type=str, default="Stage2-Augmented-FastMRI-IXI-RQV-BioMed-MaskGIT-256-patchsize-8", help="Run display name for Wandb")
    parser.add_argument("--wandb-off", action="store_true", help="Disable Wandb logging even if project is set")
    args = parser.parse_args()

    # Resolve stage from shorthand flags if not explicitly provided
    if args.stage is None:
        if args.stage1_flag and args.stage2_flag:
            parser.error("Specify only one of --stage1 or --stage2")
        if args.stage1_flag:
            args.stage = "stage1"
        elif args.stage2_flag:
            args.stage = "stage2"
        else:
            parser.error("--stage is required (or use --stage1 / --stage2 shorthand)")

    return args


def make_trainer(args):
    callbacks = [
        ModelCheckpoint(save_top_k=3, monitor="val/loss", mode="min", verbose=True,
                        dirpath=os.path.join("/home/mluser1/Musti_Anomaly_Detection/FastMRI_Sample_Work", "FastMRI_IXI_Augmented_lightningCheckpoints"),
                        filename=f"FastMRI_{args.stage}-{{epoch:03d}}-{{val/loss:.4f}}"),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    csv_logger = CSVLogger(args.log_dir, name=args.stage)
    loggers = [csv_logger]

    if args.wandb_project and not args.wandb_off:
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            save_dir=args.log_dir,
            log_model=True,
            config=vars(args),
        )
        loggers.append(wandb_logger)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        devices=[1],
        accelerator="auto",
        gradient_clip_val=1.0,
        callbacks=callbacks,
        logger=loggers,
        precision=args.precision,
    )
    return trainer


def main():
    args = parse_args()
    torch.set_float32_matmul_precision("medium")

    datamodule = SliceDataModule(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=args.augment,
        file_ext=args.file_ext,
    )

    trainer = make_trainer(args)
    codebook_size = 256
    embed_dim = 256
    if args.stage == "stage1":
        if args.pretrained_stage1_ckpt is not None:
            if not os.path.exists(args.pretrained_stage1_ckpt):
                raise FileNotFoundError(f"Pretrained Stage1 checkpoint not found: {args.pretrained_stage1_ckpt}")
            model = Stage1RVQVAE.load_from_checkpoint(
                args.pretrained_stage1_ckpt,
                lr=args.lr,
                embed_dim=embed_dim,
                codebook_size=codebook_size,
                commitment_cost=0.25,
                biomedclip_model_name=args.biomedclip_model_name,
                biomedclip_open_clip_model=args.biomedclip_open_clip_model,
                biomedclip_open_clip_pretrained=args.biomedclip_open_clip_pretrained,
                use_augmentations=False,
                strict=False,
            )
        else:
            model = Stage1RVQVAE(
                lr=args.lr,
                embed_dim=embed_dim,
                codebook_size=codebook_size,
                commitment_cost=0.25,
                biomedclip_model_name=args.biomedclip_model_name,
                biomedclip_open_clip_model=args.biomedclip_open_clip_model,
                biomedclip_open_clip_pretrained=args.biomedclip_open_clip_pretrained,
                use_augmentations=False,
            )
        trainer.fit(model, datamodule=datamodule)
    else:
        if args.stage1_ckpt is None or not os.path.exists(args.stage1_ckpt):
            raise FileNotFoundError("Stage2 training requires a valid --stage1-ckpt path")
        stage1 = Stage1RVQVAE.load_from_checkpoint(args.stage1_ckpt)
        if args.pretrained_stage2_ckpt is not None:
            if not os.path.exists(args.pretrained_stage2_ckpt):
                raise FileNotFoundError(f"Pretrained Stage2 checkpoint not found: {args.pretrained_stage2_ckpt}")
            model = FactorizedMaskGIT.load_from_checkpoint(
                args.pretrained_stage2_ckpt,
                stage1=stage1,
                lr=args.lr,
                strict=False,
            )
        else:
            model = FactorizedMaskGIT(stage1=stage1, lr=args.lr, codebook_size_level1=codebook_size, codebook_size_level2=codebook_size, embed_dim=embed_dim)
        trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
