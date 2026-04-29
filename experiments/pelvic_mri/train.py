import argparse
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger, WandbLogger
import torch

from dataset import SliceDataModule
from Model_stage_1 import Stage1RVQVAE
from Model_stage_2 import FactorizedMaskGIT


def parse_args():
    parser = argparse.ArgumentParser(description="Train RVQ-VAE (Stage1) or Factorized MaskGIT (Stage2)")
    parser.add_argument("--stage", choices=["stage1", "stage2"], required=False, help="Explicit stage selector")
    parser.add_argument("--stage1", dest="stage1_flag", action="store_true", help="Shorthand for --stage stage1")
    parser.add_argument("--stage2", dest="stage2_flag", action="store_true", help="Shorthand for --stage stage2")
    parser.add_argument("--data-dir", default="/home/mluser1/Musti_Anomaly_Detection/Data/PreSliced")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--precision", default=32, type=int)
    parser.add_argument("--stage1-ckpt", type=str, default="/home/mluser1/Musti_Anomaly_Detection/RQV-MaskGIT/lightningCheckpoints_Modified/Modified_stage1-epoch=094-val/loss=0.8587.ckpt", help="Checkpoint path for Stage1 (needed for Stage2 training)")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--wandb-project", type=str, default="RVQ-MaskGIT", help="Weights & Biases project name; set empty to disable")
    parser.add_argument("--wandb-entity", type=str, default=None, help="Weights & Biases entity (team)")
    parser.add_argument("--wandb-run-name", type=str, default="Stage2-Checkerboard-RQV-MaskGIT-256-patchsize-8-Modified", help="Run display name for Wandb")
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
                        dirpath=os.path.join("/home/mluser1/Musti_Anomaly_Detection/RQV-MaskGIT", "lightningCheckpoints_Modified"),
                        filename=f"Modified_Checkerboard_{args.stage}-{{epoch:03d}}-{{val/loss:.4f}}"),
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
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    trainer = make_trainer(args)

    if args.stage == "stage1":
        model = Stage1RVQVAE(lr=args.lr, embed_dim=256, codebook_size=192, commitment_cost=0.25)
        trainer.fit(model, datamodule=datamodule)
    else:
        if args.stage1_ckpt is None or not os.path.exists(args.stage1_ckpt):
            raise FileNotFoundError("Stage2 training requires a valid --stage1-ckpt path")
        stage1 = Stage1RVQVAE.load_from_checkpoint(args.stage1_ckpt)
        model = FactorizedMaskGIT(stage1=stage1, lr=args.lr)
        trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
