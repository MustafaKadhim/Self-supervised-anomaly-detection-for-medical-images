"""
Self-Supervised Anomaly Detection Framework for Medical Images
"""

from framework.models import Encoder, Decoder, AnomalyAutoencoder
from framework.trainers import AnomalyTrainer
from framework.datasets import MedicalImageDataset
from framework.utils import compute_anomaly_score, plot_anomaly_map

__version__ = "1.0.0"
__all__ = [
    "Encoder",
    "Decoder",
    "AnomalyAutoencoder",
    "AnomalyTrainer",
    "MedicalImageDataset",
    "compute_anomaly_score",
    "plot_anomaly_map",
]
