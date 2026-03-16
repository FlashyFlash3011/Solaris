from solaris.models.fno import FNO
from solaris.models.afno import AFNO
from solaris.models.mlp import FullyConnected
from solaris.models.meshgraphnet import MeshGraphNet
from solaris.models.constrained_fno import ConstrainedFNO
from solaris.models.residual_corrector import NeuralResidualCorrector
from solaris.models.multiscale_fno import MultiScaleFNO
from solaris.models.conformal import ConformalNeuralOperator
from solaris.models.coupled import CoupledOperator

__all__ = [
    "FNO",
    "AFNO",
    "FullyConnected",
    "MeshGraphNet",
    "ConstrainedFNO",
    "NeuralResidualCorrector",
    "MultiScaleFNO",
    "ConformalNeuralOperator",
    "CoupledOperator",
]
