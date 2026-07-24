"""Built-in autonomous research subsystems."""

from .experiment import ExperimentSubsystem
from .literature import LiteratureSubsystem
from .proof import ProofSubsystem
from .theory import TheorySubsystem

__all__ = [
    "ExperimentSubsystem",
    "LiteratureSubsystem",
    "ProofSubsystem",
    "TheorySubsystem",
]
