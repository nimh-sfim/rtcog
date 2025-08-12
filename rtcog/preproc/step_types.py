"""Enum of possible PreprocSteps."""
from enum import Enum

class StepType(Enum):
    EMA = 'ema'
    IGLM = 'iglm'
    KALMAN = 'kalman'
    SMOOTH = 'smooth'
    SNORM = 'snorm'
    WINDOWING = 'windowing'
