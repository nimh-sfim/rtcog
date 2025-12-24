"""
Register experiment plugins by defining `Processor` and `ActionSeries`
"""
from rtcog.processor.basic_processor import BasicProcessor
from rtcog.processor.esam_processor import ESAMProcessor
from rtcog.controller.action_series import BasicActionSeries, ESAMActionSeries

EXPERIMENT_REGISTRY = {
    "basic": {
        "processor": BasicProcessor,
        "action": BasicActionSeries,
    },
    "esam": {
        "processor": ESAMProcessor,
        "action": ESAMActionSeries,
    },
}


    