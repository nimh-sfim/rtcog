from rtcog.processor.preproc_processor import PreprocProcessor
from rtcog.processor.esam_processor import ESAMProcessor
from rtcog.controller.action_series import PreprocActionSeries, ESAMActionSeries

EXPERIMENT_REGISTRY = {
    "preproc": {
        "processor": PreprocProcessor,
        "action": PreprocActionSeries,
    },
    "esam": {
        "processor": ESAMProcessor,
        "action": ESAMActionSeries,
    },
}


    