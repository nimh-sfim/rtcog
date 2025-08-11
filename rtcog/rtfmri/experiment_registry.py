from rtfmri.processor.preproc_processor import PreprocProcessor
from rtfmri.processor.esam_processor import ESAMProcessor
from rtfmri.controller.action_series import PreprocActionSeries, ESAMActionSeries

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


    