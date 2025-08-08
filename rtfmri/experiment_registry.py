from rtfmri.experiment.preproc_experiment import PreprocExperiment
from rtfmri.experiment.esam_experiment import ESAMExperiment
from rtfmri.controller.action_series import PreprocActionSeries, ESAMActionSeries

EXPERIMENT_REGISTRY = {
    "preproc": {
        "experiment": PreprocExperiment,
        "action": PreprocActionSeries,
    },
    "esam": {
        "experiment": ESAMExperiment,
        "action": ESAMActionSeries,
    },
}


    