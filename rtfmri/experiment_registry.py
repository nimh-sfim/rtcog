from rtfmri.experiment.preproc_experiment import PreprocExperiment
from rtfmri.experiment.esam_experiment import ESAMExperiment
from rtfmri.gui.preproc_gui import PreprocGUI
from rtfmri.gui.esam_gui import EsamGUI
from rtfmri.action.preproc_experiment_action import PreprocExperimentAction
from rtfmri.action.esam_experiment_action import ESAMExperimentAction

EXPERIMENT_REGISTRY = {
    "preproc": {
        "experiment": PreprocExperiment,
        "action": PreprocExperimentAction,
        "gui": PreprocGUI,
    },
    "esam": {
        "experiment": ESAMExperiment,
        "action": ESAMExperimentAction,
        "gui": EsamGUI,
    }
}
