from rtfmri.experiment.preproc_experiment import PreprocExperiment
from rtfmri.experiment.esam_experiment import ESAMExperiment
from rtfmri.gui.preproc_gui import PreprocGUI
from rtfmri.gui.esam_gui import EsamGUI
from rtfmri.controller.preproc_controller import PreprocController
from rtfmri.controller.esam_controller import ESAMController

EXPERIMENT_REGISTRY = {
    "preproc": {
        "experiment": PreprocExperiment,
        "controller": PreprocController,
        "gui": PreprocGUI,
    },
    "esam": {
        "experiment": ESAMExperiment,
        "controller": ESAMController,
        "gui": EsamGUI,
    }
}
