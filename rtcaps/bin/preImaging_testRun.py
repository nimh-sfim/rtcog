import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '/data/SFIMJGC/PRJ_rtCAPs/rtcaps/')))
from rtcap_lib.experiment_qa import get_experiment_info, experiment_QA
from psychopy import event, core
from time import sleep
import logging

# Setup Logging Infrastructure
log     = logging.getLogger("preImaging_testRun")
log_fmt = logging.Formatter('[%(levelname)s - Main]: %(message)s')
log_ch  = logging.StreamHandler()
log_ch.setFormatter(log_fmt)
log.setLevel(logging.INFO)
log.addHandler(log_ch)

class fakeOpts(object):
    def __init__(self):
        self.out_prefix = 'testRun'
        self.out_dir    = '/data/TMP'
        

# Options Dialog
opts     = fakeOpts()
exp_info = get_experiment_info(opts)

# Create QA Class
cap_qa = experiment_QA(exp_info,opts)

# Show Rest Screen for 5s
s = 0
while s < 5:
    cap_qa.draw_resting_screen()
    if event.getKeys(['escape']):
        log.info('- User pressed escape key')
        core.quit()
    sleep(1)
    s = s + 1

# Record Responses
responses = cap_qa.run_full_QA()
log.info(' - Responses: %s' % str(responses))

# Close everything
cap_qa.close_psychopy_infrastructure()
