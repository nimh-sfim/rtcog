import time
import os.path as osp
import csv
from playsound import playsound

from psychopy import core, event
from psychopy.visual import TextStim, ImageStim
from psychopy.visual.slider import Slider 
from psychopy import prefs
prefs.hardware['keyboard'] = 'pygame'

from rtfmri.utils.recorder import Recorder
from rtfmri.utils.core import get_logger
from rtfmri.paths import RESOURCES_DIR
from rtfmri.gui.default_gui import DefaultGUI

log = get_logger()

class EsamGUI(DefaultGUI):
    """
    GUI class for Experience Sampling (ESAM) fMRI experiments.

    Extends `DefaultGUI` to provide oral recording and Likert-style questionnaires
    after a template "hit".

    Parameters
    ----------
    expInfo : dict
        Experiment info with screen and key configurations.
    opts : Options
        Configuration options for the experiment run.
    shared_responses : multiprocessing.Manager().dict
        Shared dictionary for returning participant responses.
    clock : SharedClock, optional
        Clock for timing events during latency testing.
    """
    def __init__(self, expInfo, opts, shared_responses, clock=None):
        super().__init__(expInfo, opts, clock)
        self.hitID = 1

        self.key_left   = expInfo['leftKey']
        self.key_right  = expInfo['rightKey']
        self.key_select = expInfo['acceptKey']
        self.red_color = [0.4, -0.9, -0.9]
        self.likert_order = None

        self.recorder = Recorder(channels=1)

        self.responses = {}
        self._shared_responses = shared_responses

        # Recording Screen
        self.rec_inst = [
            TextStim(win=self.ewin, text='Describe aloud what you were', pos=(0.0, 0.54)),
            TextStim(win=self.ewin, text='when you heard the beep', pos=(0.0,0.3)),
            TextStim(win=self.ewin, text='Press any key to stop recording', pos=(0.0,-0.3)),
            TextStim(win=self.ewin, text='when you finish.', pos=(0.0,-0.42)),
            ImageStim(win=self.ewin, image=osp.join(RESOURCES_DIR,'microphone_pic.png'), pos=(-0.5,0.0), size=(.2,.2))
        ]

        self.rec_chair = TextStim(win=self.ewin, text='[ RECORDING ]', color=self.red_color, pos=(0.0,0.0), bold=True)

        # Post-recording screen
        self.mic_ack_rec = [
            TextStim(win=self.ewin, text='Recoding Successful', pos=(0.0,0.06), color='green', bold=True),
            TextStim(win=self.ewin, text='Thank you!', pos=(0.0,-0.06), color='green', bold=True)
        ]

        # Likert Instructions
        self.likert_qa_inst = [
            TextStim(win=self.ewin, text='Now, please use the response box', pos=(0.0, 0.78)),
            TextStim(win=self.ewin, text='to answer additional questions',   pos=(0.0, 0.66)),
            TextStim(win=self.ewin, text='about what you were experiencing',   pos=(0.0, 0.54)),
            TextStim(win=self.ewin, text='right before the beep',      pos=(0.0, 0.42)),
            TextStim(win=self.ewin, text='Press any key when ready',   pos=(0.0, -0.42)),
            ImageStim(win=self.ewin, image=osp.join(RESOURCES_DIR,'resp_box_form.png'), pos=(0.0,0.0), size=(0.6,0.55))
        ]

        # Likert Questions
        self.likert_questions = opts.likert_questions

        # Likert slider opts
        self.slider_opts = {
            'win': self.ewin,
            'pos': (0, -0.1),
            'size': (1.2, 0.2),
            'labelColor': 'black',
            'markerColor': self.red_color,
            'lineColor': 'black',
            'granularity': 1,
            'labelHeight': 0.05,
             'font': 'Arial'
        }

    def record_oral_descr(self):
        """
        Record the participant’s oral description after a hit.

        Displays recording screen and plays a sound cue.
        Saves audio to a WAV file.
        """
        event.clearEvents()
        self._draw_stims(self.rec_inst + [self.rec_chair])
        playsound(osp.join(RESOURCES_DIR, 'bike_bell.wav'))

        rec_path = osp.join(self.out_dir, self.out_prefix + '.hit' + str(self.hitID).zfill(3) + '.wav')

        with self.recorder.open(rec_path, 'wb') as rec_file:
            rec_file.start_recording()

            clock = core.Clock()
            toggle_interval = 0.6
            last_toggle_time = 0

            while not event.getKeys():
                # Toggle [ RECORDING ] text every 0.6sec
                if clock.getTime() - last_toggle_time >= toggle_interval:
                    last_toggle_time = clock.getTime()
                    self.rec_chair.text = "[ RECORDING ]" if not self.rec_chair.text else ""

                self._draw_stims(self.rec_inst + [self.rec_chair])
                
                core.wait(0.1)

            rec_file.stop_recording()
    
    def draw_ack_recording_screen(self):
        """
        Display a confirmation screen after recording completes.
        """
        self._draw_stims(self.mic_ack_rec)
        time.sleep(1)
        
    def draw_likert_instructions(self):
        """
        Show instructions before presenting the Likert questions.
        """
        self._draw_stims(self.likert_qa_inst)
        event.waitKeys()

    def draw_likert_questions(self, order=None):
        """
        Display a sequence of Likert-style questions for participant response.

        Parameters
        ----------
        order : list of int, optional
            Order in which to present the questions. Defaults to sequential.

        Returns
        -------
        dict
            Dictionary of {question_name: (rating, rt)} for each response.
        """
        responses = {}
        if order is None:
            order = range(len(self.likert_questions))
        
        for q_idx in order:
            q = self.likert_questions[q_idx]
            
            q_text = TextStim(win=self.ewin, text=q['text'], pos=(0.0, 0.2), color='black',height=0.1)

            labels = q.get('labels', ['Strongly\ndisagree', 'Somewhat\ndisagree', 'Neutral', 'Somewhat agree', 'Strongly agree']) # Fall back on default labels
            ticks = list(range(1, len(labels) + 1))

            slider = Slider(
                **self.slider_opts,
                ticks=ticks,
                labels=labels,
                startValue=(len(labels) // 2) + 1
            )

            slider.markerPos = slider.startValue
            current_pos = slider.startValue

            rating = None
            event.clearEvents()

            slider.markerPos = current_pos
            self._draw_stims([q_text, slider])
            clock = core.Clock()
            
            while True:
                keys = event.getKeys()
                for key in keys:
                    if key == self.key_left and current_pos > ticks[0]:
                        current_pos -= 1
                    elif key == self.key_right and current_pos < ticks[-1]:
                        current_pos += 1
                    elif key == self.key_select:
                        rating = current_pos
                        break
                    elif key in ['escape', 'q']:
                        self.ewin.close()
                        core.quit()
                
                slider.markerPos = current_pos
                self._draw_stims([q_text, slider])

                if rating is not None:
                    rt = clock.getTime()
                    break

            responses[q['name']] = (rating, rt)
            
            core.wait(0.5)
            self.ewin.flip()

        return responses


    def run_full_QA(self):
        """
        Run the full QA block: record voice, show Likert questions, and save results.

        Returns
        -------
        dict
            Dictionary of Likert responses for this QA instance.
        """
        # 1) Play beep and record oral description
        self.record_oral_descr()
        
        # 2) Acknowledge successful recording
        self.draw_ack_recording_screen()
        
        # 3) Show instructions for likert part of QA
        self.draw_likert_instructions()
        
        # 4) Do the Likert questionnaire
        resp_dict = self.draw_likert_questions(self.likert_order)

        # 5) Prepare results
        resp_timestr = time.strftime('%Y%m%d-%H%M%S')
        resp_path = osp.join(self.out_dir, f'{self.out_prefix}.{resp_timestr}.LikertResponses{str(self.hitID).zfill(3)}.txt')
        
        self._shared_responses.clear()
        self._shared_responses.update(resp_dict)

        self.responses[resp_path] = resp_dict
        
        self.hitID += 1

        return resp_dict
    
    def save_likert_files(self):
        """
        Write all stored Likert response dictionaries to individual text files.
        """
        for resp_path, resp_dict in self.responses.items():
            with open (resp_path, 'w') as f:
                w = csv.writer(f)
                w.writerow(['question', 'rating', 'rt'])
                for key, val in resp_dict.items():
                    rating, rt = val
                    w.writerow([key, rating, round(rt, 2)])
                log.debug(f'Likert responses written to {resp_path}')
        if self.responses:
            log.info(f'All likert responses saved')
            