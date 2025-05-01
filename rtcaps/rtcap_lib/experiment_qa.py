from time import sleep
import time
import os.path as osp
import csv
import logging
from playsound import playsound
from .recorder import Recorder

from psychopy import core, event, gui, data
from psychopy.hardware import keyboard
from psychopy import logging as psychopy_logging
from psychopy.visual import TextStim, Window, ImageStim
from psychopy.visual.slider import Slider 


from config import RESOURCES_DIR
# this_dir = osp.dirname(osp.realpath(__file__))
# code_dir = osp.abspath(osp.join(this_dir, '..'))
# RESOURCES_DIR = osp.join(code_dir, 'resources/')

log = logging.getLogger("experiment_qa")
log.setLevel(logging.INFO)
log_fmt = logging.Formatter('[%(levelname)s - experiment_qa]: %(message)s')
log_ch = logging.StreamHandler()
log_ch.setFormatter(log_fmt)
log_ch.setLevel(logging.INFO)
log.addHandler(log_ch)

def get_avail_keyboards():
    available_keyboards = keyboard.getKeyboards()
    available_keyboards_labels = []
    for kb in available_keyboards:
        if kb['product'] == '': 
            available_keyboards_labels.append('Laptop Keyboard')
        else:
            available_keyboards_labels.append(kb['product'])
    return available_keyboards, available_keyboards_labels

def get_experiment_info(opts):
    available_keyboards, available_keyboards_labels = get_avail_keyboards()
    expInfo = {
        'prefix':      opts.out_prefix,
        'out_dir':     opts.out_dir,
        'keyboard':    available_keyboards_labels,
        'screen':      ['Laptop','External'],
        'fullScreen':  ['Yes','No'],
        'leftKey':     '3',
        'rightKey':    '1',
        'acceptKey':   '2',
        'triggerKey':  '5'
    }
    dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title='rtCAPs Thought Sampling')
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    expInfo['date'] = data.getDateStr()
    return expInfo

class DefaultScreen:
    def __init__(self, expInfo, opts):
        self.out_dir    = opts.out_dir
        self.out_prefix = opts.out_prefix

        if expInfo['fullScreen'] == 'Yes':
            self.fscreen = True
        else:
            self.fscreen = False

        if expInfo['screen'] == 'Laptop':
            self.screen = 0
        if expInfo['screen'] == 'External':
            self.screen = 1

        self.ewin = self._create_experiment_window()
        
        # Default Screen
        self.default_inst = [
            TextStim(win=self.ewin, text='Fixate on crosshair', pos=(0.0,0.42)),
            TextStim(win=self.ewin, text='Let your mind wander freely', pos=(0.0,0.3)),
            TextStim(win=self.ewin, text='Do not sleep', pos=(0.0,-0.3)),
            TextStim(win=self.ewin, text='X', pos=(0,0))
        ]
   
    def _create_experiment_window(self):
        return Window(
            fullscr=self.fscreen, screen=self.screen, size=(1920,1080),
            winType='pyglet', allowGUI=True, allowStencil=False,
            color=[0,0,0], colorSpace='rgb', blendMode='avg',
            useFBO=True, units='norm'
        )
    
    def _draw_stims(self, stims, flip=True):
        for stim in stims:
            stim.draw()
        if flip:
            self.ewin.flip()

    def draw_resting_screen(self):
        self._draw_stims(self.default_inst)
    
    def close_psychopy_infrastructure(self):
        log.info(' - close_psychopy_infrastructure - Function called.')
        self.ewin.flip()
        self.ewin.close()
        psychopy_logging.flush()
        core.quit()


class QAScreen(DefaultScreen):
    def __init__(self, expInfo, opts):
        super().__init__(expInfo, opts)
        self.hitID = 1

        self.key_left   = expInfo['leftKey']
        self.key_right  = expInfo['rightKey']
        self.key_select = expInfo['acceptKey']
        self.red_color = [0.4, -0.9, -0.9]
        self.likert_order = None

        self.recorder = Recorder(channels=1)

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
            'labelHeight': 0.05
        }

    def record_oral_descr(self):
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
        self._draw_stims(self.mic_ack_rec)
        sleep(1)
        
    def draw_likert_instructions(self):
        self._draw_stims(self.likert_qa_inst)
        event.waitKeys()

    def draw_likert_questions(self, order=None):
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
        # 1) Play beep and record oral description
        self.record_oral_descr()
        
        # 2) Acknowledge successful recording
        self.draw_ack_recording_screen()
        
        # 3) Show instructions for likert part of QA
        self.draw_likert_instructions()
        
        # 4) Do the Likert questionnaire
        resp_dict = self.draw_likert_questions(self.likert_order)

        # 5) Write results to file
        resp_timestr = time.strftime('%Y%m%d-%H%M%S')
        resp_path = osp.join(self.out_dir, f'{self.out_prefix}.{resp_timestr}.LikertResponses{str(self.hitID).zfill(3)}.txt')
        
        with open(resp_path, 'w') as f:
            w = csv.writer(f)
            w.writerow(['question', 'rating', 'rt'])
            for key, val in resp_dict.items():
                rating, rt = val
                w.writerow([key, rating, round(rt, 2)])
        
        log.info(f'Likert responses written to {resp_path}')
        
        self.hitID += 1

        return resp_dict
