from psychopy import core, event, gui, data, monitors #import some libraries from PsychoPy
from psychopy.hardware import keyboard
from psychopy import logging as psychopy_logging

#from psychopy.sound import Sound
from psychopy.visual import TextStim, Window, ImageStim, RatingScale
#from psychopy import microphone
from time import sleep
import time
import numpy as np
import os.path as osp
import csv
import logging
from playsound import playsound
from .recorder import Recorder

RESOURCES_DIR = '../../rtcaps/resources/'
ALERT_SOUND_FILE = 'bike_bell.wav'

log     = logging.getLogger("experiment_qa")
log.setLevel(logging.INFO)
log_fmt = logging.Formatter('[%(levelname)s - experiment_qa]: %(message)s')
log_ch  = logging.StreamHandler()
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
    expInfo = {'prefix':      opts.out_prefix,
               'out_dir':     opts.out_dir,
               'keyboard':    available_keyboards_labels,
               'screen':      ['Laptop','External'],
               'fullScreen':  ['Yes','No'],
               'leftKey':     '3',
               'rightKey':    '1',
               'acceptKey':   '2',
               'triggerKey':  '5'}
    dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title='rtCAPs Thought Sampling')
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    expInfo['date']    = data.getDateStr()
    kb_descriptor      = available_keyboards[available_keyboards_labels.index(expInfo['keyboard'])]
    return expInfo

class experiment_Preproc(object):
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

        self.ewin       = self._create_experiment_window()
        
        # Default Screen
        self.default_inst_01 = TextStim(win=self.ewin, text='Fixate on crosshair', pos=(0.0,0.42))
        self.default_inst_02 = TextStim(win=self.ewin, text='Let your mind wander freely', pos=(0.0,0.3))
        self.default_inst_03 = TextStim(win=self.ewin, text='Do not sleep', pos=(0.0,-0.3))
        self.default_chair   = TextStim(win=self.ewin, text='X', pos=(0,0))
    
    def _create_experiment_window(self):
        ewin = Window(
            fullscr=self.fscreen, screen=self.screen, size=(1920,1080),
            winType='pyglet', allowGUI=True, allowStencil=False,
            color=[0,0,0], colorSpace='rgb',
            blendMode='avg', useFBO=True, 
            units='norm')
        return ewin

    def draw_resting_screen(self):
        self.default_inst_01.draw()
        self.default_inst_02.draw()
        self.default_inst_03.draw()
        self.default_chair.draw()
        self.ewin.flip()
        return None
    
    def close_psychopy_infrastructure(self):
        log.info(' - close_psychopy_infrastructure - Function called.')
        self.ewin.flip()
        self.ewin.close()
        psychopy_logging.flush()
        core.quit()
        return None

class experiment_QA(object):
    def __init__(self, expInfo, opts):
        # Constants for easy configuration
        self.hitID = 1
        self.RS_Q_TIMEOUT = 20
        self.RS_Q_STRETCH = 2.5
        self.RS_Q_SHOW_ACCEPT = False
        self.RS_Q_MARKER_TYPE = 'glow'
        self.RS_Q_MARKER_COLOR = 'white'
        self.RS_Q_TEXT_SIZE    = 0.7
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

        self.ewin       = self._create_experiment_window()
        self.key_left   = expInfo['leftKey']
        self.key_right  = expInfo['rightKey']
        self.key_select = expInfo['acceptKey']
        self.likert_order = None
        
        # Default Screen
        self.default_inst_01 = TextStim(win=self.ewin, text='Fixate on crosshair', pos=(0.0,0.42))
        self.default_inst_02 = TextStim(win=self.ewin, text='Let your mind wander freely', pos=(0.0,0.3))
        self.default_inst_03 = TextStim(win=self.ewin, text='Do not sleep', pos=(0.0,-0.3))
        self.default_chair   = TextStim(win=self.ewin, text='X', pos=(0,0))

        # Beep / Recording Screen
        self.beep_inst_top_01  = TextStim(win=self.ewin, text='Describe aloud what you were', pos=(0.0, 0.54))
        self.beep_inst_top_02  = TextStim(win=self.ewin, text='thinking and doing', pos=(0.0,0.42))
        self.beep_inst_top_03  = TextStim(win=self.ewin, text='when you heard the beep', pos=(0.0,0.3))
        self.beep_chair        = TextStim(win=self.ewin, text='[ RECORDING ]', color='red', pos=(0.0,0.0), bold=True)
        self.beep_inst_bot_01  = TextStim(win=self.ewin, text='Press any key to stop recording', pos=(0.0,-0.3))
        self.beep_inst_bot_02  = TextStim(win=self.ewin, text='when you finish.', pos=(0.0,-0.42))
        self.mic_image         = ImageStim(win=self.ewin, image=osp.join(RESOURCES_DIR,'microphone_pic.png'), pos=(-0.5,0.0), size=(.2,.2))

        self.mic_ack_rec_01 = TextStim(win=self.ewin, text='Recoding Successful', pos=(0.0,0.06), color='green', bold=True)
        self.mic_ack_rec_02 = TextStim(win=self.ewin, text='Thank you!', pos=(0.0,-0.06), color='green', bold=True)
        self.likert_inst_01 = TextStim(win=self.ewin,text='Now, please use the response box\nto answer additional questions\nregarding what you were experiencing\nwhen you heard the beep', pos=(0.0,0.5), alignHoriz='center')

        #Likert Initial Instructions
        self.likert_qa_inst_01  = TextStim(win=self.ewin, text='Now, please use the response box', pos=(0.0, 0.78))
        self.likert_qa_inst_02  = TextStim(win=self.ewin, text='to answer additional questions',   pos=(0.0, 0.66))
        self.likert_qa_inst_03  = TextStim(win=self.ewin, text='about what you were experiencing',   pos=(0.0, 0.54))
        self.likert_qa_inst_04  = TextStim(win=self.ewin, text='right before the beep',      pos=(0.0, 0.42))
        self.likert_qa_inst_05  = TextStim(win=self.ewin, text='Press any key when ready',   pos=(0.0, -0.42))
        self.likert_qa_inst_img = ImageStim(win=self.ewin, image=osp.join(RESOURCES_DIR,'likert_instr.png'), pos=(0.0,0.0), size=(0.6,0.55))
        
        # Likert Questions
        self.likert_questions = {
        0: RatingScale(win=self.ewin,
                       scale="How alert were you?",
                       leftKeys=self.key_left, rightKeys=self.key_right, acceptKeys=self.key_select,
                       markerStart='0', marker=self.RS_Q_MARKER_TYPE, markerExpansion=0,markerColor=self.RS_Q_MARKER_COLOR,
                       choices=['Fully asleep','Somewhat sleepy','Somewhat alert','Fully alert'],
                       pos=(0.0,0.0), stretch=self.RS_Q_STRETCH, textColor='white', acceptPreText='Make a selection',
                       showAccept=self.RS_Q_SHOW_ACCEPT, maxTime=self.RS_Q_TIMEOUT, name='rs_alert', textSize=self.RS_Q_TEXT_SIZE),
        1: RatingScale(win=self.ewin,
                       scale="Were you moving any parts of your body (e.g. head, arm, leg, toes etc)?",
                       leftKeys=self.key_left, rightKeys=self.key_right, acceptKeys=self.key_select,
                       markerStart='0', marker=self.RS_Q_MARKER_TYPE, markerExpansion=0,markerColor=self.RS_Q_MARKER_COLOR,
                       choices=['Not sure','No / Disagree','Yes, a little','Yes, quite a bit', 'Yes, a lot'],
                       pos=(0.0,0.0), stretch=self.RS_Q_STRETCH, textColor='white', acceptPreText='Make a selection',
                       showAccept=self.RS_Q_SHOW_ACCEPT, maxTime=self.RS_Q_TIMEOUT, name='rs_motion', textSize=self.RS_Q_TEXT_SIZE),
        2: RatingScale(win=self.ewin,
                       scale="Was your attention focused on visual elements of the environment?",
                       leftKeys=self.key_left, rightKeys=self.key_right, acceptKeys=self.key_select,
                       markerStart='0', marker=self.RS_Q_MARKER_TYPE, markerExpansion=0,markerColor=self.RS_Q_MARKER_COLOR,
                       choices=['Strongly disagree','Somewhat disagree','Not sure','Somewhat agree', 'Strongly agree'],
                       pos=(0.0,0.0), stretch=self.RS_Q_STRETCH, textColor='white', acceptPreText='Make a selection',
                       showAccept=self.RS_Q_SHOW_ACCEPT, maxTime=self.RS_Q_TIMEOUT, name='rs_visual', textSize=self.RS_Q_TEXT_SIZE),
        3: RatingScale(win=self.ewin,
                       scale="Was your attention focused on auditory elements of the environment?",
                       leftKeys=self.key_left, rightKeys=self.key_right, acceptKeys=self.key_select,
                       markerStart='0', marker=self.RS_Q_MARKER_TYPE, markerExpansion=0,markerColor=self.RS_Q_MARKER_COLOR,
                       choices=['Strongly disagree','Somewhat disagree','Not sure','Somewhat agree', 'Strongly agree'],
                       pos=(0.0,0.0), stretch=self.RS_Q_STRETCH, textColor='white', acceptPreText='Make a selection',
                       showAccept=self.RS_Q_SHOW_ACCEPT, maxTime=self.RS_Q_TIMEOUT, name='rs_audio', textSize=self.RS_Q_TEXT_SIZE),
        4: RatingScale(win=self.ewin,
                       scale="Was your attention focused on tactile elements of the environment?",
                       leftKeys=self.key_left, rightKeys=self.key_right, acceptKeys=self.key_select,
                       markerStart='0', marker=self.RS_Q_MARKER_TYPE, markerExpansion=0,markerColor=self.RS_Q_MARKER_COLOR,
                       choices=['Strongly disagree','Somewhat disagree','Not sure','Somewhat agree', 'Strongly agree'],
                       pos=(0.0,0.0), stretch=self.RS_Q_STRETCH, textColor='white', acceptPreText='Make a selection',
                       showAccept=self.RS_Q_SHOW_ACCEPT, maxTime=self.RS_Q_TIMEOUT, name='rs_tactile', textSize=self.RS_Q_TEXT_SIZE),
        5: RatingScale(win=self.ewin,
                       scale="Was your attention focused on your internal world?",
                       leftKeys=self.key_left, rightKeys=self.key_right, acceptKeys=self.key_select,
                       markerStart='0', marker=self.RS_Q_MARKER_TYPE, markerExpansion=0,markerColor=self.RS_Q_MARKER_COLOR,
                       choices=['Strongly disagree','Somewhat disagree','Not sure','Somewhat agree', 'Strongly agree'],
                       pos=(0.0,0.0), stretch=self.RS_Q_STRETCH, textColor='white', acceptPreText='Make a selection',
                       showAccept=self.RS_Q_SHOW_ACCEPT, maxTime=self.RS_Q_TIMEOUT, name='rs_internal', textSize=self.RS_Q_TEXT_SIZE),
        6: RatingScale(win=self.ewin,
                       scale="Where in time was your attention focused?",
                       leftKeys=self.key_left, rightKeys=self.key_right, acceptKeys=self.key_select,
                       markerStart='0', marker=self.RS_Q_MARKER_TYPE, markerExpansion=0,markerColor=self.RS_Q_MARKER_COLOR,
                       choices=['No time\nin particular','Distant past\n(>1 day)','Near past\n(last 24h)','Present', 'Near future', 'Distant future'],
                       pos=(0.0,0.0), stretch=self.RS_Q_STRETCH, textColor='white', acceptPreText='Make a selection',
                       showAccept=self.RS_Q_SHOW_ACCEPT, maxTime=self.RS_Q_TIMEOUT, name='rs_time', textSize=self.RS_Q_TEXT_SIZE),
        7: RatingScale(win=self.ewin,
                       scale="What was the modality / sensory domain of your ongoing experience?",
                       leftKeys=self.key_left, rightKeys=self.key_right, acceptKeys=self.key_select,
                       markerStart='0', marker=self.RS_Q_MARKER_TYPE, markerExpansion=0,markerColor=self.RS_Q_MARKER_COLOR,
                       choices=['Exclusively\nin words','Mostly words\n& some imagery','Balance of\nwords & imagery','Mostly imagery\n& some words', 'Exclusively\nin imagery'],
                       pos=(0.0,0.0), stretch=self.RS_Q_STRETCH, textColor='white', acceptPreText='Make a selection',
                       showAccept=self.RS_Q_SHOW_ACCEPT, maxTime=self.RS_Q_TIMEOUT, name='rs_modality', textSize=self.RS_Q_TEXT_SIZE),
        8: RatingScale(win=self.ewin,
                       scale="What was the valence of your ongoing experience?",
                       leftKeys=self.key_left, rightKeys=self.key_right, acceptKeys=self.key_select,
                       markerStart='0', marker=self.RS_Q_MARKER_TYPE, markerExpansion=0,markerColor=self.RS_Q_MARKER_COLOR,
                       choices=['Very negative','Somewhat negative','Neutral','Somewhat positive', 'Very positive'],
                       pos=(0.0,0.0), stretch=self.RS_Q_STRETCH, textColor='white', acceptPreText='Make a selection',
                       showAccept=self.RS_Q_SHOW_ACCEPT, maxTime=self.RS_Q_TIMEOUT, name='rs_valence', textSize=self.RS_Q_TEXT_SIZE),
        9: RatingScale(win=self.ewin,
                       scale="Was your attention focused intentionally or unintentionally?",
                       leftKeys=self.key_left, rightKeys=self.key_right, acceptKeys=self.key_select,
                       markerStart='0', marker=self.RS_Q_MARKER_TYPE, markerExpansion=0,markerColor=self.RS_Q_MARKER_COLOR,
                       choices=['Intentionally','Unintentionally'],
                       pos=(0.0,0.0), stretch=self.RS_Q_STRETCH, textColor='white', acceptPreText='Make a selection',
                       showAccept=self.RS_Q_SHOW_ACCEPT, maxTime=self.RS_Q_TIMEOUT, name='rs_attention', textSize=self.RS_Q_TEXT_SIZE),
        10: RatingScale(win=self.ewin,
                       scale="Was your attention focused with or without awareness?",
                       leftKeys=self.key_left, rightKeys=self.key_right, acceptKeys=self.key_select,
                       markerStart='0', marker=self.RS_Q_MARKER_TYPE, markerExpansion=0,markerColor=self.RS_Q_MARKER_COLOR,
                       choices=['Not aware at all','Somewhat aware','Extremely aware'],
                       pos=(0.0,0.0), stretch=self.RS_Q_STRETCH, textColor='white', acceptPreText='Make a selection',
                       showAccept=self.RS_Q_SHOW_ACCEPT, maxTime=self.RS_Q_TIMEOUT, name='rs_attention_B', textSize=self.RS_Q_TEXT_SIZE),
}

    def _create_experiment_window(self):
        ewin = Window(
            fullscr=self.fscreen, screen=self.screen, size=(1920,1080),
            winType='pyglet', allowGUI=True, allowStencil=False,
            color=[0,0,0], colorSpace='rgb',
            blendMode='avg', useFBO=True, 
            units='norm')
        return ewin

    def draw_resting_screen(self):
        self.default_inst_01.draw()
        self.default_inst_02.draw()
        self.default_inst_03.draw()
        self.default_chair.draw()
        self.ewin.flip()
        return None
    
    def draw_alert_screen(self):
        self.beep_inst_top_01.draw()
        self.beep_inst_top_02.draw()
        self.beep_inst_top_03.draw()
        self.beep_inst_bot_01.draw()
        self.beep_inst_bot_02.draw()
        self.beep_chair.draw()
        self.mic_image.draw()
        self.ewin.flip()
        playsound(osp.join(RESOURCES_DIR,ALERT_SOUND_FILE))
        return None
    
    def record_oral_descr(self):
        rec = Recorder(channels=1)
        i = 0
        op = 1
        rec_path = osp.join(self.out_dir,self.out_prefix+'.hit'+str(self.hitID).zfill(3)+'.wav')
        with rec.open(rec_path,'wb') as rec_file:
            rec_file.start_recording()
            while not event.getKeys():
                i = i + 1
                if i > 5000:
                    i  = 0
                    op = int(not(op))
                    self.beep_inst_top_01.draw()
                    self.beep_inst_top_02.draw()
                    self.beep_inst_top_03.draw()
                    self.beep_chair = TextStim(win=self.ewin, text='[ RECORDING ]', color='red', pos=(0.0,0.0), bold=True, opacity=op)
                    self.beep_chair.draw()
                    self.beep_inst_bot_01.draw()
                    self.beep_inst_bot_02.draw()
                    self.mic_image.draw()
                    self.ewin.flip()
            rec_file.stop_recording()
        return None
    
    def draw_ack_recording_screen(self):
        self.mic_ack_rec_01.draw()
        self.mic_ack_rec_02.draw()
        self.ewin.flip()
        sleep(1)
        return None
        
    def close_psychopy_infrastructure(self):
        log.info(' - close_psychopy_infrastructure - Function called.')
        self.ewin.flip()
        self.ewin.close()
        psychopy_logging.flush()
        core.quit()
        return None
        
    def draw_likert_instructions(self):
        self.likert_qa_inst_01.draw()
        self.likert_qa_inst_02.draw()
        self.likert_qa_inst_03.draw()
        self.likert_qa_inst_04.draw()
        self.likert_qa_inst_05.draw()
        self.likert_qa_inst_img.draw()
        self.ewin.flip()
        
        while not event.getKeys():
            sleep(0.01)
            
    def draw_likert_questions(self, order=None):
        responses = {}
        if order == None:
            order = np.arange(len(self.likert_questions))
        for q_idx in order:
            aux_q = self.likert_questions[q_idx]
            aux_q.reset()
            event.clearEvents()
            while aux_q.noResponse:
                aux_q.draw()
                self.ewin.flip()
                if event.getKeys(['escape']):
                    core.quit()
            responses[aux_q.name] = [aux_q.getRating(), aux_q.getRT(), aux_q.getHistory()]
        return responses

    def run_full_QA(self):
        # 1) Play beep and instruct subject to talk
        self.draw_alert_screen()
        
        # 2) Record oral description
        event.clearEvents()
        self.record_oral_descr()
        
        # 3) Acknowledge successful recording
        self.draw_ack_recording_screen()
        
        # 4) Show instructions for likert part of QA
        self.draw_likert_instructions()
        
        # 5) Do the Likert Questionare
        resp_dict = self.draw_likert_questions(self.likert_order)
        resp_timestr = time.strftime("%Y%m%d-%H%M%S")
        resp_path = osp.join(self.out_dir,self.out_prefix+'.'+resp_timestr+'.LikertResponses'+str(self.hitID).zfill(3)+'.txt')
        w = csv.writer(open(resp_path, "w"))
        for key, val in resp_dict.items():
            w.writerow([key, val])
        self.hitID = self.hitID + 1
        return resp_dict
