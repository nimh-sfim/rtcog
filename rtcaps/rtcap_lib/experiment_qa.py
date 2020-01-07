from psychopy import core, event, gui, data, monitors #import some libraries from PsychoPy
from psychopy.hardware import keyboard
from psychopy import logging as psychopy_logging

from psychopy.sound import Sound
from psychopy.visual import TextStim, Window, ImageStim, RatingScale
from psychopy import microphone
from time import sleep
import time
import numpy as np
import os.path as osp
import csv
import logging

RESOURCES_DIR = '../../rtcaps/resources/'

log = logging.getLogger("experiment_qa")
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

def get_experiment_info():
    available_keyboards, available_keyboards_labels = get_avail_keyboards()
    available_monitors = monitors.getAllMonitors()
    expInfo = {'participant': 'rtcsbj', 'run': '001','keyboard':available_keyboards_labels,'monitor':available_monitors}
    dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title='rtCAPs Thought Sampling')
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    expInfo['date']    = data.getDateStr()
    kb_descriptor      = available_keyboards[available_keyboards_labels.index(expInfo['keyboard'])]
    monitor_descriptor ='testMonitor'
    return expInfo, kb_descriptor, monitor_descriptor

class experiment_QA(object):
    def __init__(self,kb, monitor, opts):
        # Constants for easy configuration
        self.RS_Q_TIMEOUT = 20
        self.RS_Q_STRETCH = 2.5
        self.RS_Q_SHOW_ACCEPT = False
        self.RS_Q_MARKER_TYPE = 'glow'
        self.RS_Q_MARKER_COLOR = 'white'
        self.out_dir    = opts.out_dir
        self.out_prefix = opts.out_prefix
        self.fscreen    = opts.fullscreen
        self.ewin       = self._create_experiment_window(monitor)
        self.kb         = kb
        self.monitor    = monitor
        self.key_left   = 'a'
        self.key_right  = 's'
        self.key_select = ['w','z']
        self.likert_order = None
        
        # Default Screen
        self.default_inst_01 = TextStim(win=self.ewin, text='Fixate on crosshair', pos=(0.0,0.42))
        self.default_inst_02 = TextStim(win=self.ewin, text='Let you mind wander freely', pos=(0.0,0.3))
        self.default_inst_03 = TextStim(win=self.ewin, text='Do not sleep', pos=(0.0,-0.3))
        self.default_chair   = TextStim(win=self.ewin, text='X', pos=(0,0))

        # Beep / Recording Screen
        self.beep_inst_top_01  = TextStim(win=self.ewin, text='Describe aloud what you were', pos=(0.0, 0.54))
        self.beep_inst_top_02  = TextStim(win=self.ewin, text='thinking and doing', pos=(0.0,0.42))
        self.beep_inst_top_03  = TextStim(win=self.ewin, text='when you heard the beep', pos=(0.0,0.3))
        self.beep_chair        = TextStim(win=self.ewin, text='[ RECORDING ]', color='red', pos=(0.0,0.0), bold=True)
        self.beep_inst_bot_01  = TextStim(win=self.ewin, text='Press any key to stop recording', pos=(0.0,-0.3))
        self.beep_inst_bot_02  = TextStim(win=self.ewin, text='when you finish.', pos=(0.0,-0.42))
        self.beep_sound        = Sound(osp.join(RESOURCES_DIR,'beep.wav'))
        self.mic_image         = ImageStim(win=self.ewin, image=osp.join(RESOURCES_DIR,'microphone_pic.png'), pos=(-0.5,0.0), size=(.2,.2))

        microphone.switchOn()
        #self.mic            = microphone.AdvAudioCapture(saveDir=self.out_dir, filename=self.out_prefix+'_OralResponse')
        self.mic            = microphone.AudioCapture(saveDir=self.out_dir,filename=self.out_prefix+'_OralResponse')
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
                       scale="What emotions (if any) were associated with your thoughts?",
                       leftKeys=self.key_left, rightKeys=self.key_right, acceptKeys=self.key_select,
                       markerStart='0', marker=self.RS_Q_MARKER_TYPE, markerExpansion=0,markerColor=self.RS_Q_MARKER_COLOR,
                       choices=['Very Sad','Sad','Neutral/None','Happy','Very Happy'],
                       pos=(0.0,0.0), stretch=self.RS_Q_STRETCH, textColor='white', acceptPreText='Make a selection',
                       showAccept=self.RS_Q_SHOW_ACCEPT, maxTime=self.RS_Q_TIMEOUT, name='rs_emotion'),
        1: RatingScale(win=self.ewin,
                       scale="Were you moving any part of your body?",
                       leftKeys=self.key_left, rightKeys=self.key_right, acceptKeys=self.key_select,
                       markerStart='0', marker=self.RS_Q_MARKER_TYPE, markerExpansion=0,markerColor=self.RS_Q_MARKER_COLOR,
                       low = 0, high = 4,
                       labels=['Strongly Disagree','Strongly Agree'],
                       pos=(0.0,0.0), stretch=self.RS_Q_STRETCH, textColor='white', acceptPreText='Make a selection',
                       showAccept=self.RS_Q_SHOW_ACCEPT, maxTime=self.RS_Q_TIMEOUT, name='rs_motion'),
        2: RatingScale(win=self.ewin,
                       scale="Was your attention focused on visual elements of the environment?",
                       leftKeys=self.key_left, rightKeys=self.key_right, acceptKeys=self.key_select,
                       markerStart='0', marker=self.RS_Q_MARKER_TYPE, markerExpansion=0,markerColor=self.RS_Q_MARKER_COLOR,
                       low = 0, high = 4,
                       labels=['Strongly Disagree','Strongly Agree'],
                       pos=(0.0,0.0), stretch=self.RS_Q_STRETCH, textColor='white', acceptPreText='Make a selection',
                       showAccept=self.RS_Q_SHOW_ACCEPT, maxTime=self.RS_Q_TIMEOUT, name='rs_visual'),
        3: RatingScale(win=self.ewin,
                       scale="Was your attention focused on auditory elements of the environment?",
                       leftKeys=self.key_left, rightKeys=self.key_right, acceptKeys=self.key_select,
                       markerStart='0', marker=self.RS_Q_MARKER_TYPE, markerExpansion=0,markerColor=self.RS_Q_MARKER_COLOR,
                       low = 0, high = 4,
                       labels=['Strongly Disagree','Strongly Agree'],
                       pos=(0.0,0.0), stretch=self.RS_Q_STRETCH, textColor='white', acceptPreText='Make a selection',
                       showAccept=self.RS_Q_SHOW_ACCEPT, maxTime=self.RS_Q_TIMEOUT, name='rs_audio'),
        4: RatingScale(win=self.ewin,
                       scale="Was your attention focused on tactile elements of the environment?",
                       leftKeys=self.key_left, rightKeys=self.key_right, acceptKeys=self.key_select,
                       markerStart='0', marker=self.RS_Q_MARKER_TYPE, markerExpansion=0,markerColor=self.RS_Q_MARKER_COLOR,
                       low = 0, high = 4,
                       labels=['Strongly Disagree','Strongly Agree'],
                       pos=(0.0,0.0), stretch=self.RS_Q_STRETCH, textColor='white', acceptPreText='Make a selection',
                       showAccept=self.RS_Q_SHOW_ACCEPT, maxTime=self.RS_Q_TIMEOUT, name='rs_somatosensory'),
        5: RatingScale(win=self.ewin,
                       scale="How alert were you?",
                       leftKeys=self.key_left, rightKeys=self.key_right, acceptKeys=self.key_select,
                       markerStart='0', marker=self.RS_Q_MARKER_TYPE, markerExpansion=0,markerColor=self.RS_Q_MARKER_COLOR,
                       low = 0, high = 4,
                       labels=['Sleeping','Dizzy','Fully Alert'],
                       pos=(0.0,0.0), stretch=self.RS_Q_STRETCH, textColor='white', acceptPreText='Make a selection',
                       showAccept=self.RS_Q_SHOW_ACCEPT, maxTime=self.RS_Q_TIMEOUT, name='rs_wakefulness'),
        6: RatingScale(win=self.ewin,
                       scale="Were your thoughts focues on a given moment in time?",
                       leftKeys=self.key_left, rightKeys=self.key_right, acceptKeys=self.key_select,
                       markerStart='0', marker=self.RS_Q_MARKER_TYPE, markerExpansion=0,markerColor=self.RS_Q_MARKER_COLOR,
                       choices=['N/A','Distant Past','Past','Present','Future','Distant Future'],
                       pos=(0.0,0.0), stretch=self.RS_Q_STRETCH, textColor='white', acceptPreText='Make a selection',
                       showAccept=self.RS_Q_SHOW_ACCEPT, maxTime=self.RS_Q_TIMEOUT, name='rs_time'),
        7: RatingScale(win=self.ewin,
                       scale="What was the primary form of your thoughts?",
                       leftKeys=self.key_left, rightKeys=self.key_right, acceptKeys=self.key_select,
                       markerStart='0', marker=self.RS_Q_MARKER_TYPE, markerExpansion=0,markerColor=self.RS_Q_MARKER_COLOR,
                       low = 0, high = 4,
                       labels=['Exclusively\nin Words','Mix of\nWords & Images','Exclusively\nin Images'],
                       pos=(0.0,0.0), stretch=self.RS_Q_STRETCH, textColor='white', acceptPreText='Make a selection',
                       showAccept=self.RS_Q_SHOW_ACCEPT, maxTime=self.RS_Q_TIMEOUT, name='rs_form'),
        }

    def _create_experiment_window(self,monitor):
        ewin = Window(
            size=(1024, 768), fullscr=self.fscreen, screen=0, 
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor=monitor, color=[0,0,0], colorSpace='rgb',
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
        self.beep_sound.play()
        self.beep_inst_top_01.draw()
        self.beep_inst_top_02.draw()
        self.beep_inst_top_03.draw()
        self.beep_inst_bot_01.draw()
        self.beep_inst_bot_02.draw()
        self.beep_chair.draw()
        self.mic_image.draw()
        self.ewin.flip()
        return None
    
    def record_oral_descr(self):
        self.mic.reset()
        self.mic.record(60, block=False)
        i  = 0
        op = 1
        while self.mic.recorder.running:
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
            if event.getKeys():
                self.mic.stop()

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
        resp_path = osp.join(self.out_dir,self.out_prefix+'.'+resp_timestr+'.LikertResponses.txt')
        w = csv.writer(open(resp_path, "w"))
        for key, val in resp_dict.items():
            w.writerow([key, val])
        return resp_dict
