from psychopy.visual import TextStim, Window, ImageStim, RatingScale, Rect, PatchStim
from psychopy.hardware import keyboard
from psychopy import core, event, gui, data, monitors #import some libraries from PsychoPy
import numpy as np
from time import sleep
import os.path as osp
from playsound import playsound
from .recorder import Recorder

RESOURCES_DIR = '../../rtcaps/resources/'
ALERT_SOUND_FILE = 'bike_bell.wav'

log     = logging.getLogger("experiment_qa_v2")
log.setLevel(logging.INFO)
log_fmt = logging.Formatter('[%(levelname)s - experiment_qa_v2]: %(message)s')
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
               'upKey':       '2',
               'downKey':     '4',
               'triggerKey':  '5'}
    dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title='rtCAPs Thought Sampling')
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    expInfo['date']    = data.getDateStr()
    kb_descriptor      = available_keyboards[available_keyboards_labels.index(expInfo['keyboard'])]
    return expInfo

class experimentQA_v2(object):
    def __init__(self, expInfo, opts)):
        self.INSTRUCTIONS_TEXT_HEIGHT = 0.08
        self.RS_Q_TEXT_HEIGHT         = 0.8
        self.RS_Q_TIMEOUT             = 30
        self.RS_Q_STRETCH             = 2.5
        self.RS_Q_SHOW_ACCEPT         = False
        self.RS_Q_MARKER_TYPE         = 'glow'
        self.RS_Q_MARKER_COLOR        = 'white'
        self.FORMS_MARKER_COLOR       = 'red'
        
        # Output Information
        self.out_dir    = opts.out_dir
        self.out_prefix = opts.out_prefix

        # Screen Information
        if expInfo['fullScreen'] == 'Yes':
            self.fscreen = True
        else:
            self.fscreen = False
        if expInfo['screen'] == 'Laptop':
            self.screen = 0
        if expInfo['screen'] == 'External':
            self.screen = 1

        # Create Experimental Window
        self.ewin       = self._create_experiment_window()
        
        # Button Box Keys
        self.key_left   = expInfo['leftKey']
        self.key_right  = expInfo['rightKey']
        self.key_down   = expInfo['downKey']
        self.key_up     = expInfo['upKey']
        
        # Default Screen
        self.default_inst_01 = TextStim(win=self.ewin, height=self.INSTRUCTIONS_TEXT_HEIGHT,  text='Fixate on the crosshair', pos=(0.0,0.55))
        self.default_inst_02 = TextStim(win=self.ewin, height=self.INSTRUCTIONS_TEXT_HEIGHT,  text='Let your mind wander freely to any', pos=(0.0,0.4))
        self.default_inst_03 = TextStim(win=self.ewin, height=self.INSTRUCTIONS_TEXT_HEIGHT,  text='thoughts or sensations that come up.', pos=(0.0,0.25))
        self.default_inst_04 = TextStim(win=self.ewin, height=self.INSTRUCTIONS_TEXT_HEIGHT,  text='Do not sleep', pos=(0.0,-0.25))
        self.default_chair   = TextStim(win=self.ewin, text='X', pos=(0,0))
        
        # Beep / Recording Screen
        self.beep_inst_cartoon = ImageStim(win=self.ewin, image=osp.join(RESOURCES_DIR,'report_period_graph.png'), pos=(+0.00,+0.05), size=(.5,.3))
        self.beep_inst_top_01  = TextStim(win=self.ewin, height=self.INSTRUCTIONS_TEXT_HEIGHT, pos=(+0.00, +0.66), text='Describe out loud your ongoing')
        self.beep_inst_top_02  = TextStim(win=self.ewin, height=self.INSTRUCTIONS_TEXT_HEIGHT, pos=(+0.00, +0.54), text='experience within the last 5 to 10 seconds')
        self.beep_inst_top_03  = TextStim(win=self.ewin, height=self.INSTRUCTIONS_TEXT_HEIGHT, pos=(+0.00, +0.42), text='when you heard the beep, including:')
        self.beep_inst_top_04  = TextStim(win=self.ewin, height=self.INSTRUCTIONS_TEXT_HEIGHT, pos=(+0.00, +0.30), text='thoughts, sensations, and feelings.')
        self.beep_chair        = TextStim(win=self.ewin, text='[ RECORDING ]', color='red', pos=(0.0,-0.20), bold=True)
        self.beep_inst_bot_01  = TextStim(win=self.ewin, height=self.INSTRUCTIONS_TEXT_HEIGHT, pos=(+0.00, -0.42),text='Press any button when you are')
        self.beep_inst_bot_02  = TextStim(win=self.ewin, height=self.INSTRUCTIONS_TEXT_HEIGHT, pos=(+0.00, -0.54),text='finished with recording.')

        # Recording Successful / General Instructions
        self.mic_ack_rec_01 = TextStim(win=self.ewin, text='Recoding Successful', pos=(0.0,0.70), color='green', bold=True)
        self.mic_ack_rec_02 = TextStim(win=self.ewin, text='Thank you!', pos=(0.0,0.58), color='green', bold=True)
        self.q_gen_instr_01 = TextStim(win=self.ewin, height=self.INSTRUCTIONS_TEXT_HEIGHT, pos=(+0.00, +0.40), text='We will now ask you some questions about your')
        self.q_gen_instr_02 = TextStim(win=self.ewin, height=self.INSTRUCTIONS_TEXT_HEIGHT, pos=(+0.00, +0.28), text='ongoing experience within the last 5-10 seconds.')
        self.q_gen_instr_03 = TextStim(win=self.ewin, height=self.INSTRUCTIONS_TEXT_HEIGHT, pos=(+0.00, -0.20), text='Use the response box to choose answers that')
        self.q_gen_instr_04 = TextStim(win=self.ewin, height=self.INSTRUCTIONS_TEXT_HEIGHT, pos=(+0.00, -0.32), text='most accurately characterizes your experience.')
        self.q_gen_instr_05 = TextStim(win=self.ewin, height=self.INSTRUCTIONS_TEXT_HEIGHT, pos=(+0.00, -0.85), text='Press any button when ready')
        self.q_gen_cartoon  = ImageStim(win=self.ewin, image=osp.join(RESOURCES_DIR,'report_period_graph.png'), pos=(+0.00,+0.05), size=(.5,.3))
        self.q_gen_rbox     = ImageStim(win=self.ewin, image=osp.join(RESOURCES_DIR,'resp_box_notext.png'), pos=(+0.00,-0.60), size=(.2,.3))
        
        # First Set of Generic Questions
        self.likert_qs_cartoon   = ImageStim(win=self.ewin, image=osp.join(RESOURCES_DIR,'report_period_graph.png'), pos=(0.0,+0.5), size=(.5,.3))
        self.likert_qs_rbox      = ImageStim(win=self.ewin, image=osp.join(RESOURCES_DIR,'resp_box_single_likert.png'), pos=(0.0,-0.5), size=(.2,.3))
        self.likert_qs_genset_01 = {
            0: RatingScale(win=self.ewin, scale="How alert were you?",
                       leftKeys=self.key_left, rightKeys=self.key_right, acceptKeys=[self.key_up, self.key_down],
                       markerStart='0', marker=self.RS_Q_MARKER_TYPE, markerExpansion=0,markerColor=self.RS_Q_MARKER_COLOR,
                       choices=['Fully asleep','Somewhat sleepy','Somewhat alert','Fully alert'],
                       pos=(0.0,0.0), stretch=self.RS_Q_STRETCH, textColor='white', acceptPreText='Make a selection',
                       showAccept=self.RS_Q_SHOW_ACCEPT, maxTime=self.RS_Q_TIMEOUT, name='alertness', textSize=self.RS_Q_TEXT_HEIGHT),
            1: RatingScale(win=self.ewin, scale="Were you moving any parts of your body (e.g. head, arm, leg, toes etc)?",
                       leftKeys=self.key_left, rightKeys=self.key_right, acceptKeys=[self.key_up, self.key_down],
                       markerStart='0', marker=self.RS_Q_MARKER_TYPE, markerExpansion=0,markerColor=self.RS_Q_MARKER_COLOR,
                       choices=['Not sure','No / Disagree','Yes, a little','Yes, quite a bit', 'Yes, a lot'],
                       pos=(0.0,0.0), stretch=self.RS_Q_STRETCH, textColor='white', acceptPreText='Make a selection',
                       showAccept=self.RS_Q_SHOW_ACCEPT, maxTime=self.RS_Q_TIMEOUT, name='motion', textSize=self.RS_Q_TEXT_HEIGHT),
            2: RatingScale(win=self.ewin, scale="What was the valence of your ongoing experience?",
                       leftKeys=self.key_left, rightKeys=self.key_right, acceptKeys=[self.key_up, self.key_down],
                       markerStart='0', marker=self.RS_Q_MARKER_TYPE, markerExpansion=0,markerColor=self.RS_Q_MARKER_COLOR,
                       choices=['N/A','Very negative','Somewhat negative','Neutral','Somewhat positive', 'Very positive'],
                       pos=(0.0,0.0), stretch=self.RS_Q_STRETCH, textColor='white', acceptPreText='Make a selection',
                       showAccept=self.RS_Q_SHOW_ACCEPT, maxTime=self.RS_Q_TIMEOUT, name='valence', textSize=self.RS_Q_TEXT_HEIGHT),
            3: RatingScale(win=self.ewin, scale="Were was your attention focused?",
                       leftKeys=self.key_left, rightKeys=self.key_right, acceptKeys=[self.key_up, self.key_down],
                       markerStart='0', marker=self.RS_Q_MARKER_TYPE, markerExpansion=0,markerColor=self.RS_Q_MARKER_COLOR,
                       choices=['Fully in\ninternal world','Mostly in\ninternal world','Not Sure','Mostly in\nexternal world', 'Fully in\nexternal world'],
                       pos=(0.0,0.0), stretch=self.RS_Q_STRETCH, textColor='white', acceptPreText='Make a selection',
                       showAccept=self.RS_Q_SHOW_ACCEPT, maxTime=self.RS_Q_TIMEOUT, name='focus', textSize=self.RS_Q_TEXT_HEIGHT)
            }
        
        # Internal Thoughts Screen Elements
        self.internal_text_top            = TextStim(win=self.ewin, text='Please rate the content of your INTERNAL WORLD experience regarding the following qualities:', pos=(0.0,0.6), height=0.05, wrapWidth=2)
        self.internal_text_accept_instr   = TextStim(win=self.ewin, text='(Press Down again to finish)', pos =(0.0,-0.65), height=0.05)
        self.internal_accept_button_box_y = -0.55
        self.internal_accept_button_box   = Rect(win=self.ewin, size=(0.7,0.2), pos=(0.0,self.internal_accept_button_box_y), lineWidth=2.0, fillColor='white')
        self.internal_accept_button_text  = TextStim(win=self.ewin, text='Accept Responses', pos=(0.0,self.internal_accept_button_box_y), color='Grey', height=0.05)
        self.internal_cursor_over_accept  = False
        self.internal_cursor_row          = 0
        self.internal_cursor_col          = 0
        self.internal_cursor_xys = np.array([
            [(-0.7,+0.35),(-0.45,+0.35),(-0.15,+0.35),(0.15,+0.35),(0.45,+0.35),(0.75,+0.35)],
            [(-0.7,+0.20),(-0.45,+0.20),(-0.15,+0.20),(0.15,+0.20),(0.45,+0.20),(0.75,+0.20)],
            [(-0.7,+0.05),(-0.45,+0.05),(-0.15,+0.05),(0.15,+0.05),(0.45,+0.05),(0.75,+0.05)],
            [(-0.7,-0.10),(-0.45,-0.10),(-0.15,-0.10),(0.15,-0.10),(0.45,-0.10),(0.75,-0.10)],
            [(-0.7,-0.30),(-0.45,-0.30),(-0.15,-0.30),(0.15,-0.30),(0.45,-0.30),(0.75,-0.30)],
        ])
        self.internal_selections = np.array([[1,0,0,0,0,0],
                                    [1,0,0,0,0,0],
                                    [1,0,0,0,0,0],
                                    [1,0,0,0,0,0],
                                    [1,0,0,0,0,0]])
                                    
        [self.internal_cursor_max_row, self.internal_cursor_max_col, _] = self.internal_cursor_xys.shape
        self.internal_cursor_max_row = self.internal_cursor_max_row - 1
        self.internal_cursor_max_col = self.internal_cursor_max_col - 1
        self.internal_cursor = PatchStim(win=self.ewin, units='norm', tex=None, mask='gauss', 
                                         color=self.FORMS_MARKER_COLOR, opacity=0.35, autoLog=False,
                                         pos=self.internal_cursor_xys[self.internal_cursor_row, 
                                         self.internal_cursor_col], size=0.2)
        self.internal_form_matrix = {'internal_visual': {'label_x':-0.85,
                                                         'label_y':+0.35,
                                                         'label':'Visual\nImagery',
                                                         'show_options':True , 
                                                         'options':['N/A','Strongly Disagree','Somewhat Agree','Not Sure','Somewhat Agree','Strongly Agree'], 
                                                         'option_box_xs':self.internal_cursor_xys[0,:,0], 
                                                         'option_box_ys':self.internal_cursor_xys[0,:,1],
                                                         'option_text_xs':self.internal_cursor_xys[0,:,0], 
                                                         'option_text_ys':[0.43]*6 },
                                'internal_audio':  {'label_x':-0.85,
                                                    'label_y':+0.20,
                                                    'label':'Audio\nImagery',
                                                    'show_options':False,
                                                    'options':['N/A','Strongly Disagree','Somewhat Agree','Not Sure','Somewhat Agree','Strongly Agree'],
                                                    'option_box_xs':self.internal_cursor_xys[1,:,0],
                                                    'option_box_ys':self.internal_cursor_xys[1,:,1]},
                                'internal_tactile':{'label_x':-0.85,
                                                    'label_y':+0.05,
                                                    'label':'Tactile\nImagery',
                                                    'show_options':False,
                                                    'options':['N/A','Strongly Disagree','Somewhat Agree','Not Sure','Somewhat Agree','Strongly Agree'],
                                                    'option_box_xs':self.internal_cursor_xys[2,:,0],
                                                    'option_box_ys':self.internal_cursor_xys[2,:,1]},
                                'internal_words':  {'label_x':-0.85,
                                                    'label_y':-0.10,
                                                    'label':'Inner\nSpeech',
                                                    'show_options':False,
                                                    'options':['N/A','Strongly Disagree','Somewhat Agree','Not Sure','Somewhat Agree','Strongly Agree'],
                                                    'option_box_xs':self.internal_cursor_xys[3,:,0],
                                                    'option_box_ys':self.internal_cursor_xys[3,:,1]},
                                'internal_time':   {'label_x':-0.85,
                                                    'label_y':-0.30,'label':'Time',
                                                    'show_options':True,
                                                    'options':['N/A', 'Distant past','Near past','Present', 'Near future', 'Distant future'],
                                                    'option_box_xs':self.internal_cursor_xys[4,:,0],
                                                    'option_box_ys':self.internal_cursor_xys[4,:,1],
                                                    'option_text_xs':self.internal_cursor_xys[4,:,0],
                                                    'option_text_ys':[-0.22]*6} }
                                
        self.internal_form_cartoon        = ImageStim(win=self.ewin, image=osp.join(RESOURCES_DIR,'report_period_graph.png'), pos=(-0.60,self.internal_accept_button_box_y - 0.05), size=(.5,.3))
        self.internal_form_rbox           = ImageStim(win=self.ewin, image=osp.join(RESOURCES_DIR,'resp_box_form.png'), pos=(+0.60, self.internal_accept_button_box_y - 0.03), size=(.2,.3))

        # External Thoughts Screen Elements
        self.external_accept_button_box_y = -0.40
        self.external_text_top            = TextStim(win=self.ewin, text='Please rate the focus on you attention on different elements of the EXTERNAL environment:', pos=(0.0,0.5), height=0.05, wrapWidth=2)
        self.external_text_accept_instr   = TextStim(win=self.ewin, text='(Press Down again to finish)', pos =(0.0,-0.50), height=0.05)
        self.external_accept_button_box   = Rect(win=self.ewin, size=(0.7,0.2), pos=(0.0,self.external_accept_button_box_y), lineWidth=2.0, fillColor='white')
        self.external_accept_button_text  = TextStim(win=self.ewin, text='Accept Responses', pos=(0.0,self.external_accept_button_box_y), color='Grey', height=0.05)
        self.external_cursor_over_accept  = False
        self.external_cursor_row          = 0
        self.external_cursor_col          = 0
        self.external_cursor_xys = np.array([
            [(-0.7,+0.20),(-0.45,+0.20),(-0.15,+0.20),(0.15,+0.20),(0.45,+0.20),(0.75,+0.20)],
            [(-0.7,+0.05),(-0.45,+0.05),(-0.15,+0.05),(0.15,+0.05),(0.45,+0.05),(0.75,+0.05)],
            [(-0.7,-0.10),(-0.45,-0.10),(-0.15,-0.10),(0.15,-0.10),(0.45,-0.10),(0.75,-0.10)],
        ])
        self.external_selections = np.array([[1,0,0,0,0,0],
                                    [1,0,0,0,0,0],
                                    [1,0,0,0,0,0],
                                    [1,0,0,0,0,0],
                                    [1,0,0,0,0,0]])
                                    
        [self.external_cursor_max_row, self.external_cursor_max_col, _] = self.external_cursor_xys.shape
        self.external_cursor_max_row = self.external_cursor_max_row - 1
        self.external_cursor_max_col = self.external_cursor_max_col - 1
        self.external_cursor = PatchStim(win=self.ewin, units='norm', tex=None, mask='gauss', 
                                         color=self.FORMS_MARKER_COLOR, opacity=0.35, autoLog=False,
                                         pos=self.external_cursor_xys[self.external_cursor_row, 
                                         self.external_cursor_col], size=0.2)
        self.external_form_matrix = {'external_visual': {'label_x':-0.85,
                                                         'label_y':+0.20,
                                                         'label':'Visual\nElements',
                                                         'show_options':True , 
                                                         'options':['N/A','Strongly Disagree','Somewhat Agree','Not Sure','Somewhat Agree','Strongly Agree'], 
                                                         'option_box_xs':self.external_cursor_xys[0,:,0], 
                                                         'option_box_ys':self.external_cursor_xys[0,:,1],
                                                         'option_text_xs':self.external_cursor_xys[0,:,0], 
                                                         'option_text_ys':[0.28]*6 },
                                'external_audio':  {'label_x':-0.85,
                                                    'label_y':+0.05,
                                                    'label':'Audio\nElements',
                                                    'show_options':False,
                                                    'options':['N/A','Strongly Disagree','Somewhat Agree','Not Sure','Somewhat Agree','Strongly Agree'],
                                                    'option_box_xs':self.external_cursor_xys[1,:,0],
                                                    'option_box_ys':self.external_cursor_xys[1,:,1]},
                                'external_tactile':{'label_x':-0.85,
                                                    'label_y':-0.10,
                                                    'label':'Tactile\nElements',
                                                    'show_options':False,
                                                    'options':['N/A','Strongly Disagree','Somewhat Agree','Not Sure','Somewhat Agree','Strongly Agree'],
                                                    'option_box_xs':self.external_cursor_xys[2,:,0],
                                                    'option_box_ys':self.external_cursor_xys[2,:,1]},
                                }
        self.external_form_cartoon        = ImageStim(win=self.ewin, image=osp.join(RESOURCES_DIR,'report_period_graph.png'), pos=(-0.60,self.external_accept_button_box_y - 0.05), size=(.5,.3))
        self.external_form_rbox           = ImageStim(win=self.ewin, image=osp.join(RESOURCES_DIR,'resp_box_form.png'), pos=(+0.60, self.external_accept_button_box_y - 0.03), size=(.2,.3))

        # Second Set of Generic Questions
        self.likert_qs_genset_02 = {
            0: RatingScale(win=self.ewin, scale="Were you in control of your thoughts/experience or did it emerge spontaneously?",
                       leftKeys=self.key_left, rightKeys=self.key_right, acceptKeys=[self.key_up, self.key_down],
                       markerStart='0', marker=self.RS_Q_MARKER_TYPE, markerExpansion=0,markerColor=self.RS_Q_MARKER_COLOR,
                       choices=['Fully in control','Somewhat in control','Not Sure','Somewhat spontaneously','Fully spontaneously'],
                       pos=(0.0,0.0), stretch=self.RS_Q_STRETCH, textColor='white', acceptPreText='Make a selection',
                       showAccept=self.RS_Q_SHOW_ACCEPT, maxTime=self.RS_Q_TIMEOUT, name='intention', textSize=self.RS_Q_TEXT_HEIGHT),
            1: RatingScale(win=self.ewin, scale="To what extent was your ongoing experience effortful?",
                       leftKeys=self.key_left, rightKeys=self.key_right, acceptKeys=[self.key_up, self.key_down],
                       markerStart='0', marker=self.RS_Q_MARKER_TYPE, markerExpansion=0,markerColor=self.RS_Q_MARKER_COLOR,
                       choices=['Not at all','Somewhat effortful','Very effortful','Extremely effortful'],
                       pos=(0.0,0.0), stretch=self.RS_Q_STRETCH, textColor='white', acceptPreText='Make a selection',
                       showAccept=self.RS_Q_SHOW_ACCEPT, maxTime=self.RS_Q_TIMEOUT, name='effort', textSize=self.RS_Q_TEXT_HEIGHT),
            2: RatingScale(win=self.ewin, scale="Did anything salient/surprising occur in your ongoing experience?",
                       leftKeys=self.key_left, rightKeys=self.key_right, acceptKeys=[self.key_up, self.key_down],
                       markerStart='0', marker=self.RS_Q_MARKER_TYPE, markerExpansion=0,markerColor=self.RS_Q_MARKER_COLOR,
                       choices=['Not at all','Somewhat salient','Very salient','Extremely salient'],
                       pos=(0.0,0.0), stretch=self.RS_Q_STRETCH, textColor='white', acceptPreText='Make a selection',
                       showAccept=self.RS_Q_SHOW_ACCEPT, maxTime=self.RS_Q_TIMEOUT, name='salience', textSize=self.RS_Q_TEXT_HEIGHT),
            3: RatingScale(win=self.ewin, scale="How aware were your of your ongoing experience?",
                       leftKeys=self.key_left, rightKeys=self.key_right, acceptKeys=[self.key_up, self.key_down],
                       markerStart='0', marker=self.RS_Q_MARKER_TYPE, markerExpansion=0,markerColor=self.RS_Q_MARKER_COLOR,
                       choices=['Not aware at all','Somewhat aware','Extremely aware'],
                       pos=(0.0,0.0), stretch=self.RS_Q_STRETCH, textColor='white', acceptPreText='Make a selection',
                       showAccept=self.RS_Q_SHOW_ACCEPT, maxTime=self.RS_Q_TIMEOUT, name='awareness', textSize=self.RS_Q_TEXT_HEIGHT),
            4: RatingScale(win=self.ewin, scale="How long was the ongoing experience you are describing?",
                       leftKeys=self.key_left, rightKeys=self.key_right, acceptKeys=[self.key_up, self.key_down],
                       markerStart='0', marker=self.RS_Q_MARKER_TYPE, markerExpansion=0,markerColor=self.RS_Q_MARKER_COLOR,
                       choices=['Very brief (~1s)','Somwhat brief (2-5s)','Somewhat lengthy (5-10s)','Very lengthy (>10s)'],
                       pos=(0.0,0.0), stretch=self.RS_Q_STRETCH, textColor='white', acceptPreText='Make a selection',
                       showAccept=self.RS_Q_SHOW_ACCEPT, maxTime=self.RS_Q_TIMEOUT, name='exp_length', textSize=self.RS_Q_TEXT_HEIGHT)
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
        self.default_inst_04.draw()
        self.default_chair.draw()
        self.ewin.flip()
        return None
    
    def draw_alert_screen(self):
        self.beep_inst_top_01.draw()
        self.beep_inst_top_02.draw()
        self.beep_inst_top_03.draw()
        self.beep_inst_top_04.draw()
        self.beep_inst_bot_01.draw()
        self.beep_inst_bot_02.draw()
        self.beep_chair.draw()
        self.beep_inst_cartoon.draw()
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
                    self.beep_inst_top_04.draw()
                    self.beep_chair = TextStim(win=self.ewin, text='[ RECORDING ]', color='red', pos=(0.0,0.0), bold=True, opacity=op)
                    self.beep_chair.draw()
                    self.beep_inst_bot_01.draw()
                    self.beep_inst_bot_02.draw()
                    self.beep_inst_report_period_cartoon.draw()
                    #self.mic_image.draw()
                    self.ewin.flip()
            rec_file.stop_recording()
        return None

    def draw_ack_recording_screen(self):
        self.mic_ack_rec_01.draw()
        self.mic_ack_rec_02.draw()
        self.q_gen_instr_01.draw()
        self.q_gen_instr_02.draw()
        self.q_gen_instr_03.draw()
        self.q_gen_instr_04.draw()
        self.q_gen_instr_05.draw()
        self.q_gen_cartoon.draw()
        self.q_gen_rbox.draw()
        self.ewin.flip()
        while not event.getKeys():
            sleep(0.01)
        return None
    
    def show_rs_q_genset01(self):
        responses = {}
        order     = np.arange(len(self.likert_qs_genset_01))
        for q_idx in order:
            aux_q = self.likert_qs_genset_01[q_idx]
            aux_q.reset()
            event.clearEvents()
            while aux_q.noResponse:
                self.likert_qs_cartoon.draw()
                self.likert_qs_rbox.draw()
                aux_q.draw()
                self.ewin.flip()
                if event.getKeys(['escape']):
                    core.quit()
            responses[aux_q.name] = [aux_q.getRating(), aux_q.getRT(), aux_q.getHistory()]
        return responses
    
    def show_rs_q_genset02(self):
        responses = {}
        order     = np.arange(len(self.likert_qs_genset_02))
        for q_idx in order:
            aux_q = self.likert_qs_genset_02[q_idx]
            aux_q.reset()
            event.clearEvents()
            if ((aux_q.name=='effort') and (responses['intention'][0] in ['Fully in control','Somewhat in control'])) or (aux_q.name!='effort'):
                while aux_q.noResponse:
                    self.likert_qs_cartoon.draw()
                    self.likert_qs_rbox.draw()
                    aux_q.draw()
                    self.ewin.flip()
                    if event.getKeys(['escape']):
                        core.quit()
                responses[aux_q.name] = [aux_q.getRating(), aux_q.getRT(), aux_q.getHistory()]
        return responses

    def update_internal_cursor(self,key):
        if key == self.key_up:
            if self.internal_cursor_over_accept == True:
                self.internal_cursor_row= self.internal_cursor_max_row
                self.internal_cursor.setPos(self.internal_cursor_xys[self.internal_cursor_row, self.internal_cursor_col])
                self.internal_cursor_over_accept = False
                return
            if self.internal_cursor_row> 0:
                self.internal_cursor_row= self.internal_cursor_row- 1
                self.internal_cursor.setPos(self.internal_cursor_xys[self.internal_cursor_row, self.internal_cursor_col])
            return
        if key == self.key_down:
            if self.internal_cursor_row== self.internal_cursor_max_row:
                self.internal_selections[self.internal_cursor_row,:] = np.zeros(self.internal_selections.shape[1])
                self.internal_selections[self.internal_cursor_row, self.internal_cursor_col ] = 1
                self.internal_cursor.setPos((0.0,self.internal_accept_button_box_y))
                self.internal_cursor_over_accept = True
                return
            if self.internal_cursor_row< self.internal_cursor_max_row:
                self.internal_selections[self.internal_cursor_row,:] = np.zeros(self.internal_selections.shape[1])
                self.internal_selections[self.internal_cursor_row, self.internal_cursor_col ] = 1
                self.internal_cursor_row= self.internal_cursor_row+ 1
                self.internal_cursor.setPos(self.internal_cursor_xys[self.internal_cursor_row, self.internal_cursor_col])
                
            return
        if key == self.key_left:
            self.internal_cursor_over_accept = False
            if self.internal_cursor_col > 0:
                self.internal_cursor_col = self.internal_cursor_col - 1
                self.internal_cursor.setPos(self.internal_cursor_xys[self.internal_cursor_row, self.internal_cursor_col])
            return
        if key == self.key_right:
            self.internal_cursor_over_accept = False
            if self.internal_cursor_col < self.internal_cursor_max_col:
                self.internal_cursor_col = self.internal_cursor_col + 1
                self.internal_cursor.setPos(self.internal_cursor_xys[self.internal_cursor_row, self.internal_cursor_col])
            return
    
    def update_external_cursor(self,key):
        if key == self.key_up:
            if self.external_cursor_over_accept == True:
                self.external_cursor_row= self.external_cursor_max_row
                self.external_cursor.setPos(self.external_cursor_xys[self.external_cursor_row, self.external_cursor_col])
                self.external_cursor_over_accept = False
                return
            if self.external_cursor_row> 0:
                self.external_cursor_row= self.external_cursor_row- 1
                self.external_cursor.setPos(self.external_cursor_xys[self.external_cursor_row, self.external_cursor_col])
            return
        if key == self.key_down:
            if self.external_cursor_row== self.external_cursor_max_row:
                self.external_selections[self.external_cursor_row,:] = np.zeros(self.external_selections.shape[1])
                self.external_selections[self.external_cursor_row, self.external_cursor_col ] = 1
                self.external_cursor.setPos((0.0,self.external_accept_button_box_y))
                self.external_cursor_over_accept = True
                return
            if self.external_cursor_row< self.external_cursor_max_row:
                self.external_selections[self.external_cursor_row,:] = np.zeros(self.external_selections.shape[1])
                self.external_selections[self.external_cursor_row, self.external_cursor_col ] = 1
                self.external_cursor_row= self.external_cursor_row+ 1
                self.external_cursor.setPos(self.external_cursor_xys[self.external_cursor_row, self.external_cursor_col])
                
            return
        if key == self.key_left:
            self.external_cursor_over_accept = False
            if self.external_cursor_col > 0:
                self.external_cursor_col = self.external_cursor_col - 1
                self.external_cursor.setPos(self.external_cursor_xys[self.external_cursor_row, self.external_cursor_col])
            return
        if key == self.key_right:
            self.external_cursor_over_accept = False
            if self.external_cursor_col < self.external_cursor_max_col:
                self.external_cursor_col = self.external_cursor_col + 1
                self.external_cursor.setPos(self.external_cursor_xys[self.external_cursor_row, self.external_cursor_col])
            return

    def show_internal_world_questions(self):
        responses_ready = False
        while not responses_ready:
            # Draw Instructions on top
            self.internal_text_top.draw()
            for r,(item_id,item) in enumerate(self.internal_form_matrix.items()):
                label = TextStim(win=self.ewin, text=item['label'], pos=(item['label_x'],item['label_y']), height=0.05)
                label.draw()
                for c in range(len(item['options'])):
                    if self.internal_selections[r,c] == 1:
                        box_fill = 'white'
                    else:
                        box_fill = None
                    box = Rect(win=self.ewin, size=(0.05,0.1), pos=(item['option_box_xs'][c], item['option_box_ys'][c]), lineWidth=2.0, fillColor=box_fill)
                    box.draw()
                if item['show_options']:
                    for c in range(len(item['options'])):
                       text = TextStim(win=self.ewin, text=item['options'][c], pos=(item['option_text_xs'][c], item['option_text_ys'][c]), height=0.05)
                       text.draw()
            self.internal_accept_button_box.draw()
            self.internal_accept_button_text.draw()
            self.internal_form_cartoon.draw()
            self.internal_form_rbox.draw()
            if self.internal_cursor_over_accept:
                self.internal_text_accept_instr.draw()
            self.internal_cursor.draw()
            self.ewin.flip()
            pressed_keys = event.getKeys(['escape', self.key_up, self.key_down, self.key_left, self.key_right])
            if 'escape' in pressed_keys:
                core.quit()
            else:
                for key in pressed_keys:
                   if (key == self.key_down) and (self.internal_cursor_over_accept) == True:
                        responses_ready = True
                   else:
                    self.update_internal_cursor(key)
        
        # Create Responses Dictionary Object to return
        responses = {}
        for q_i, question in enumerate(self.internal_form_matrix.keys()):
            
            s    = np.asscalar(np.squeeze(np.argwhere(self.internal_selections[q_i,:]>0)))
            opts = self.internal_form_matrix[question]['options']
            responses[question]=[opts[s], 0.0, [(opts[s],0.0)]]
        return responses

    def show_external_world_questions(self):
        responses_ready = False
        while not responses_ready:
            # Draw Instructions on top
            self.external_text_top.draw()
            for r,(item_id,item) in enumerate(self.external_form_matrix.items()):
                label = TextStim(win=self.ewin, text=item['label'], pos=(item['label_x'],item['label_y']), height=0.05)
                label.draw()
                for c in range(len(item['options'])):
                    if self.external_selections[r,c] == 1:
                        box_fill = 'white'
                    else:
                        box_fill = None
                    box = Rect(win=self.ewin, size=(0.05,0.1), pos=(item['option_box_xs'][c], item['option_box_ys'][c]), lineWidth=2.0, fillColor=box_fill)
                    box.draw()
                if item['show_options']:
                    for c in range(len(item['options'])):
                       text = TextStim(win=self.ewin, text=item['options'][c], pos=(item['option_text_xs'][c], item['option_text_ys'][c]), height=0.05)
                       text.draw()
            self.external_accept_button_box.draw()
            self.external_accept_button_text.draw()
            self.external_form_cartoon.draw()
            self.external_form_rbox.draw()
            if self.external_cursor_over_accept:
                self.external_text_accept_instr.draw()
            self.external_cursor.draw()
            self.ewin.flip()
            pressed_keys = event.getKeys(['escape', self.key_up, self.key_down, self.key_left, self.key_right])
            if 'escape' in pressed_keys:
                core.quit()
            else:
                for key in pressed_keys:
                   if (key == self.key_down) and (self.external_cursor_over_accept) == True:
                        responses_ready = True
                   else:
                    self.update_external_cursor(key)
        
        # Create Responses Dictionary Object to return
        responses = {}
        for q_i, question in enumerate(self.external_form_matrix.keys()):
            
            s    = np.asscalar(np.squeeze(np.argwhere(self.external_selections[q_i,:]>0)))
            opts = self.external_form_matrix[question]['options']
            responses[question]=[opts[s], 0.0, [(opts[s],0.0)]]
        return responses

    def close_psychopy_infrastructure(self):
        log.info(' - close_psychopy_infrastructure - Function called.')
        self.ewin.flip()
        self.ewin.close()
        psychopy_logging.flush()
        core.quit()
        return None

    def run_full_QA(self):
        # 1) Play beep and instruct subject to talk
        self.draw_alert_screen()
        
        # 2) Record oral description
        event.clearEvents()
        self.record_oral_descr()
        
        # 3) Acknowledge successful recording
        self.draw_ack_recording_screen()
        
        # 4) Do First set of likert questions
        responses = self.show_rs_q_genset01()
        
        # 5) Run Internal or External Form (as appropriate)
        att_focus_selection = responses['focus'][0]
        if att_focus_selection in ['Fully in\ninternal world','Mostly in\ninternal world']:
            responses.update(self.show_internal_world_questions())
        if att_focus_selection in ['Fully in\nexternal world','Mostly in\nexternal world']:
            responses.update(self.show_external_world_questions())

        # 6) Do Second set of likert questions
        responses.update(self.show_rs_q_genset02())
        
        # 7) Write responses to disk
        resp_timestr = time.strftime("%Y%m%d-%H%M%S")
        resp_path    = osp.join(self.out_dir,self.out_prefix+'.'+resp_timestr+'.LikertResponses'+str(self.hitID).zfill(3)+'.txt')
        w            = csv.writer(open(resp_path, "w"))
        for key, val in responses.items():
            w.writerow([key, val])
        self.hitID   = self.hitID + 1
        return responses
