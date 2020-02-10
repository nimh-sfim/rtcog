import os.path as osp
from bokeh.palettes import Category10_7

PRJDIR      = '/data/SFIMJGC/PRJ_rtCAPs'
CAPS_DIR    = osp.join(PRJDIR,'Others')

CAP_indexes = [25,    4,    18,    28,    24,    11,    21]
CAP_labels  = ['VPol','DMN','SMot','Audi','ExCn','rFPa','lFPa']
CAPLabel2Int = {'VPol':25,'DMN':4,'Smot':18,'Audi':28,'ExCn':24,'rFPa':11,'lFPa':21}
CAP_colors   = {x:y for x,y in zip(CAP_labels,Category10_7)}

Question_Dict={'rs_alert'        : 'How alert were you?',
               'rs_motion'       : 'Were you moving any parts of your body (e.g. head, arm, leg, toes etc)?',
               'rs_visual'       : 'Was your attention focused on visual elements of the environment?',
               'rs_audio'        : 'Was your attention focused on auditory elements of the environment?',
               'rs_tactile'      : 'Was your attention focused on tactile elements of the environment?',
               'rs_internal'     : 'Was your attention focused on your internal world?',
               'rs_time'         : 'Where in time was your attention focused?',
               'rs_modality'     : 'What was the modality / sensory domain of your ongoing experience?',
               'rs_valence'      : 'What was the valence of your ongoing experience?',
               'rs_attention'    : 'Was your attention focused intentionally or unintentionally?',
               'rs_attention_B'  : 'Was your attention focused with or without awareness?',
               }

Question_ToNum={'rs_alert'       : {'Fully asleep':1/4,'Somewhat sleepy':2/4,'Somewhat alert':3/4,'Fully alert':4/4},
                'rs_motion'      : {'Not sure':1/5,'No / Disagree':2/5,'Yes, a little':3/5,'Yes, quite a bit':4/5, 'Yes, a lot':5/5},
                'rs_visual'      : {'Strongly disagree':1/5,'Somewhat disagree':2/5,'Not sure':0,'Somewhat agree':4/5, 'Strongly agree':5/5},
                'rs_audio'       : {'Strongly disagree':1/5,'Somewhat disagree':2/5,'Not sure':0,'Somewhat agree':4/5, 'Strongly agree':5/5},
                'rs_tactile'     : {'Strongly disagree':1/5,'Somewhat disagree':2/5,'Not sure':0,'Somewhat agree':4/5, 'Strongly agree':5/5},
                'rs_internal'    : {'Strongly disagree':1/5,'Somewhat disagree':2/5,'Not sure':0,'Somewhat agree':4/5, 'Strongly agree':5/5},
                'rs_time'        : {'No time\\nin particular':1/6,'Distant past\\n(>1 day)':2/6,'Near past\\n(last 24h)':3/6,'Present':4/6, 'Near future':5/6, 'Distant future':6/6},
                'rs_modality'    : {'Exclusively\\nin words':1/5,'Mostly words\\n& some imagery':2/5,'Balance of\\nwords & imagery':3/5,'Mostly imagery\\n& some words':4/5, 'Exclusively\\nin imagery':5/5},
                'rs_valence'     : {'Very negative':1/5,'Somewhat negative':2/5,'Neutral':3/5,'Somewhat positive':4/5, 'Very positive':5/5},
                'rs_attention'   : {'Intentionally':0.1,'Unintentionally':1},
                'rs_attention_B' : {'Not aware at all':1/3,'Somewhat aware':2/3,'Extremely aware':3/3}
               }
