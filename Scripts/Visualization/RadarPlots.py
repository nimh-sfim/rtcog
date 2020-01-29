# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.0
#   kernelspec:
#     display_name: psychopy_pyo_nilearn 0.6
#     language: python
#     name: psychopy_pyo_nilearn
# ---

import glob
import numpy as np
import ntpath
import re
import pandas as pd
import hvplot.pandas
import holoviews as hv
import os.path as osp
from bokeh.plotting import figure, show, output_notebook
from bokeh.models import ColumnDataSource, LabelSet, HoverTool
import json
import panel as pn
pn.extension()


# +
def get_answer(file):
    out = {}
    aux   = pd.read_csv(file,header=None, names=['Question','Responses'], sep=',').set_index('Question')
    for q in aux.index:
        aux_q_list          = aux.loc[q].tolist()[0]
        [aux_q_p1,aux_q_p2] = aux_q_list.lstrip('[').rstrip(']').split('[')
        aux_r_str = re.findall(r"'([^']*)'", aux_q_p1)[0] 
        aux_rt    = float(aux_q_p1.replace("\'"+aux_r_str+"\'",'').split(',')[1])
        out[q] = (aux_r_str,aux_rt)
    return out

def get_runs_per_subject(s):
    # Obtain All Response Files for this Subject 
    wdir   = osp.join(PRJDIR,'PrcsData',s,'D01_OriginalData/')
    this_sbj_files = glob.glob(wdir+'/'+s+'_*.Responses.???.txt')
    # Obtain the list of Runs for this particular subjecgt
    aux_runs = []
    for file in this_sbj_files:
        [s,r]=ntpath.basename(file).split('.')[0].split('_')
        aux_runs.append(r)
    avail_runs = list(set(aux_runs))
    avail_runs.sort()
    return avail_runs

def read_transcript(path):
    with open(path, 'r') as file:
        data = file.read().replace('\n', '')
    return data


# -

SBJs       = ['PILOT03','PILOT04','PILOT05']
CAP_Labels = ['VPol','DMN','SMot','Audi','ExCn','rFPa','lFPa']
PRJDIR     = '/data/SFIMJGC/PRJ_rtCAPs/'

# +
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
# -

# Get list of all data files available per subject
Runs_Per_Subject = {}
Response_Files   = {}
Transcript_Files = {}
Hit_Files        = {}
SVRscores_Files  = {}
Options_Files    = {}
QAOnset_Files    = {}
QAOffset_Files   = {}
for s in SBJs:
    Runs_Per_Subject[s] = get_runs_per_subject(s)
    for r in Runs_Per_Subject[s]:
        Hit_Files[(s,r)]        = osp.join(PRJDIR,'PrcsData',s,'D01_OriginalData',s+'_'+r+'.hits.npy')
        aux_response_files      = glob.glob(str(osp.join(PRJDIR,'PrcsData',s,'D01_OriginalData'))+'/'+s+'_'+r+'.Responses.???.txt')
        aux_response_files.sort()
        Response_Files[(s,r)]   = aux_response_files
        aux_transcript_files    = glob.glob(str(osp.join(PRJDIR,'PrcsData',s,'D01_OriginalData'))+'/'+s+'_'+r+'.Responses_Oral_Transcript.???.txt')
        aux_transcript_files.sort()
        Transcript_Files[(s,r)] = aux_transcript_files
        SVRscores_Files[(s,r)]  = osp.join(PRJDIR,'PrcsData',s,'D01_OriginalData',s+'_'+r+'.svrscores.npy')
        Options_Files[(s,r)]    = osp.join(PRJDIR,'PrcsData',s,'D01_OriginalData',s+'_'+r+'_Options.json')
        QAOnset_Files[(s,r)]    = osp.join(PRJDIR,'PrcsData',s,'D01_OriginalData',s+'_'+r+'.qa_onsets.txt')
        QAOffset_Files[(s,r)]   = osp.join(PRJDIR,'PrcsData',s,'D01_OriginalData',s+'_'+r+'.qa_offsets.txt')
        

DF_Behav = pd.DataFrame(columns=['Subject','Run','Hit_ID','CAP','TR','Question','Resp_Str','Resp_Int','Resp_Time','Text'])
for s in SBJs:
    for r in Runs_Per_Subject[s]:
        hit_info   = pd.DataFrame(np.load(Hit_Files[(s,r)]).T, columns=CAP_Labels)
        hits       = hit_info[hit_info.T.sum()>0]
        for i,file in enumerate(Response_Files[(s,r)]):
            # Get Transcription of Oral Response if already available
            try:
                trans_file = Transcript_Files[(s,r)][i]
                transcript = read_transcript(trans_file)
            except:
                transcript = ''
            # Get responses to likert scale question
            resp = get_answer(file)
            # For each likert scale question insert an entry into the DF_Behavior dataframe
            for q in resp.keys():
                DF_Behav = DF_Behav.append({'Subject':s,
                                        'Run'      : r,
                                        'Hit_ID'   : i+1,
                                        'TR'       : int(hits.iloc[i].name),
                                        'CAP'      : hits.iloc[i][hits.iloc[i]==1.0].index[0],
                                        'Question' : q,
                                        'Resp_Str' : resp[q][0],
                                        'Resp_Int' : Question_ToNum[q][resp[q][0]],
                                        'Resp_Time': float(resp[q][1]),
                                        'Text'     : transcript},ignore_index=True)


# ***
# # Graphical Stuff - Radar Plots

# +
def unit_poly_verts(theta, centre ):
    """Return vertices of polygon for subplot axes.
    This polygon is circumscribed by a unit circle centered at (0.5, 0.5)
    """
    x0, y0, r = [centre] * 3
    verts = [(r*np.cos(t) + x0, r*np.sin(t) + y0) for t in theta]
    return verts

def radar_patch(r, theta, centre ):
    """ Returns the x and y coordinates corresponding to the magnitudes of 
    each variable displayed in the radar plot
    """
    # offset from centre of circle
    offset = 0.0
    yt = (r*centre + offset) * np.sin(theta) + centre 
    xt = (r*centre + offset) * np.cos(theta) + centre 
    return xt, yt

def generate_radar_chart_from_resp(resp):
    num_vars = len(resp)
    centre   = 0.5
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
    theta += np.pi/2
    verts = unit_poly_verts(theta, centre)
    x     = [v[0] for v in verts] 
    y     = [v[1] for v in verts] 
    p     = figure(title="Subject Response")
    text  = ['Q'+str(i).zfill(2) for i in range(1,num_vars+1)]+[''] #list(resp.keys())+['']
    source = ColumnDataSource({'x':x + [centre],'y':y + [1],'text':text})
    #p.line(x="x", y="y", source=source)
    p.circle(x=0.5,y=0.5,radius=.5, line_color='black', fill_color=None)
    for i,j in zip(x,y):
        p.line(x=[centre,i],y=[centre,j], line_color='black', line_dash='dashed')
    labels = LabelSet(x="x",y="y",text="text",source=source)
    p.add_layout(labels)
    f = []
    for q in resp.keys():
        f.append(Question_ToNum[q][resp[q][0]])
    f = np.array(f)
    xt, yt = radar_patch(f, theta, centre)
    p.patch(x=xt, y=yt, fill_alpha=0.15, fill_color='red', line_color='black', line_width=2)
    return p

def generate_radar_chart_from_vals(vals, strs):
    centre   = 0.5
    num_vars = len(vals)
    theta    = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
    theta   += np.pi/2
    verts    = unit_poly_verts(theta, centre)
    x        = [v[0] for v in verts] 
    y        = [v[1] for v in verts]

    p =figure(match_aspect=True)
    # Draw Outter Dots
    out_dots_TOOLTIPS = [("Q:", "@desc")]
    out_dots_src = ColumnDataSource({'x':x,'y':y,'desc':list(Question_Dict.values())})
    g_out_dots = p.circle(x='x',y='y', color='black', source=out_dots_src)
    out_dots_hover = HoverTool(renderers=[g_out_dots], tooltips=out_dots_TOOLTIPS)
    p.add_tools(out_dots_hover)

    # Draw Outter Circle
    p.circle(x=0.5,y=0.5,radius=0.5, fill_color=None, line_color='black', line_alpha=0.5)

    # Draw concentrical lines
    for i,j in zip(x,y):
        p.line(x=[centre,i],y=[centre,j], line_color='black', line_dash='dashed', line_alpha=0.5)

    #Draw intermediate circles
    p.circle(x=0.5,y=0.5,radius=.5, line_color='black', fill_color=None, line_alpha=0.5)
    p.circle(x=0.5,y=0.5,radius=.1, line_color='black', fill_color=None, line_alpha=0.5, line_dash='dashed')
    p.circle(x=0.5,y=0.5,radius=.2, line_color='black', fill_color=None, line_alpha=0.5, line_dash='dashed')
    p.circle(x=0.5,y=0.5,radius=.3, line_color='black', fill_color=None, line_alpha=0.5, line_dash='dashed')
    p.circle(x=0.5,y=0.5,radius=.4, line_color='black', fill_color=None, line_alpha=0.5, line_dash='dashed')

    # Visual Aspects
    p.xgrid.visible=False
    p.ygrid.visible=False
    p.xaxis.visible=False
    p.yaxis.visible=False
    p.toolbar.logo=None
    p.toolbar_location='below'

    # Draw Question IDs
    labels_txt = ['Q'+str(i).zfill(2) for i in range(1,num_vars+1)]
    labels_src = ColumnDataSource({'x':[i if i >= 0.5 else i-.05 for i in x],'y':[i if i >= 0.5 else i-.05 for i in y],'text':labels_txt})
    labels     = LabelSet(x="x",y="y",text="text",source=labels_src)
    p.add_layout(labels)
    xt, yt = radar_patch(np.array(vals), theta, centre)
    p.patch(x=xt, y=yt, fill_alpha=0.15, fill_color='red', line_color='black', line_width=2)

    # Patch hovering
    patch_dots_TOOLTIPS = [("Response:","@desc")]
    patch_dots_src = ColumnDataSource({'xt':xt,'yt':yt,'desc':strs})
    patch_dots = p.circle(x='xt',y='yt',color='black', source=patch_dots_src)
    patch_dots_hover = HoverTool(renderers=[patch_dots], tooltips=patch_dots_TOOLTIPS)
    p.add_tools(patch_dots_hover)
    p.width=425
    p.height=425
    return p



# -

# ***
# # Graphical Stuff - Dashboard

# +
Title            = pn.panel('# Individual Hit Summary View')
SBJ_Select       = pn.widgets.Select(name='Subject ID:', value=SBJs[0], options=SBJs)
initial_run_list = get_runs_per_subject(SBJs[0])
RUN_Select       = pn.widgets.Select(name='Run ID:',     value=initial_run_list[0], options=initial_run_list)
initial_cap_list = list(DF_Behav[(DF_Behav['Subject']==SBJs[0]) & (DF_Behav['Run']==initial_run_list[0])]['CAP'].unique())
CAP_TYPE_Select  = pn.widgets.Select(name='CAP Type:', value = initial_cap_list[0], options=initial_cap_list)
@pn.depends(SBJ_Select.param.value, watch=True)
def _update_run_select_list(sbj):
    run_list = get_runs_per_subject(sbj)
    RUN_Select.options = run_list
    RUN_Select.value   = run_list[0]
    
@pn.depends(SBJ_Select.param.value, RUN_Select.param.value)
def hit_count_table(sbj,run):
    hits_file = osp.join(PRJDIR,'PrcsData',sbj,'D01_OriginalData',sbj+'_'+run+'.hits.npy')
    hits = pd.DataFrame(np.load(hits_file).T,columns=CAP_Labels)
    hits_count = hits.sum().to_frame().T
    hits_count.index=['Number of Hits']
    return hits_count

@pn.depends(SBJ_Select.param.value, RUN_Select.param.value, watch=True)
def get_avail_hit_types_select(sbj,run):
    options = list(DF_Behav[(DF_Behav['Subject']==sbj) & (DF_Behav['Run']==run)]['CAP'].unique())
    CAP_TYPE_Select.options = options
    CAP_TYPE_Select.value   = options[0]

@pn.depends(SBJ_Select.param.value, RUN_Select.param.value, CAP_TYPE_Select.param.value, watch=True)
def get_svrscore_plot(sbj,run,cap):
    svr_file            = SVRscores_Files[(sbj,run)]
    options_file        = Options_Files[(sbj,run)]
    hits_file           = Hit_Files[(sbj,run)]
    qaonset_file        = QAOnset_Files[(sbj,run)]
    qaoffset_file       = QAOffset_Files[(sbj,run)]
    with open(options_file) as file:
        data = json.load(file)
    with open(qaonset_file,'r') as file:
        qa_onsets = file.read().split('\n')
        qa_onsets = [int(i) for i in qa_onsets[:-1]]
    with open(qaoffset_file,'r') as file:
        qa_offsets = file.read().split('\n')
        qa_offsets = [int(i) for i in qa_offsets[:-1]]
    ZTH       = data['hit_zth']
    VOLS_NOQA = data['vols_noqa']
    svrscores           = np.load(svr_file).T
    hits                = np.load(hits_file).T
    
    SVR_Scores_DF       = pd.DataFrame(svrscores, columns=CAP_Labels)
    SVR_Scores_DF['TR'] = SVR_Scores_DF.index
    SVRscores_curve     = SVR_Scores_DF.hvplot(legend='top', label='SVR Scores', x='TR').opts(width=2400, toolbar='below')
    Threshold_line      = hv.HLine(ZTH).opts(color='black', line_dash='dashed', line_width=1)
    Hits_ToPlot         = hits * svrscores
    Hits_ToPlot[Hits_ToPlot==0.0] = None
    Hits_DF             = pd.DataFrame(Hits_ToPlot, columns=CAP_Labels)
    Hits_DF['TR']       = Hits_DF.index
    Hits_Marks          = Hits_DF.hvplot(legend='top', label='SVR Scores', 
                                             x='TR', kind='scatter', marker='circle', 
                                             alpha=0.5, s=100).opts(width=1500)
    qa_boxes = []
    for (on,off) in zip(qa_onsets,qa_offsets):
        qa_boxes.append(hv.Box(x=on+((off-on)/2),y=0,spec=(off-on,10)))
    QA_periods = hv.Polygons(qa_boxes).opts(alpha=.2, color='blue', line_color=None)
    
    wait_boxes = []
    for off in qa_offsets:
        wait_boxes.append(hv.Box(x=off+(VOLS_NOQA/2),y=0,spec=(VOLS_NOQA,10)))
    WAIT_periods = hv.Polygons(wait_boxes).opts(alpha=.2, color='cyan', line_color=None)
            
    plot_layout         = (SVRscores_curve * Threshold_line * Hits_Marks * QA_periods * WAIT_periods)
    
    hit_TRs = list(DF_Behav[(DF_Behav['Subject']==sbj) & (DF_Behav['Run']==run) & (DF_Behav['CAP']==cap)]['TR'].unique())
    hit_lines = []
    for TR in hit_TRs:
        plot_layout = plot_layout * hv.VLine(TR).opts(color='black', line_width=3)
    
    return plot_layout

@pn.depends(SBJ_Select.param.value, RUN_Select.param.value, CAP_TYPE_Select.param.value, watch = True)
def get_cap_hits_mosaic(sbj,run,cap):
    cap_texts    = pn.Row()
    cap_radars   = pn.Row()
    cap_titles   = pn.Row()
    cap_rt_maps  = []
    cap_off_maps = []
    hitIDs_for_this_cap = list(DF_Behav[(DF_Behav['Subject']==sbj) & (DF_Behav['Run']==run) & (DF_Behav['CAP']==cap)]['Hit_ID'].unique())
    for hitID in hitIDs_for_this_cap:
        # Title
        hit_tr = DF_Behav[(DF_Behav['Subject']==sbj) & (DF_Behav['Run']==run) & (DF_Behav['CAP']==cap) & (DF_Behav['Hit_ID']==hitID)]['TR'].unique()[0]
        title  = '<h3>'+cap+' | #Hit = '+str(hitID)+' | TR = '+str(hit_tr)+'</h3>'
        cap_titles.append(pn.pane.HTML(title, width=420))
        # Transcript
        cap_texts.append(pn.panel(DF_Behav[(DF_Behav['Subject']==sbj) & (DF_Behav['Run']==run) & (DF_Behav['CAP']==cap) & (DF_Behav['Hit_ID']==hitID)]['Text'].iloc[0], width=420))
        # Radar Plots
        likert_values  = list(DF_Behav[(DF_Behav['Subject']==sbj) & (DF_Behav['Run']==run) & (DF_Behav['CAP']==cap) & (DF_Behav['Hit_ID']==hitID)]['Resp_Int'].values)
        likert_strings = list(DF_Behav[(DF_Behav['Subject']==sbj) & (DF_Behav['Run']==run) & (DF_Behav['CAP']==cap) & (DF_Behav['Hit_ID']==hitID)]['Resp_Str'].values)
        likert_strings = [a.replace('\\n',' ') for a in likert_strings]
        cap_radars.append(generate_radar_chart_from_vals(likert_values,likert_strings))
    return pn.Column(cap_titles, cap_texts, cap_radars)


# -

pn.pane.HTML('<h3>hi</h3>', width=420)

APP = pn.Column(Title,
                pn.Row(SBJ_Select, RUN_Select, hit_count_table),
                get_svrscore_plot,
                CAP_TYPE_Select,
                get_cap_hits_mosaic)

APP.show()

# ***
