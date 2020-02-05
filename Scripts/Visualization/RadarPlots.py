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

import os.path as osp
import pandas as pd
import numpy as np
import panel as pn
import holoviews as hv
import hvplot.pandas
import json
from bokeh.models import ColumnDataSource, LabelSet, HoverTool
from bokeh.plotting import figure
from nilearn.image import load_img
pn.extension()
import sys
sys.path.append('/data/SFIMJGC/PRJ_rtCAPs/rtcaps/')
from rtcap_lib.utils.exp_defs import PRJDIR, CAP_indexes, CAP_labels, CAPLabel2Int, Question_Dict, Question_ToNum, CAP_colors
from rtcap_lib.utils.files import get_runs_per_subject, grab_all_files, load_all_data
from rtcap_lib.dashboards.radar import generate_radar_chart_from_vals, generate_avg_radar_chart_for_cap

SBJs          = ['PILOT03','PILOT04','PILOT05']
DASHBOARD_DIR = osp.join(PRJDIR,'Dashboards')

DF_Behav, Files = load_all_data(PRJDIR,SBJs, CAP_labels, Question_ToNum, get_file_dict=True)
for k,v in Files.items():
    exec ('%s=%s'%(k,v))


def disable_logo(plot, element):
    plot.state.toolbar.logo = None


# ***
# # Graphical Stuff - Dashboard

# +
Title            = pn.panel('# Individual Hit Summary View')
SBJ_Select       = pn.widgets.Select(name='Subject ID:', value=SBJs[0], options=SBJs)
initial_run_list = get_runs_per_subject(PRJDIR, SBJs[0])
RUN_Select       = pn.widgets.Select(name='Run ID:',     value=initial_run_list[0], options=initial_run_list)
initial_cap_list = list(DF_Behav[(DF_Behav['Subject']==SBJs[0]) & (DF_Behav['Run']==initial_run_list[0])]['CAP'].unique())
CAP_TYPE_Select  = pn.widgets.Select(name='CAP Type:', value = initial_cap_list[0], options=initial_cap_list)
@pn.depends(SBJ_Select.param.value, watch=True)
def _update_run_select_list(sbj):
    run_list = get_runs_per_subject(PRJDIR, sbj)
    RUN_Select.options = run_list
    RUN_Select.value   = run_list[0]
    
@pn.depends(SBJ_Select.param.value, RUN_Select.param.value)
def hit_count_table(sbj,run):
    hits_file = osp.join(PRJDIR,'PrcsData',sbj,'D01_OriginalData',sbj+'_'+run+'.hits.npy')
    hits = pd.DataFrame(np.load(hits_file).T,columns=CAP_labels)
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
    
    SVR_Scores_DF       = pd.DataFrame(svrscores, columns=CAP_labels)
    SVR_Scores_DF['TR'] = SVR_Scores_DF.index
    SVRscores_curve     = SVR_Scores_DF.hvplot(legend='top', label='SVR Scores', x='TR').opts(width=1000, height=200, toolbar='right',finalize_hooks=[disable_logo])
    Threshold_line      = hv.HLine(ZTH).opts(color='black', line_dash='dashed', line_width=1)
    Hits_ToPlot         = hits * svrscores
    Hits_ToPlot[Hits_ToPlot==0.0] = None
    Hits_DF             = pd.DataFrame(Hits_ToPlot, columns=CAP_labels)
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
    hitIDs_for_this_cap = list(DF_Behav[(DF_Behav['Subject']==sbj) & (DF_Behav['Run']==run) & (DF_Behav['CAP']==cap)]['Hit_ID'].unique())
    layout = pn.Row()
    for idx,hitID in enumerate(hitIDs_for_this_cap):
        # Title
        hit_tr = DF_Behav[(DF_Behav['Subject']==sbj) & (DF_Behav['Run']==run) & (DF_Behav['CAP']==cap) & (DF_Behav['Hit_ID']==hitID)]['TR'].unique()[0]
        title  = '<h2>'+cap+' | #Hit = '+str(hitID)+' | TR = '+str(hit_tr)+'</h2>'
        pn_title = pn.pane.HTML(title, width=420)
        # Transcript
        pn_transcript = pn.panel(DF_Behav[(DF_Behav['Subject']==sbj) & (DF_Behav['Run']==run) & (DF_Behav['CAP']==cap) & (DF_Behav['Hit_ID']==hitID)]['Text'].iloc[0], width=420, height=125)
        # Radar Plots
        likert_values  = list(DF_Behav[(DF_Behav['Subject']==sbj) & (DF_Behav['Run']==run) & (DF_Behav['CAP']==cap) & (DF_Behav['Hit_ID']==hitID)]['Resp_Int'].values)
        likert_strings = list(DF_Behav[(DF_Behav['Subject']==sbj) & (DF_Behav['Run']==run) & (DF_Behav['CAP']==cap) & (DF_Behav['Hit_ID']==hitID)]['Resp_Str'].values)
        likert_strings = [a.replace('\\n',' ') for a in likert_strings]
        pn_radar = generate_radar_chart_from_vals(likert_values,likert_strings, Question_Dict, color=CAP_colors[cap])
        # RT Maps
        hit_sv_LL = pn.pane.PNG(osp.join(DASHBOARD_DIR,'SurfViews',sbj+'_'+run+'.Hit_'+cap+'_'+str(idx+1).zfill(2)+'_Left_Lateral.png'), height=130)
        hit_sv_LM = pn.pane.PNG(osp.join(DASHBOARD_DIR,'SurfViews',sbj+'_'+run+'.Hit_'+cap+'_'+str(idx+1).zfill(2)+'_Left_Medial.png'), height=130)
        hit_sv_RL = pn.pane.PNG(osp.join(DASHBOARD_DIR,'SurfViews',sbj+'_'+run+'.Hit_'+cap+'_'+str(idx+1).zfill(2)+'_Right_Lateral.png'), height=130)
        hit_sv_RM = pn.pane.PNG(osp.join(DASHBOARD_DIR,'SurfViews',sbj+'_'+run+'.Hit_'+cap+'_'+str(idx+1).zfill(2)+'_Right_Medial.png'), height=130)
        rt_pn_caps_sv = pn.GridBox(ncols=2,nrows=2, objects=[hit_sv_LL,hit_sv_RL,hit_sv_LM,hit_sv_RM], height=300, background='white', margin=(10,10))
        
        # Offline Maps
        hit_sv_LL = pn.pane.PNG(osp.join(DASHBOARD_DIR,'SurfViews',sbj+'_'+run+'.offline.Hit_'+cap+'_'+str(idx+1).zfill(2)+'_Left_Lateral.png'), height=130)
        hit_sv_LM = pn.pane.PNG(osp.join(DASHBOARD_DIR,'SurfViews',sbj+'_'+run+'.offline.Hit_'+cap+'_'+str(idx+1).zfill(2)+'_Left_Medial.png'), height=130)
        hit_sv_RL = pn.pane.PNG(osp.join(DASHBOARD_DIR,'SurfViews',sbj+'_'+run+'.offline.Hit_'+cap+'_'+str(idx+1).zfill(2)+'_Right_Lateral.png'), height=130)
        hit_sv_RM = pn.pane.PNG(osp.join(DASHBOARD_DIR,'SurfViews',sbj+'_'+run+'.offline.Hit_'+cap+'_'+str(idx+1).zfill(2)+'_Right_Medial.png'), height=130)
        offline_pn_caps_sv = pn.GridBox(ncols=2,nrows=2, objects=[hit_sv_LL,hit_sv_RL,hit_sv_LM,hit_sv_RM], height=300, background='white', margin=(10,10))
        layout.append(pn.Column(pn.Row(pn.layout.HSpacer(), pn_title, pn.layout.HSpacer()),
                                pn_transcript,
                                pn_radar, 
                                pn.Row(pn.layout.HSpacer(), pn.pane.HTML('<h2>Realtime Preprocessing</h2>'), pn.layout.HSpacer()),
                                rt_pn_caps_sv, 
                                pn.Row(pn.layout.HSpacer(), pn.pane.HTML('<h2>Offline Preprocessing</h2>'), pn.layout.HSpacer()),
                                offline_pn_caps_sv)) 
    return layout


@pn.depends(CAP_TYPE_Select.param.value, watch = True)
def get_cap_surface(cap):
    caps_LL_path = osp.join(PRJDIR,'Others','Video','CAPs_'+cap+'_Left_Lateral.png')
    caps_LM_path = osp.join(PRJDIR,'Others','Video','CAPs_'+cap+'_Left_Medial.png')
    caps_RL_path = osp.join(PRJDIR,'Others','Video','CAPs_'+cap+'_Right_Lateral.png')
    caps_RM_path = osp.join(PRJDIR,'Others','Video','CAPs_'+cap+'_Right_Medial.png')
    caps_LL = pn.pane.PNG(caps_LL_path, height=100)
    caps_LM = pn.pane.PNG(caps_LM_path, height=100)
    caps_RL = pn.pane.PNG(caps_RL_path, height=100)
    caps_RM = pn.pane.PNG(caps_RM_path, height=100)
    return pn.GridBox(ncols=2,nrows=2, objects=[caps_LL,caps_RL,caps_LM,caps_RM], height=210, background='white', margin=(10,10))

@pn.depends(SBJ_Select.param.value,RUN_Select.param.value, CAP_TYPE_Select.param.value, watch=True)
def get_cap_combined_radar(sbj,run,cap):
    figure = generate_avg_radar_chart_for_cap(DF_Behav,cap,Question_Dict,sbj=sbj, run=run)
    figure.height = 250
    figure.width  = 250
    output = pn.panel(figure)
    return output



# -

APP = pn.Column(Title,
                pn.Row(pn.Column(SBJ_Select, RUN_Select,hit_count_table, CAP_TYPE_Select), get_cap_surface, get_cap_combined_radar,get_svrscore_plot),
                get_cap_hits_mosaic)

APP.servable()


