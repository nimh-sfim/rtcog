import os.path as osp
import glob
import ntpath
import re
import pandas as pd
import numpy  as np

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

def get_runs_per_subject(PRJDIR, s):
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

def grab_all_files(PRJDIR, subjects_list):
    Runs_Per_Subject = {}
    Response_Files   = {}
    Transcript_Files = {}
    Hit_Files        = {}
    SVRscores_Files  = {}
    Options_Files    = {}
    QAOnset_Files    = {}
    QAOffset_Files   = {}
    for s in subjects_list:
        Runs_Per_Subject[s] = get_runs_per_subject(PRJDIR, s)
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
    output = {}
    output['Runs_Per_Subject'] = Runs_Per_Subject
    output['Response_Files']   = Response_Files
    output['Transcript_Files'] = Transcript_Files
    output['Hit_Files']        = Hit_Files
    output['SVRscores_Files']  = SVRscores_Files
    output['Options_Files']    = Options_Files
    output['QAOnset_Files']    = QAOnset_Files
    output['QAOffset_Files']   = QAOffset_Files
    return output

def load_all_data(PRJDIR, SBJs, CAP_labels, Q2N, get_file_dict=True):
    Files_Dict       = grab_all_files(PRJDIR,SBJs)
    DF               = pd.DataFrame(columns=['Subject','Run','Hit_ID','CAP','TR','Question','Resp_Str','Resp_Int','Resp_Time','Text'])
    Hit_Files        = Files_Dict['Hit_Files']
    Response_Files   = Files_Dict['Response_Files']
    Transcript_Files = Files_Dict['Transcript_Files']
    Runs_Per_Subject = Files_Dict['Runs_Per_Subject']
    for s in SBJs:
        for r in Runs_Per_Subject[s]:
            hit_info   = pd.DataFrame(np.load(Hit_Files[(s,r)]).T, columns=CAP_labels)
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
                    DF = DF.append({'Subject':s,
                                    'Run'      : r,
                                    'Hit_ID'   : i+1,
                                    'TR'       : int(hits.iloc[i].name),
                                    'CAP'      : hits.iloc[i][hits.iloc[i]==1.0].index[0],
                                    'Question' : q,
                                    'Resp_Str' : resp[q][0],
                                    'Resp_Int' : Q2N[q][resp[q][0]],
                                    'Resp_Time': float(resp[q][1]),
                                    'Text'     : transcript},ignore_index=True)
    if get_file_dict:
        return DF, Files_Dict
    else:
        return DF