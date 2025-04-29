import numpy as np
import logging
log     = logging.getLogger("svr_methods")
log.setLevel(logging.INFO)
log_fmt = logging.Formatter('[%(levelname)s - svr_methods]: %(message)s')
log_ch  = logging.StreamHandler()
log_ch.setFormatter(log_fmt)
log_ch.setLevel(logging.DEBUG)
log.addHandler(log_ch)
def is_hit_method01(vol, CAP_labels, hit_opts, SVRscores, rtPredictions, Vol_LastQEnded):
    pred_this_vol = SVRscores.loc[vol]        # SVR scores for this volume
    # 1) See how many CAPs have score > z_th
    num_matches = np.sum(pred_this_vol > hit_opts['z_th'])     # Number of CAPs with score above threshold
    if num_matches > 0 :                                       # Extract Label of all CAPs with score > z_th
        matches = [CAP_labels[i] for i in np.where(pred_this_vol > hit_opts['z_th'])[0]]
    else:
        matches = None
    
    # 2) If only one match, and looking for hits:
    if (num_matches == 1) and (vol > Vol_LastQEnded + hit_opts['vols2discard_postHit'] + hit_opts['vols4hit']):
        this_hit = matches[0]                                                    # Single CAP with score > z_th
        is_above_z_th = np.repeat(False,hit_opts['vols4hit']-1)                  # Container initialized to False for all vols needed to be accounted for
        for ii,i in enumerate(np.arange(vol-hit_opts['vols4hit']+1, vol)):       # For all vols that have a say
            if rtPredictions != {}:                                       # Extract what CAPs were above z_th
                relevant_matches = rtPredictions[i]['matches']
            else:
                relevant_matches = None
            if (relevant_matches is not None) and (this_hit in relevant_matches): # If this_hit was also hit in this volume, set relevant_matches[vol] = True
                is_above_z_th[ii] = True
        if np.all(is_above_z_th):                                                 # If this_hit was also a hit in all past 'vols4hit' volumes, then hit is confirmed.
            hit = this_hit
        else:
            hit = None
    else:
        hit = None
    
    return matches, hit

def is_hit_rt01(t,caps_labels,svrscores,hit_thr,hit_v4hit):
    log.info(' ============== entering is_hit_rt01 [t=%d] [thr=%f]=======================' % (t,hit_thr))
    hit = None
    this_t_svrscores = svrscores[:,t] 
    log.debug(' === is_hit_rt01 - this_t_svrscores: ' + ', '.join(f'{f:.3f}' for f in this_t_svrscores))
    this_t_above_thr = this_t_svrscores >= hit_thr
    log.debug(' === is_hit_rt01 - this_t_above_thr: ' + ', '.join(f'{f:.3f}' for f in this_t_above_thr))
    this_t_nmatches  = np.sum(this_t_above_thr)
    log.debug(' === is_hit_rt01 - this_t_nmatches %d' % this_t_nmatches)
    # Obtain list of CAPs with svrscore > th at this given t volume
    if this_t_nmatches > 0:
        this_t_matches = [caps_labels[i] for i in np.where(this_t_svrscores >= hit_thr)[0]]
    else:
        this_t_matches = None
    log.debug(' === is_hit_rt01 - this_t_nmatches [%s]' % str(this_t_matches))
    # I will consider this volume a hit, only if a single CAP is above threshold
    if this_t_nmatches == 1:
        this_t_hit   = this_t_matches[0]            # CAP with svrscore > thresh for volume (t)
        above_thr    = np.repeat(False,hit_v4hit-1) # Container with False for all previous volumes (whether or not the CAP was also a hit)
        log.debug(' === is_hit_rt01 - above_thr %s' % str(above_thr))
        for ii,tt in enumerate(np.arange(t-hit_v4hit+1, t)):
            aux_svrscores = svrscores[:,tt]
            aux_matches   = [caps_labels[i] for i in np.where(aux_svrscores >= hit_thr)[0]]
            if this_t_hit in aux_matches:
                above_thr[ii] = True
            log.debug(' === is_hit_rt01 [%d] - above_thr %s' % (tt,str(above_thr)))
        log.info(' === is_hit_rt01 [FINAL] - above_thr %s' % str(above_thr))
        if np.all(above_thr):
            hit = this_t_hit
    return hit