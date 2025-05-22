import numpy as np
import logging
log     = logging.getLogger("svr_methods")
log.setLevel(logging.DEBUG)
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


def is_hit_rt01(t, caps_labels, svrscores, hit_thr, nconsec_vols):
    """Determines if a specific time point `t` represents a "hit" for a CAP based on if SVR scores
    exceed a threshold.

    A time point is considered a hit if:
    - Exactly one CAP's SVR score is >= `hit_thr` at time `t`
    - That same CAP has also been above `hit_thr` nconsec_vols (including the current volume)

    Parameters:
    ----------
    t : int
        The current time point.
    
    caps_labels : list of str
        List of CAP labels corresponding to the rows of `svrscores`.

    svrscores : np.ndarray of shape (n_caps, n_timepoints)
        SVR scores for each CAP across time thus far.

    hit_thr : float
        The score threshold above which a CAP is considered "active".

    nconsec_vols : int
        Number of consecutive time points (including current) that the CAP must exceed the threshold.

    Returns:
    -------
    str or None
        The label of the CAP that qualifies as a hit at time `t`, or None if no hit is found.
    """
    log.info(f' ============== entering is_hit_rt01 [t={t}] [thr={hit_thr}]=======================')
    this_t_svrscores = svrscores[:,t]
    this_t_matches = np.where(this_t_svrscores >= hit_thr)[0]
    nmatches = len(this_t_matches)

    log.debug(' === is_hit_rt01 - this_t_svrscores: ' + ', '.join(f'{f:.3f}' for f in this_t_svrscores))
    log.debug(' === is_hit_rt01 - nmatches %d' % nmatches)
    
    if nmatches != 1:
        return None # Hit is none if more than one CAP is above threshold
    
    this_t_hit_label = caps_labels[this_t_matches[0]]
    this_t_hit = this_t_matches[0]

    # Ensure CAP exceeds threshold for nconsec_vols time points (including current)
    prev_svr = svrscores[:, t - nconsec_vols + 1 : t]
    if np.all(prev_svr[this_t_hit] >= hit_thr):
        return this_t_hit_label

    



# def is_hit_rt01(t,caps_labels,svrscores,hit_thr,hit_v4hit):
#     log.info(' ============== entering is_hit_rt01 [t=%d] [thr=%f]=======================' % (t,hit_thr))

#     hit = None
#     this_t_svrscores = svrscores[:,t] 
#     this_t_above_thr = this_t_svrscores >= hit_thr
#     this_t_nmatches  = np.sum(this_t_above_thr)

#     log.debug(' === is_hit_rt01 - this_t_svrscores: ' + ', '.join(f'{f:.3f}' for f in this_t_svrscores))
#     log.debug(' === is_hit_rt01 - this_t_above_thr: ' + ', '.join(f'{f:.3f}' for f in this_t_above_thr))
#     log.debug(' === is_hit_rt01 - this_t_nmatches %d' % this_t_nmatches)

#     # Obtain list of CAPs with svrscore > th at this given t volume
#     if this_t_nmatches > 0:
#         this_t_matches = [caps_labels[i] for i in np.where(this_t_svrscores >= hit_thr)[0]]
#     else:
#         this_t_matches = None
#     log.debug(' === is_hit_rt01 - this_t_nmatches [%s]' % str(this_t_matches))
#     # I will consider this volume a hit, only if a single CAP is above threshold
#     if this_t_nmatches == 1:
#         this_t_hit   = this_t_matches[0] # CAP with svrscore > thresh for volume (t)
#         above_thr    = np.repeat(False,hit_v4hit-1) # Container with False for all previous volumes (whether or not the CAP was also a hit)
#         log.debug(' === is_hit_rt01 - above_thr %s' % str(above_thr))
#         for ii,tt in enumerate(np.arange(t-hit_v4hit+1, t)):
#             aux_svrscores = svrscores[:,tt]
#             aux_matches   = [caps_labels[i] for i in np.where(aux_svrscores >= hit_thr)[0]]
#             if this_t_hit in aux_matches:
#                 above_thr[ii] = True
#             log.debug(' === is_hit_rt01 [%d] - above_thr %s' % (tt,str(above_thr)))
#         log.info(' === is_hit_rt01 [FINAL] - above_thr %s' % str(above_thr))
#         if np.all(above_thr):
#             hit = this_t_hit
#     return hit