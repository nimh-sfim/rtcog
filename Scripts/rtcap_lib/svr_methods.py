import numpy as np

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