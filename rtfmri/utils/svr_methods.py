import numpy as np
import logging
log     = logging.getLogger("svr_methods")
log.setLevel(logging.DEBUG)
log_fmt = logging.Formatter('[%(levelname)s - svr_methods]: %(message)s')
log_ch  = logging.StreamHandler()
log_ch.setFormatter(log_fmt)
log_ch.setLevel(logging.DEBUG)
log.addHandler(log_ch)

# TODO: Write documentation for how others can create a hit methodd
def is_hit_rt01(t, template_labels, svrscores, hit_thr, nconsec_vols):
    """
    Determines if a specific time point `t` represents a "hit" for a template based on if SVR scores
    exceed a threshold.

    A time point is considered a hit if:
    - Exactly one template's SVR score is >= `hit_thr` at time `t`
    - That same template has also been above `hit_thr` nconsec_vols (including the current volume)

    Parameters
    -----------
    t : int
        The current time point.
    
    template_labels : list of str
        List of template labels corresponding to the rows of `svrscores`.

    svrscores : np.ndarray of shape (n_templates, n_timepoints)
        SVR scores for each template across time thus far.

    hit_thr : float
        The score threshold above which a template is considered "active".

    nconsec_vols : int
        Number of consecutive time points (including current) that the template must exceed the threshold.

    Returns
    -------
    str or None
        The label of the template that qualifies as a hit at time `t`, or None if no hit is found.
    """
    log.info(f' ============== entering is_hit_rt01 [t={t}] [thr={hit_thr}]=======================')
    this_t_svrscores = svrscores[:,t]
    this_t_matches = np.where(this_t_svrscores >= hit_thr)[0]
    nmatches = len(this_t_matches)

    log.debug(' === is_hit_rt01 - this_t_svrscores: ' + ', '.join(f'{f:.3f}' for f in this_t_svrscores))
    log.debug(' === is_hit_rt01 - nmatches %d' % nmatches)
    
    if nmatches != 1:
        return None # Hit is none if more than one template is above threshold
    
    this_t_hit_label = template_labels[this_t_matches[0]]
    this_t_hit = this_t_matches[0]

    # Ensure template exceeds threshold for nconsec_vols time points (including current)
    prev_svr = svrscores[:, t - nconsec_vols + 1 : t]
    if np.all(prev_svr[this_t_hit] >= hit_thr):
        return this_t_hit_label

    