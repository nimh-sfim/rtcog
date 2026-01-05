import numpy as np
import itertools

from rtcog.preproc.helpers.preproc_utils import welford

class KalmanFilter:
    """
    Kalman filter for low pass filtering, spike removal, and signal smoothing.
    """
    def __init__(self, mask_Nv, n_cores, pool):
        self.mask_Nv = mask_Nv
        self.n_cores = n_cores
        self.pool = pool
        self.initialized = False
        self.Nv = mask_Nv
        self.welford_M = np.zeros(self.Nv)
        self.welford_S = np.zeros(self.Nv)
        self.welford_std = np.zeros(self.Nv)
        self.S_x = np.zeros(self.Nv)
        self.S_P = np.zeros(self.Nv)
        self.fPositDerivSpike = np.zeros(self.Nv)
        self.fNegatDerivSpike = np.zeros(self.Nv)

    def initialize_pool(self):
        if not self.initialized:
            _ = self.pool.map(KalmanFilter.kalman_filter_mv, self.initialize_kalman_pool())
            self.initialized = True

    def update_welford(self, k, x):
        M, S, std = welford(k, x, self.welford_M, self.welford_S)
        self.welford_M = M
        self.welford_S = S
        self.welford_std = std

    def run_volume(self, n, data):
        """
        Run Kalman filter on a single volume. Parallelizes via pools.
        """
        if n > 2:
            v_groups = [int(i) for i in np.linspace(0, self.Nv, self.n_cores + 1)]
            v_start = v_groups[:-1]
            v_end = v_groups[1:]
            o_data, o_fPos, o_fNeg = [], [], []
            o_S_x, o_S_P = [], []
            data_std_sq = np.power(self.welford_std, 2)
            inputs = ({'d': data[v_s:v_e],
                       'std': self.welford_std[v_s:v_e],
                       'S_x': self.S_x[v_s:v_e],
                       'S_P': self.S_P[v_s:v_e],
                       'S_Q': 0.25 * data_std_sq[v_s:v_e],
                       'S_R': data_std_sq[v_s:v_e],
                       'fPos': self.fPositDerivSpike[v_s:v_e],
                       'fNeg': self.fNegatDerivSpike[v_s:v_e],
                       'vox': np.arange(v_s, v_e)}
                      for v_s, v_e in zip(v_start, v_end))
            res = self.pool.map(KalmanFilter.kalman_filter_mv, inputs)
            for j in range(self.n_cores):
                o_data.append(res[j][0])
                o_fPos.append(res[j][1])
                o_fNeg.append(res[j][2])
                o_S_x.append(res[j][3])
                o_S_P.append(res[j][4])
            # Combine results
            all_data = list(itertools.chain(*o_data))
            all_S_x = list(itertools.chain(*o_S_x))
            all_S_P = list(itertools.chain(*o_S_P))
            all_fPos = list(itertools.chain(*o_fPos))
            all_fNeg = list(itertools.chain(*o_fNeg))
            self.S_x[:] = all_S_x
            self.S_P[:] = all_S_P
            self.fPositDerivSpike[:] = all_fPos
            self.fNegatDerivSpike[:] = all_fNeg
            return np.array(all_data).reshape((self.Nv, 1))
        else:
            return np.zeros((self.Nv, 1))

    def initialize_kalman_pool(self):
        """Initialize pool up front to avoid delay later"""
        return [
            {
                'd': np.zeros((1, 1)),
                'std': np.zeros((1)),
                'S_x': np.zeros((1)),
                'S_P': np.zeros((1)),
                'S_Q': np.zeros((1)),
                'S_R': np.zeros((1)),
                'fPos': np.zeros((1)),
                'fNeg': np.zeros((1)),
                'vox': np.zeros((1))
            }
            for _ in range(self.n_cores)
        ]

    @staticmethod
    def kalman_filter_mv(input_dict):
        Nv = input_dict['vox'].shape[0]
        out_d_mv    = []
        out_fPos_mv = []
        out_fNeg_mv = []
        out_S_x_mv  = []
        out_S_P_mv  = []
        out_vox_mv  = []
        for v in np.arange(Nv):
            input_d = input_dict['d'][v, 0]
            input_ts_STD = input_dict['std'][v]
            input_S = {'Q': input_dict['S_Q'][v],
                       'R': input_dict['S_R'][v],
                       'x': input_dict['S_x'][v],
                       'P': input_dict['S_P'][v]}
            input_fPos = input_dict['fPos'][v]
            input_fNeg = input_dict['fNeg'][v]
            kalmTh     = 0.9 * input_ts_STD
            [out_d, out_S, out_fPos, out_fNeg] = KalmanFilter._kalman_filter(kalmTh, input_d,input_S, input_fPos, input_fNeg)


            for (l,i) in zip([out_d_mv,out_fPos_mv,out_fNeg_mv,out_S_x_mv,out_S_P_mv, out_vox_mv],
                             [out_d,   out_fPos,   out_fNeg,   out_S['x'],out_S['P'],input_dict['vox'][v]]):
                l.append(i)
        return [out_d_mv, out_fPos_mv, out_fNeg_mv, out_S_x_mv, out_S_P_mv, out_vox_mv]
    
    @staticmethod
    def _kalman_filter(kalmTh, kalmIn, S, fPositDerivSpike, fNegatDerivSpike):
        # TODO refactor to class method
        """
        Perform Kalman Low-pass filtering and despiking
        Based on: Koush Y., Zvyagintsev M., Dyck M., Mathiak K.A., Mathiak K. (2012)
        Signal quality and Bayesian signal processing in neurofeedback based on 
        real-time fMRI. Neuroimage 59:478-89
        
        Parameters
        ----------
        kalmTh : 
            Spike-detection Threshold
        kalmIn :
            Input data
        S :
            Parameter structure
        fPositDerivSpike :
            Counter for spikes with positive derivative
        fNegatDerivSpike :
            Counter for spikes with negative derivative
        
        Returns
        -------
        kalmOut :
            Filtered Output
        S :
            Parameter Structure
        fPositDerivSpike :
            Counter for spikes with positive derivative
        fNegatDerivSpike :
            Counter for spikes with negative derivative
        """
        
        # Preset
        A = 1
        H = 1
        I = 1
        
        # Kalman Filter
        S['x'] = A * S['x']
        S['P'] = ( A * S['P'] * A ) + S['Q']
        if ((H*S['P']*H) + S['R']) > 0.0:
            K = S['P'] * H * (1/( (H*S['P']*H) + S['R']) )
        else:
            K = 0
        tmp_x  = S['x']
        tmp_p  = S['P']
        diff   = K * (kalmIn - (H*S['x']))
        S['x'] = S['x'] + diff
        S['P'] = (I - (K * H)) * S['P']
        # Spikes identification and correction
        if np.abs(diff) < kalmTh:
            kalmOut = H * S['x']
            fNegatDerivSpike = 0;
            fPositDerivSpike = 0;
        else:
            if diff > 0:
                if fPositDerivSpike < 1:
                    kalmOut = H * tmp_x
                    S['x'] = tmp_x
                    S['P'] = tmp_p
                    fPositDerivSpike = fPositDerivSpike + 1
                else:
                    kalmOut = H * S['x']
                    fPositDerivSpike = 0
            else:
                if fNegatDerivSpike < 1:
                    kalmOut = H * tmp_x
                    S['x'] = tmp_x
                    S['P'] = tmp_p
                    fNegatDerivSpike = fNegatDerivSpike + 1
                else:
                    kalmOut = H * S['x']
                    fNegatDerivSpike = 0

        return kalmOut,S,fPositDerivSpike,fNegatDerivSpike