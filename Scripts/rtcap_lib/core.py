from scipy.signal import exponential

def create_win(M,center=0,tau=3):
    win = exponential(M,center,tau,False)
    print ('++ Create Window: Window Values [%s]' % str(win))
    return win