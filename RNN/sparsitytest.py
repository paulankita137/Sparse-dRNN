# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 13:54:51 2023

@author: ap3737
"""

import numpy as np
import time

def sparse_fullforce_test(dt, showplots=0):
    dt_per_s = round(1/dt)
    
    # From the paper, and the online demo:
    t = np.expand_dims(np.linspace(0,2,2*dt_per_s+1),1)
    omega = np.zeros((2*dt_per_s+1,1))
    omega = np.linspace(2*np.pi, 6*np.pi, 1*dt_per_s+1)
    targ = np.zeros((2*dt_per_s+1,1))
    targ[0:(1*dt_per_s+1),0] = np.sin(t[0:(1*dt_per_s+1),0]*omega)
    targ[1*dt_per_s:(2*dt_per_s+1)] = -np.flipud(targ[0:(1*dt_per_s+1)])
    
    
    inp = np.zeros(targ.shape)
    inp[0:round(0.05*dt_per_s),0] = np.ones((round(0.05*dt_per_s)))
    hints = np.zeros(targ.shape)

    
    return inp, targ, hints

sparse_fullforce_test(dt=0.001,showplots=1);




import sparsityRNN
#%load_ext line_profiler

p = sparsityRNN.create_parameters(dt=0.001)
p['g'] = 1.5 # From paper
p['ff_num_batches'] = 2
p['ff_trials_per_batch'] = 2
p['test_init_trials']=5
p['init_act_scale'] = 0.1
p['network_size'] = 200
p['noise_std'] = np.sqrt(2*p['dt'])

rnn = sparsityRNN.differentialRNN(p,1,1)

rnn.train(sparse_fullforce_test, monitor_training=1)


rnn.test(sparse_fullforce_test)





rnn.p['test_trials']=25
rnn.test(sparse_fullforce_test)
