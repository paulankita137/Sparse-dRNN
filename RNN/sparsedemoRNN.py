# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 13:56:24 2023

@author: ap3737
"""


import numpy as np
import numpy.random as npr
from scipy import sparse
import matplotlib.pyplot as plt


def create_parameters(dt=0.001):
    
    p = {'network_size': 500,              # Number of units in network
        'dt': dt,                         # Time step for RNN.
        'tau': 0.01,                     # Time constant of neurons
        'noise_std': 0,                    # Amount of noise on RNN neurons
        'g': 1,                         # Gain of the network (scaling for recurrent weights)
        #'p': 1,                            # Controls the sparseness of the network. 1=>fully connected.
        'p': 0.02,
        'inp_scale': 1,                 # Scales the initialization of input weights
        'out_scale': 1,                 # Scales the initialization of output weights
        'bias_scale': 0,                # Scales the amount of bias on each unit
        'init_act_scale' : 1,            # Scales how activity is initialized
        ##### Training parameters for full-FORCE 
        'ff_steps_per_update': 2 ,  # Average number of steps per weight update
        'ff_alpha': 0.001,  # "Learning rate" parameter (should be between 1 and 100)
        'ff_num_batches': 10,
        'ff_trials_per_batch': 100,  # Number of inputs/targets to go through
        'ff_init_trials': 3,
        #### Testing parameters
        'test_trials': 10,
        'test_init_trials': 1,
    
    }
    return p

class RNN:
    

    def __init__(self, hyperparameters, num_inputs, num_outputs):
        
        self.p = hyperparameters
        rnn_size = self.p['network_size']
        self.rnn_par = {'inp_weights': (npr.rand(num_inputs, rnn_size)-0.5) * 2 * self.p['inp_scale'],
                        'rec_weights': np.array(sparse.random(rnn_size, rnn_size, self.p['p'], 
                                                data_rvs = npr.randn).todense()) * self.p['g']/np.sqrt(rnn_size),
                        'out_weights': (npr.rand(rnn_size, num_outputs)-0.5) * 2 * self.p['out_scale'],
                        'bias': (npr.rand(1,rnn_size)-0.5)*2 * self.p['bias_scale'],
                        'theta_weights':(npr.rand(rnn_size, num_outputs)-0.5) * 2 * self.p['out_scale']
                        }
        self.act = npr.randn(1,self.rnn_par['rec_weights'].shape[0])*self.p['init_act_scale']

        self.theta = self.p['p']

    def initialize_act(self):
        '''Any time you want to reset the activity of the network to low random values'''
        self.act = npr.randn(1,self.rnn_par['rec_weights'].shape[0])*self.p['init_act_scale']

    def run(self,inputs, record_flag=0):
       
        p = self.p
        activity = self.act
        theta = self.theta
        rnn_params = self.rnn_par
        def rnn_update(inp, activity):
            dx = p['dt']/p['tau'] * (-activity + np.dot(np.tanh(activity),rnn_params['rec_weights']) + 
                                        np.dot(inp, rnn_params['inp_weights']) + rnn_params['bias'] + 
                                        npr.randn(1,activity.shape[1]) * p['noise_std'])
            return activity + dx
        
        
        
        
        def rnn_output(activity):
            return np.dot(np.tanh(activity), rnn_params['out_weights'])
        
        
        def rnn_theta(theta):
            theta=self.p['p'] + theta
            return theta
       
        def rnn_sparse_output(activity,theta):
            
            xi=np.sign(activity)*np.tanh(abs(activity) - theta)
            
            
            return np.dot(xi, rnn_params['out_weights'])
        
        
        
        activity_whole = []
        output_whole = []
        if record_flag==1:  # Record output and activity only if this is active
            activity_whole = np.zeros(((inputs.shape[0]),rnn_params['rec_weights'].shape[0]))
            t=0
         
        for inp in inputs:
            activity = rnn_update(inp, activity)
            #output_whole.append(rnn_output(activity))
            #theta = rnn_theta(theta)
            output_whole.append(rnn_sparse_output(activity, theta))

            if record_flag==1:
                activity_whole[t,:] = activity
                t+=1
        output_whole = np.reshape(output_whole, (inputs.shape[0],-1))
        self.act = activity
        
        return output_whole, activity_whole



    def train(self,inps_and_targs, monitor_training=0, **kwargs):
        
        # First, initialize some parameters
        p = self.p
        self.initialize_act()
        N = p['network_size']
        self.rnn_par['rec_weights'] = np.zeros((N,N))
        self.rnn_par['out_weights'] = np.zeros((self.rnn_par['out_weights'].shape))


        # Need to initialize a target-generating network, used for computing error:
        # First, take some example inputs, targets, and hints to get the right shape
        try:
            D_inputs, D_targs, D_hints = inps_and_targs(dt=p['dt'], **kwargs)[0:3]
            D_num_inputs = D_inputs.shape[1]
            D_num_targs = D_targs.shape[1]
            D_num_hints = D_hints.shape[1]
            D_num_total_inps = D_num_inputs + D_num_targs + D_num_hints
        except:
            raise ValueError('Check your inps_and_targs function. Must have a hints output as well!')

        # Then instantiate the network and pull out some relevant weights
        DRNN = RNN(hyperparameters = self.p, num_inputs = D_num_total_inps, num_outputs = 1)
        w_targ = np.transpose(DRNN.rnn_par['inp_weights'][D_num_inputs:(D_num_inputs+D_num_targs),:])
        w_hint = np.transpose(DRNN.rnn_par['inp_weights'][(D_num_inputs+D_num_targs):D_num_total_inps,:])
        Jd = np.transpose(DRNN.rnn_par['rec_weights'])

        ################### Monitor training with these variables:
        J_err_ratio = []
        J_err_mag = []
        J_norm = []

        w_err_ratio = []
        w_err_mag = []
        w_norm = []
        
        
        theta_err_ratio = []
        theta_err_mag = []
        theta_norm = []
        ###################

        # Let the networks settle from the initial conditions
        print('Initializing',end="")
        for i in range(p['ff_init_trials']):
            print('.',end="")
            inp, targ, hints = inps_and_targs(dt=p['dt'], **kwargs)[0:3]
            D_total_inp = np.hstack((inp,targ,hints))
            DRNN.run(D_total_inp)
            self.run(inp)
        print('')

        # Now begin training
        print('Training network...')
        # Initialize the inverse correlation matrix
        P = np.eye(N)/p['ff_alpha']
        for batch in range(p['ff_num_batches']):
            print('Batch %g of %g, %g trials: ' % (batch+1, p['ff_num_batches'], p['ff_trials_per_batch']),end="")
            for trial in range(p['ff_trials_per_batch']):
                if np.mod(trial,50)==0:
                    print('')
                print('.',end="")
                # Create input, target, and hints. Combine for the driven network
                inp, targ, hints = inps_and_targs(dt=p['dt'], **kwargs)[0:3]  # Get relevant time series
                D_total_inp = np.hstack((inp,targ,hints))
                # For recording:
                dx = [] # Driven network activity
                x = []    # RNN activity
                z = []    # RNN output
                for t in range(len(inp)):
                    # Run both RNNs forward and get the activity. Record activity for potential plotting
                    dx_t = DRNN.run(D_total_inp[t:(t+1),:], record_flag=1)[1][:,0:5]
                    z_t, x_t = self.run(inp[t:(t+1),:], record_flag=1)

                    dx.append(np.squeeze(np.tanh(dx_t) + np.arange(5)*2))
                    z.append(np.squeeze(z_t))
                    x.append(np.squeeze(np.tanh(x_t[:,0:5]) + np.arange(5)*2))

                    if npr.rand() < (1/p['ff_steps_per_update']):
                        # Extract relevant values
                        r = np.transpose(np.tanh(self.act))
                        rd = np.transpose(np.tanh(DRNN.act))
                        J = np.transpose(self.rnn_par['rec_weights'])
                        w = np.transpose(self.rnn_par['out_weights'])
                        theta = self.theta
                        xi = np.transpose(np.sign(r)*np.tanh(abs(r) - theta))
                        xi = np.dot(xi, self.rnn_par['out_weights'])
                        # Now for the RLS algorithm:
                        # Compute errors
                        J_err = (np.dot(J,r) - np.dot(Jd,rd) 
                                -np.dot(w_targ,targ[t:(t+1),:].T) - np.dot(w_hint, hints[t:(t+1),:].T))
                        w_err = np.dot(w,r) - targ[t:(t+1),:].T
                        
                        
                        #theta_err =   np.prod(z_t,w_targ,np.sign(xi)) - np.prod(targ[t:(t+1),:].T,w_targ,np.sign(xi))
                        
                        #theta_err = np.prod(z_t * w_targ * np.sign(xi)) - np.prod(targ[t:(t+1), :].T * w_targ * np.sign(xi))

                        theta_err = np.prod(z_t * w_targ * np.sign(x_t)) - np.prod(targ[t:(t+1), :].T * w_targ * np.sign(x_t))
                        # Compute the gain (k) and running estimate of the inverse correlation matrix
                        Pr = np.dot(P,r)
                        k = np.transpose(Pr)/(1 + np.dot(np.transpose(r), Pr))
                        P = P - np.dot(Pr,k)

                        # Update weights
                        w = w - np.dot(w_err, k)
                        J = J - np.dot(J_err, k)
                        theta = theta - np.dot(theta_err, k)
                        
                        self.rnn_par['rec_weights'] = np.transpose(J)
                        self.rnn_par['out_weights'] = np.transpose(w)
                        self.rnn_par['theta_weights'] = np.transpose(theta)


                        if monitor_training==1:
                            J_err_plus = (np.dot(J,r) - np.dot(Jd,rd) 
                                        -np.dot(w_targ,targ[t:(t+1),:].T) - np.dot(w_hint, hints[t:(t+1),:].T))
                            J_err_ratio = np.hstack((J_err_ratio, np.squeeze(np.mean(J_err_plus/J_err))))
                            J_err_mag = np.hstack((J_err_mag, np.squeeze(np.linalg.norm(J_err))))
                            J_norm = np.hstack((J_norm,np.squeeze(np.linalg.norm(J))))

                            w_err_plus = np.dot(w,r) - targ[t:(t+1),:].T
                            w_err_ratio = np.hstack((w_err_ratio, np.squeeze(w_err_plus/w_err)))
                            w_err_mag = np.hstack((w_err_mag, np.squeeze(np.linalg.norm(w_err))))
                            w_norm = np.hstack((w_norm, np.squeeze(np.linalg.norm(w))))
                            
                            
                            theta_err_plus = z_t*w_targ*np.sign(xi) - (targ[t:(t+1),:].T*w_targ*np.sign(xi))

                            theta_err_ratio = np.hstack((theta_err_ratio, np.squeeze(theta_err_plus/theta_err)))
                            theta_err_mag = np.hstack((theta_err_mag, np.squeeze(np.linalg.norm(theta_err))))
                            theta_norm = np.hstack((theta_norm, np.squeeze(np.linalg.norm(theta))))
            ########## Batch callback
            print('') # New line after each batch

            # Convert lists to arrays
            dx = np.array(dx)
            x = np.array(x)
            z = np.array(z)

            if batch==0:
                # Set up plots
                training_fig = plt.figure()
                ax_unit = training_fig.add_subplot(2,1,1)
                ax_out = training_fig.add_subplot(2,1,2)
                tvec = np.arange(0,len(inp))*p['dt']

                # Create output and target lines
                lines_targ_out = plt.Line2D(tvec, targ, linestyle='--',color='r')
                lines_out = plt.Line2D(tvec, z, color='b')
                ax_out.add_line(lines_targ_out)
                ax_out.add_line(lines_out)

                # Create recurrent unit and DRNN target lines
                lines_targ_unit = {}
                lines_unit = {}
                for i in range(5):
                    lines_targ_unit['%g' % i] = plt.Line2D(tvec, dx[:,i], linestyle='--', color='r')
                    lines_unit['%g' % i]= plt.Line2D(tvec, x[:,i], color='b')
                    ax_unit.add_line(lines_targ_unit['%g' % i])
                    ax_unit.add_line(lines_unit['%g' % i])
                
                # Set up the axes
                ax_out.set_xlim([0,p['dt']*len(inp)])
                ax_unit.set_xlim([0,p['dt']*len(inp)])
                ax_out.set_ylim([-1.2,1.2])
                ax_unit.set_ylim([-2,10])
                ax_out.set_title('Output')
                ax_unit.set_title('Recurrent units, batch %g' % (batch+1))

                # Labels
                ax_out.set_xlabel('Time (s)')
                ax_out.legend([lines_targ_out,lines_out],['Target','RNN'], loc=1)
            else:
                # Update the plot
                tvec = np.arange(0,len(inp))*p['dt']
                ax_out.set_xlim([0,p['dt']*len(inp)])
                ax_unit.set_xlim([0,p['dt']*len(inp)])
                ax_unit.set_title('Recurrent units, batch %g' % (batch+1))
                lines_targ_out.set_xdata(tvec)
                lines_targ_out.set_ydata(targ)
                lines_out.set_xdata(tvec)
                lines_out.set_ydata(z)
                np.savetxt('targunitz.csv',targ)
                np.savetxt('outunitz.csv',z)

                for i in range(5):
                    lines_targ_unit['%g' % i].set_xdata(tvec)
                    lines_targ_unit['%g' % i].set_ydata(dx[:,i])
                    lines_unit['%g' % i].set_xdata(tvec)
                    lines_unit['%g' % i].set_ydata(x[:,i])
            training_fig.canvas.draw()
                


        if monitor_training==1:
        # Now for some visualization to see how things went:
            stats_fig = plt.figure(figsize=(8,10))
            plt.subplot(3,2,1)
            plt.title('Recurrent learning error ratio')
            plt.plot(J_err_ratio)

            plt.subplot(3,2,3)
            plt.title('Recurrent error magnitude')
            plt.plot(J_err_mag)        

            plt.subplot(3,2,5)
            plt.title('Recurrent weights norm')
            plt.plot(J_norm)

            plt.subplot(3,2,2)
            plt.plot(w_err_ratio)
            plt.title('Output learning error ratio')

            plt.subplot(3,2,4)
            plt.plot(w_err_mag)
            plt.title('Output error magnitude')

            plt.subplot(3,2,6)
            plt.plot(w_norm)
            plt.title('Output weights norm')
            stats_fig.canvas.draw()
            
            plt.subplot(3,2,2)
            plt.plot(theta_err_ratio)
            plt.title('Theshold learning error ratio')

            plt.subplot(3,2,4)
            plt.plot(theta_err_mag)
            plt.title('Theshold error magnitude')

            plt.subplot(3,2,6)
            plt.plot(theta_norm)
            plt.title('Threshold weights norm')
            stats_fig.canvas.draw()

        print('Done training!')

    
    def test(self, inps_and_targs, **kwargs):
        p = self.p
       

        self.initialize_act()
        print('Initializing',end="")
        for i in range(p['test_init_trials']):
            print('.',end="")
            inp, targ = inps_and_targs(dt=p['dt'], **kwargs)[0:2]
            self.run(inp)
        print('')

        inp, targ = inps_and_targs(dt=p['dt'], **kwargs)[0:2]
        test_fig = plt.figure()
        ax = test_fig.add_subplot(1,1,1)
        tvec = np.arange(0,len(inp))*p['dt']
        line_inp = plt.Line2D(tvec, targ, linestyle='--', color='g')
        line_targ = plt.Line2D(tvec, targ, linestyle='--', color='r')
        line_out = plt.Line2D(tvec, targ, color='b')
        print('tvecshape',tvec.shape)
        print('targshape',targ.shape)
        np.savetxt('targunitz.csv',line_targ)
        np.savetxt('outunitz.csv',line_out)

        
        ax.add_line(line_inp)
        ax.add_line(line_targ)
        ax.add_line(line_out)
        ax.legend([line_inp, line_targ, line_out], ['Input','Target','Output'], loc=1)
        ax.set_title('RNN Testing: Wait')
        ax.set_xlim([0,p['dt']*len(inp)])
        ax.set_ylim([-1.2,1.2])
        ax.set_xlabel('Time (s)')
        test_fig.canvas.draw()

        E_out = 0 # Running squared error
        V_targ = 0  # Running variance of target
        print('Testing: %g trials' % p['test_trials'])
        for idx in range(p['test_trials']):
            print('.',end="")
            inp, targ = inps_and_targs(dt=p['dt'], **kwargs)[0:2]

            tvec = np.arange(0,len(inp))*p['dt']
            ax.set_xlim([0,p['dt']*len(inp)])
            line_inp.set_xdata(tvec)
            line_inp.set_ydata(inp)
            line_targ.set_xdata(tvec)
            line_targ.set_ydata(targ)
            out = self.run(inp)[0]
            line_out.set_xdata(tvec)
            line_out.set_ydata(out)
            ax.set_title('RNN Testing, trial %g' % (idx+1))
            test_fig.canvas.draw()
            
            E_out = E_out + np.dot(np.transpose(out-targ), out-targ)
            V_targ = V_targ + np.dot(np.transpose(targ), targ)
        print('')
        E_norm = E_out/V_targ
        print('Normalized error: %g' % E_norm)
        return E_norm









