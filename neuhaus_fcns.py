#These functions allow quick manipulation and visualization of the Sarah Neuhaus radiocarbon and TOC
#model (Neuhaus et al., 2021, The Cryosphere). 

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
import math
import numpy as np

#Constants from paper:
tau = 8033              #mean lifetime of 14C based off of the Libby half life.
modern_ratio = 1.176810*10**(-12)
                        #Modern ratio of 14C/12C to transform to fraction modern - from the Neuhaus et al. text. 

DEBUG = False

def debug(*args):
    if DEBUG == True:
        print(args)



def model_retreat(t_earliest, t_readvance, a, A, N_t0=0.28):
    '''
    Iteration of the model for time period that SLM is exposed to the ocean, t_earliest to T_readvance

        Input variables:
            t_earliest (float): the calendar year before present that ice could have retreated in the Holocene. In
                Neuhaus et al., 2021, the authors use 8000 y for this input, citing Lee et al., 2017; McKay et 
                al., 2016; Spector et al., 2017
            t_readvance (float): the calendar year before present that ice readvanced over the point of interest. In
                Neuhaus et al., 2021, the authors use 4000 years. Venturelli et al. (2023) measure 6200 years for
                this timing. 
            a (float): accumulation rate of 14C in sediment, authors used 9.23*10**(-18) g/yr/100g sediment
            A (float): accumulation rate of 12C in sediment, authors used 9*10**(-6) g/yr/100g sediment
            N_0 (float, default=0.28): Non-zero initial condition of %TOC (g/100g sediment)

        Output variables: 
            fM_0 (list): the fractions modern calculated from dividing 14C by 12C and multiplying by the modern ratio
                (see constants above).
            N_0 (list): the amount of 12C deposited at each time step
            n_0 (list): the amount of 14C deposited at each time step
            t0_steps (list): the time steps of the retreat model

            
            
    '''
    
    t0_steps = np.linspace(t_earliest-t_earliest, t_earliest-t_readvance, 100)
    #Make A a list so that the function handles variable accumulation rates:
    if type(A) is not list:
        A = [A]*len(t0_steps)
        debug('A was a scalar. Changed to a vector of length of t0_steps.')
    debug('What type is A now? : ', type(A), 'Should always be a list')

    n_0 = [a*tau*(1-math.exp(-t/tau)) for t in t0_steps]
    N_0 = [N_t0 + acc*t for (acc, t) in zip(A, t0_steps)]

    fM_0 = [(C14/C12)/modern_ratio for (C14, C12) in zip(n_0, N_0)]
    

    return fM_0, N_0, n_0, t0_steps

def model_readvance(t_earliest, t_readvance, n_0, N_0, t0_steps):
    '''
    Iteration of the model for time period that SLM is exposed to the ocean, t_earliest to T_readvance

        Input variables:
            t_earliest (float): the calendar year before present that ice could have retreated in the Holocene. In
                Neuhaus et al., 2021, the authors use 8000 y for this input, citing Lee et al., 2017; McKay et 
                al., 2016; Spector et al., 2017
            t_readvance (float): the calendar year before present that ice readvanced over the point of interest. In
                Neuhaus et al., 2021, the authors use 4000 years. Venturelli et al. (2023) measure 6200 years for
                this timing. 
            t0_steps (list/array): The time steps from the model_retreat function
            n_0 (list): The amount of 14C deposition output from model_retreat function
            N_0 (list): The amount of 12C deposition output from the model_retreat function.
        
        Output variables:
            fM_1 (list): the fractions modern calculated from dividing 14C by 12C and multiplying by the modern ratio
                (see constants above).
            N_1 (list): the amount of 12C deposited at each time step
            n_1 (list): the amount of 14C deposited at each time step
            t1_steps (list): the time steps of the retreat model

    '''


    t1_steps = np.linspace(t_earliest-t_readvance, t_earliest, 100)

    n_1 = [n_0[-1]*math.exp(-(t-max(t0_steps))/tau) for t in t1_steps]
    N_1 = [N_0[-1] for t in t1_steps]
    

    fM_1 = [(C14/C12)/modern_ratio for (C14, C12) in zip(n_1, N_1)]
    
    

    return fM_1, N_1, n_1, t1_steps

def model_coupler(t_earliest, t_readvance, a, A):
    '''
    Couple the retreat and readvance models

        Input variables:
            t_earliest (float): the calendar year before present that ice could have retreated in the Holocene. In
                Neuhaus et al., 2021, the authors use 8000 y for this input, citing Lee et al., 2017; McKay et 
                al., 2016; Spector et al., 2017
            t_readvance (float): the calendar year before present that ice readvanced over the point of interest. In
                Neuhaus et al., 2021, the authors use 4000 years. Venturelli et al. (2023) measure 6200 years for
                this timing. 
            a (float): accumulation rate of 14C in sediment, authors used 9.23*10**(-18) g/yr/100g sediment
            A (float): accumulation rate of 12C in sediment, authors used 9*10**(-6) g/yr/100g sediment

        Output variables:
            retreat_output_dict (dictionary): timesteps, 12C, 14C, and fM of the retreat model ouptuts
            readvance_output_dict (dictionary): timesteps, 12C, 14C, and fM of the readvance model outputs

    
    '''
    fM_0, N_0, n_0, t0_steps = model_retreat(t_earliest, t_readvance, a, A)

    fM_1, N_1, n_1, t1_steps = model_readvance(t_earliest, t_readvance, n_0, N_0, t0_steps)

    
    debug('14C Test - should equal 0:', n_0[-1]-n_1[0], 'Actual value: ', n_0[-1], '-', n_1[0])
    debug('%TOC Test - should equal 0: ', N_0[-1]-N_1[0], 'Actual value: ', N_0[-1], '-', N_1[0])
    debug('fM Test - should equal 0: ', fM_0[-1]-fM_1[0])
    debug('Length of list test - Length of A input: ', len([A]), 'Length of fM_0 retreat output: ', len(fM_0))

    retreat_output_dict = {'time_steps':t0_steps, 'C12':N_0, 'C14':n_0, 'fM':fM_0}
    readvance_output_dict = {'time_steps':t1_steps, 'C12':N_1, 'C14':n_1, 'fM':fM_1}



    return retreat_output_dict, readvance_output_dict

def fig_4_emulator(retreat_out_dict, readvance_out_dict, colors=['tomato', 'cornflowerblue']):
    '''
    This function emulates the type of plot shown in Figure 4 of Neuhaus et al., 2021. This plot contains two panels,
    one with the retreated ice showing years since retreat on the x-axis, and a double y axis with fM on one and %TOC
    on the other. The original plot was blue (fM) and red (%TOC).

        Input variables:
            retreat_out_dict (dictionary): Output from the model_coupler function, or similar structure (dictionary
                keys of time_steps, C12, C14, fM)
            readvance_out_dict (dictionary): Output from the model_coupler function, or similar structure (dictionary
                keys of time_steps, C12, C14, fM)
            colors (list of strings corresponding to pyplot colors): list of colors. First in list is the %TOC, second
                is the fraction modern. Only first two colors from the list will be used. 
        
        Output variables;
            fig (pyplot figure object): Figure on which the axes are plotted
            ax (pyplot axes object): n=2 array of axes. ax[0] is the plot of t0 (retreated ice) and ax[1] is the plot
                of t1 (readvanced ice)
            ax0 (pyplot axes object): axis on which the right side y-axis is plotted for fM in the t0 (retreated ice)
                subplot
            ax1 (pyplot axes object): axis on which the right side y-axis is plotted for fM in the t1 (readvanced ice)
                subplot

    '''

    #Assign variables from dictionaries:
    t0_steps = retreat_out_dict['time_steps']
    N_0 = retreat_out_dict['C12']
    fM_0 = retreat_out_dict['fM']
    t1_steps = readvance_out_dict['time_steps']
    N_1 = readvance_out_dict['C12']
    fM_1 = readvance_out_dict['fM']
    
    #Plot results

    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True)
    fig.set_facecolor('white')

    color = colors[0]
    ax[0].set_xlabel('time (y since retreat)')
    ax[0].set_ylabel('%TOC', color=color)
    ax[0].set_title('T$_0$')
    ax[0].plot(t0_steps, N_0, color='tomato')
    ax[0].tick_params(axis='y', labelcolor=color)
    ax0 = ax[0].twinx()

    #plot fraction modern on different y-axis, same subplot
    color = colors[1]
    ax0.set_ylabel('', color=color)  # we already handled the x-label with ax1
    ax0.plot(t0_steps, fM_0, color=color)
    ax0.tick_params(axis='y', labelcolor=color)


    #Plot Phase 2 in second subplot - ice sheet has readvanced

    color = colors[0]
    ax[1].set_xlabel('time (y since retreat)')
    ax[1].set_ylabel('', color=color)
    ax[1].set_title('T$_1$')
    ax[1].plot(t1_steps, N_1, color='tomato')
    ax[1].tick_params(axis='y', labelcolor=color)
    ax1 = ax[1].twinx()

    #plot fraction modern on different y-axis, same subplot
    color=colors[1]
    ax1.set_ylabel('fraction modern', color=color) 
    ax1.plot(t1_steps, fM_1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.sharey(ax0)
    #ax0.yaxis.set_ticklabels([])
    #ax0.set_yticks([])

    return fig, ax, ax0, ax1



def sensitivity_test_plotter(model_out_dict, iter_var, colormap='copper'):
    '''
    Plots outputs of sensitivity tests of the Neuhaus et al 2021 model for radiocarbon accumulation. 

        Input variables:
            model_out_dict (dictionary of dictionaries): This is the output of a loop through different variables in the 
                model_coupler function. That function returns a dictionary of dictionaries, top level is retreat or advance
                as dictionary keys, and each of those dictionaries has the keys of time_steps,  C12, C14, and fM. This function
                sutures both output dictionaries effectively into one subplot for both TOC and fM to show a continuous time
                domain.
            colormap (string): string referring to a pyplot colormap. Default is copper, and the function reverts to default if
                an unrecognized colormap is entered. 
            iter_var (list of floats): This is generally the variable that was varied and looped through to create the model_
                _out_dict. 

        Output variables:
            axes (array of pyplot axes): array of axes to allow for additional formatting. 
    '''
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)

    #Get the colormap:
    colors = plt.get_cmap(colormap)
    #Normalize the colormap to the length of the test variable:
    vmin, vmax = min(iter_var), max(iter_var)
    normalizer = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    for ind, (k, v) in enumerate(model_out_dict.items()):
        debug('Plot iteration number: ', ind)
        col=colors(normalizer(iter_var[ind]))
        ax[0].plot(v['retreat']['time_steps'], v['retreat']['C12'], color=col, label=k) #Add T0 plot of TOC
        ax[1].plot(v['retreat']['time_steps'], v['retreat']['fM'], color=col, label=k)  #Add T0 plot of TOC
        ax[0].plot(v['readvance']['time_steps'], v['readvance']['C12'], color=col) #Add T1 plot of TOC
        ax[1].plot(v['readvance']['time_steps'], v['readvance']['fM'], color=col)  #Add T1 plot of TOC

    #Update graph labels and limits:
    top_lims = ax[0].get_ylim()
    bot_lims = ax[1].get_ylim()
    ax[0].set(ylabel='%TOC', title='%TOC')
    ax[1].set(ylabel='fM', xlabel='years since initial retreat', title='fraction modern')
    ax[0].legend()

    #Add vertical lines representing the time at which the grounding line crossed the site during readvance:
    for ind, (k, v) in enumerate(model_out_dict.items()):
        debug('Vert line iteration number: ', ind)
        col=colors(normalizer(iter_var[ind]))
        ax[0].vlines(v['retreat']['time_steps'][-1], min(top_lims), max(top_lims), color=col, linestyle='-.')
        ax[1].vlines(v['retreat']['time_steps'][-1], min(bot_lims), max(bot_lims), color=col, linestyle='-.')

def precision_round(number, digits=3):
    '''
    Rounds scientific notation numbers to number of digits. Courtesy of Josh Duran, Stack Overflow, 16 Dec 2020
    '''
    power = "{:e}".format(number).split('e')[1]
    return round(number, -(int(power) - digits))

def convert_reservoir_to_raw_n(R):
    '''
    Converts from reservoir ages (something we all understand) to the raw numbers of 14C nuclides per 100 g of sediment
    required to run the Neuhaus et al model.

        Input variables:
            R (float or list of floats): postive integers representing the inherent of the modern ocean at the time of
                formation of a CaCO3 test or organic molecule from primary production. Lists can be accepted. 
        Ouput variables:
            n (float or list of floats): numbers on the magnitude of 10^(-18) which represent the number of 14C nuclides
                in 100 g of sediment. These numbers are necessary to run the Neuhaus model
    '''
    #Check for list:
    if type(R) is not list:
        print('Not a list - Variable R is a ', type(R), 'converting to a list.')
        if isinstance(R, int) and isinstance(R, float):
            print('Variable R is a ', type(R))
            R = [R]     #Makes it a list of one
        elif isinstance(R, np.ndarray):
            print('Variable is an ', type(R), '. Changing to list.')
            R = R.tolist()
        else:
            print("Must enter a float, integer, or a list of floats or integers. You entered a: ", type(R))
            return
        
    new_a = []
    A = 9*10**(-6)          #Accumulation rate of 12C, g/yr/100g sediment (from base conditions in paper)

    for ind, reservoir_age in enumerate(R):
        print(ind, reservoir_age)
        #Convert to fraction modern:
        F = math.exp(-reservoir_age/8033)
        #F is equal to the ratio of fluxes a/A, so we can calculate an a value for each F:
        new_a.append(F*modern_ratio*A)

    return new_a

def Neuhaus_analytical_solution(T_0, T_1, time_step, L, depth_step, D, C_i, C_0):
    '''
    This provides the analytical solution for the phase 1 model of Neuhaus et al., 2020. The Phase 1 model assumes a straight vertical profile of 
    meltwater solute concentration or conductivity (C_0) due to isolation for tens of thousands (at least) years. There is an instantaneous change in
    boundary conditions at t = 0 to ocean water (C_1), and the model is solved for different amounts of time up to T_0. These different profiles 
    fill an output array which has a number of columns equal to the number of time slices and a number of rows equal to the depth resolution; each
    column is a verticle porewater profile.The solutions (profiles) can be uniquely calculated for a given number of years that the sediment was
    overlain by seawater (grounding line retreated landward of site).

        Input variables:
            T_0 (float): Number of years before present that grounding line retreated past the position of the data
                opening up the site to seawater at the sediment-water boundary
            T_1 (float): Number of years before present that the grounding line advanced pas the position of the site
                from which the porewater data come, closing the site off from seawater.
            time_step (integer): Number of years between each calculation of the model
            L (integer): Number of centimeters of depth over which to perform calculations. This should be considerably
                deeper than the depth of the deepest data point to avoid boundary condition issues.
            D (float): diffusivity coefficient, cm^2/y
            C_i (float): Initial concentration of the solute prior to initial grounding line retreat, equivalent to 
                100% subglacial meltwater concentrations if known. Englacial concentrations can be used as well
            C_0 (float): Concentration of solute in seawater

        Output variables:
            z (list, floats): depths at which calcuations were carried out
            phase1_arr (array of floats): n x m array of solute concentration data calculated at n depths 
                along L and m time slices betweeen T_0 and T_1
            **Both of these output variables are necessary to use the plotting functions of the outputs


    '''
    #Make z axis (depth from 0 to L)
    Nx = L//depth_step                      # number of spatial points
    z = np.linspace(0, L, Nx)
    duration = T_0 - T_1
    #Make empty arrays of zeroes to receive the output
    conc_profiles_phase1 = np.zeros((duration//time_step, Nx))
    print('Shape of c_profiles_p1 should be ', str(T_0), '-', str(T_1), ' divided by the time step: ', conc_profiles_phase1.shape)
    init_cond = np.ones(Nx)*C_0      # initial condition is set as a straight profile at the value of C_0
    conc_profiles_phase1[0, :] = init_cond      #install initial condition in what is to become the output array

    for j, time in enumerate(range(1, conc_profiles_phase1.shape[0])):      #For every time slice except the first one (column, initial boundary condition)
        #print(j)
        for k, z in enumerate(range(0, L, depth_step)):                     #For every depth in the working time slice (works through rows)
            #print('Time:Depth:j:d = ', time, ':', c, ':', j, ':', d)
            conc_profiles_phase1[j+1, k] = math.erfc(z/(2*math.sqrt(D*time)))*(C_i-C_0)+C_0

    phase1_arr = conc_profiles_phase1.copy()
    return z, phase1_arr

def Neuhaus_PW_solute_model(boundary_condition_profile_analytical, D, T_1, C_0, L, depth_step, time_step, u=None, check_handoff=False):
    ''' 
    This solves the advection-diffusion model of phase two of the Neuhaus et al 2020 paper. It uses the output of the analytical solution from 
    seawater as the boundary condition. Users can specify to use advection by setting an advection value, or they 
    use diffusion only by setting the value to None. (kwarg u). The model iterates over each time step constructing
    a profile of the solute over the entire depth of the model, L. 
        Input variables:
            boundary_conditions_profile_analytical (array of floats): The output of the analytical solution, phas31_arr,
                or similarly formated boundary conditions for the intial state of the advection diffusion model.
                This routine replaces the uppermost value of the last sime slice of phase1_arr with the sub-
                glacial melt value of the solute concentration to initiate the model.
            D (float): diffusivity coefficient, cm^2/y
            T_1 (float): Number of years before present that the grounding line advanced pas the position of the site
                from which the porewater data come, closing the site off from seawater.
            C_0 (float): Concentration of solute in seawater
            L (integer): Number of centimeters of depth over which to perform calculations. This should be considerably
                deeper than the depth of the deepest data point to avoid boundary condition issues.
            depth_step (integer, cm): The distance between calculations of solute concentrations, must be 
                smaller than L.
            time_step (integer): Number of years between each calculation of the model
            u (float, kwarg, default=None): advection velocity, cm/y. Dafault value is None; this elicits the code
                to run without the advection term (diffusion only model)
            check_handoff (boolean, default=False): displays the boundary condition profile at the end of Phase
                1, after the change in initial conditions (the handoff to Phase 2), and at the first iteration
                of the diffusion (or advection and diffusion) model 

        Output variables:
            z (list, floats): depths at which calcuations were carried out
            phase1_arr (array of floats): n x m array of solute concentration data calculated at n depths 
                along L and m time slices betweeen T_0 and T_1
            tag (string): Description of whether advection was used or not, used in saving graphics so that 
                the end user can obtain a record of whether the model was diffusion only or not.
            **The first two of these output variables are necessary to use the plotting functions of the outputs



    '''
    # Initialize concentraton array with output from phase 1:
    
    boundary_cond_p2 = boundary_condition_profile_analytical[-1, :]      # The last time slice from Phase 1 corresponding to last year of open seawater contact
    boundary_cond_p2[0] = C_0

    Nx = L//depth_step                      # number of spatial points
    Nt = T_1//time_step                  # number of time steps, phase 2
    dx = L/Nx
    dt = T_1/Nt

    # Create an array to store concentrations at each time step
    readvance_cs = np.zeros((Nt, Nx))
    readvance_cs[0, :] = boundary_cond_p2 # initial condition, from phase 1
    
    # Time step using finite differences
    if u == None:
        print('Diffusion only model will be used. Specify kwarg u to add advection.')
        tag = 'Diffusion_only'
    else:
        print('Diffusion and advection being employed.')
        tag = 'Advection_diffusion'
        peclet = u*200/D
        print('Peclet number: ', 0.15*200/D)


    p = boundary_cond_p2.copy()
    for t in range(1, Nt):
        for j in range(1, Nx - 1):
            if u == None:
                p[j] = p[j] + D *dt/dx**2*(p[j+1] - 2*p[j] +p[j-1])     #delete the advection term
            else:
                p[j] = p[j] + u*dt/dx*(p[j-1] - p[j]) + D *dt/dx**2*(p[j+1] - 2*p[j] +p[j-1])     #use advection term
                
            
        readvance_cs[t,:] = p.copy()
        phase2_arr = readvance_cs.copy()
    
    z = np.linspace(0, L, Nx)
    if check_handoff == True:
        handoff_test(boundary_condition_profile_analytical, boundary_cond_p2, readvance_cs, z)
    
    return z, phase2_arr, tag
    

def visualize_phase1(phase1_arr, z, T_0, time_step, cmap='copper'):
    '''
    Plots curves from the array of Phase 1 profiles, the initial condition, the final condition (initial condition 
    for Phase 2), and 3 from the middle. 

        Input variables:
            phase1_arr (array, floats): The output of the analytical solution function, Phase 1 of the Neuhaus
                et al. model
            z (array or list, floats): The depths (cm) at which calculations were made
            T_0 (float): The time at which the grounding line first retreated past the position from which the
                data were taken, exposing the site to salt water
            time_step (integer): the number of years between each calculation of profiles
            cmap (matplotlib array of colors): This is the colormap that you want to use to plot mutliple curves

        Output variables:
            fig (figure handle): the figure handle of the graphic
            ax (ases handle): the axes handle on which the data are plotted


    '''

    fig, ax = plt.subplots(nrows=1, ncols=1)

    #xes = np.linspace(0, max_depth, max_depth//depth_step)
    cols = [0, len(phase1_arr[:,0])//3, len(phase1_arr[:, 0])//3*2, -1]
    colormap = cm.get_cmap(cmap, len(cols))
    for ind, t in enumerate(cols):
        if t==0:
            name = 'Initial Cond.'
        elif t==-1:
            name = 'Final Cond.'
        else:
            name = str(T_0 - t*time_step)+' ybp'
        ax.plot(phase1_arr[t, :], z, c=colormap(ind), label=name)
    
    ax.yaxis.set_inverted(True)
    ax.set(
        title='Phase 1, '+str(T_0) + '-' + str(T_0-len(phase1_arr[:,0])*time_step)+'ybp',
        ylabel='depth (cm)',
        xlabel='Conductivity ($\mu S cm^{-1} @ 25^°C$)'
    )
    plt.legend()

    return fig, ax

def visualize_phase2(phase_2_profiles, pw_df, z, slice=5, cmap='copper'):

    '''
    Plots curves from the array of Phase 2 profiles, the advection-diffusion (or diffusion only) model
    condition, with the final condition in bold. 

        Input variables:
            phase_2_profiles (array, floats): The output of the solute transport moel, Phase 2 of the Neuhaus
                et al. model
            pw_df (DataFrame): The solute concentration measurements (data)
            z (array or list, floats): The depths (cm) at which calculations were made
            slice (integer, default=5): The number of profiles to plot

        Output variables:
            fig (figure handle): the figure handle of the graphic
            ax (ases handle): the axes handle on which the data are plotted
    
    
    '''


    fig, ax = plt.subplots(nrows=1, ncols=2)
    cond = [c for c in pw_df.loc[:, ('Conductivity', '(µS cm-1 @ 25°C)')]]
    depth = [d for d in pw_df.loc[:, ('Composite Depth (median)', 'cm')]]

    Nt = phase_2_profiles.shape[0]
    print('Nt = ', Nt)
    colormap = cm.get_cmap(cmap, slice+2)
    ctr=1
    #Plot Phase 2 only:
    for t in range(0, Nt, Nt//slice):
        # plot every "Nt//slice" time step from model
        print('Time step: ', str(Nt-t))
        ax[0].plot(phase_2_profiles[t, :], z, c=colormap(ctr), label=f'{Nt-t}'+'ybp')
        ctr = ctr + 1
    #Add observations:
    ax[0].plot(cond, depth, linestyle='', marker='^', markersize=10, mec='k', color='tomato')
    #Add final time step from model:
    ax[0].plot(phase_2_profiles[-1, :], z, color='k', linewidth=3, label='Final step')
    ax[0].set(ylabel='depth (cm)', title='Ph. 2 A-D '+str(Nt)+'-0 ybp')
    ax[0].yaxis.set_inverted(True)
    ax[0].legend() 

    #Plot Phase 2, zoomed with data and only last iteration of the model:
    ax[1].plot(cond, depth, linestyle='', marker='^', markersize=10, mec='k', color='tomato')
    ax[1].plot(phase_2_profiles[-1, :], z, color='k', linewidth=2)
    ax[1].set(title='Mod vs. Data', ylim=[0, 250])
    ax[1].yaxis.set_inverted(True)

    fig.supxlabel('Conductivity ($\mu S cm^{-1} @ 25^°C$)')

    return fig, ax

def visualize_allphases(c_profiles_p1, phase_2_profiles, pw_df, T_0, T_1, z, slice=5, cmap='copper'):
    '''
    Plots curves from the array of Phase 1 profiles (retreat), the Phase 2 profiles (advance), and the data.
    In each of the first two panels, the initial conditions and the final conditions are differentiated
    in the legend. 

        Input variables:
            c_profiles_p1 (array, floats): The output of the analytical solution function, Phase 1 of the Neuhaus
                et al. model
            phase_2_profiles (array, floats): The output of the solute transport model
            pw_df (DataFrame): The solute concentration measurements (data)
            T_0 (float): The time at which the grounding line first retreated past the position from which the
                data were taken, exposing the site to salt water
            T_1 (float): Number of years before present that the grounding line advanced pas the position of the site
                from which the porewater data come, closing the site off from seawater.
            z (array or list, floats): The depths (cm) at which calculations were made

            
            
        Output variables:
            fig (figure handle): the figure handle of the graphic
            ax (ases handle): the axes handle on which the data are plotted
            slice (integer, default=5): The number of profiles to plot
    
    '''


    fig, ax = plt.subplots(nrows=1, ncols=3)
    cond = [c for c in pw_df.loc[:, ('Conductivity', '(µS cm-1 @ 25°C)')]]
    depth = [d for d in pw_df.loc[:, ('Composite Depth (median)', 'cm')]]

    #Plot Phase 1:
    Nt_0 = c_profiles_p1.shape[0]
    #Activate color map, set counter at 1 so that the colormap does not include black. 
    colormap = cm.get_cmap(cmap, slice+2)
    ctr=1
    for t in range(0, Nt_0, Nt_0//slice):
        # plot every "Nt//slice" time step from model
        ax[0].plot(c_profiles_p1[t, :], z, c=colormap(ctr), label=f'{T_0-t}'+'ybp')
        ctr = ctr + 1
    #Add observations:
    ax[0].plot(cond, depth, linestyle='', marker='^', markersize=10, mec='k', color='tomato')
    #Add final time step from model:
    ax[0].plot(c_profiles_p1[-1, :], z, color='k', linewidth=2, label='Ph. 2 ' \
    'bound')
    ax[0].set(ylabel='depth (cm)', title='Ph. 1: Retreat')
    ax[0].yaxis.set_inverted(True)
    ax[0].legend() 

    #Plot Phase 2:
    Nt_1 = phase_2_profiles.shape[0]
    ctr = 1     #Reset Counter...
    for t in range(0, Nt_1, Nt_1//slice):
        # plot every "Nt//slice" time step from model
        ax[1].plot(phase_2_profiles[t, :], z, c=colormap(ctr), label=f'{T_1-t}'+'ybp')
        ctr = ctr + 1
    #Add observations:
    ax[1].plot(cond, depth, linestyle='', marker='^', markersize=10, mec='k', color='tomato')
    #Add final time step from model:
    ax[1].plot(phase_2_profiles[-1, :], z, color='k', linewidth=3, label='Present')
    ax[1].set(ylabel='depth (cm)', title='Ph. 2: Readvance')
    ax[1].yaxis.set_inverted(True)
    ax[1].legend() 

    #Plot Phase 2, zoomed in with data
    ax[2].plot(cond, depth, linestyle='', marker='^', markersize=10, mec='k', color='tomato')
    ax[2].plot(phase_2_profiles[-1, :], z, color='k', linewidth=2)
    ax[2].set(title='Mod vs. Data', ylim=[0, 250])
    ax[2].yaxis.set_inverted(True)

    fig.supxlabel('Conductivity ($\mu S cm^{-1} @ 25^°C$)')

    return fig, ax

def handoff_test(boundary_condition_profile_analytical, boundary_cond_p2, readvance_cs, z):
    '''
    Designed to work within the Neuhaus Solute Model function. Shows the length of the final time slice of the numerical solution that is
    an argument of the model, the first column of the output array, and  the final output of the model's initial condition. These 
    should be the same. A plot is made with the final time slice of the input array and the first time slice of the output array. Again, 
    these should be the same. 

    '''

    print('Length and first 10 elements of final time slice of input array (analytical solution): ', len(boundary_condition_profile_analytical[:, -1]), boundary_condition_profile_analytical[-1, 0:10])
    print('Length and first 10 elements of hand-off to solute transport model output: ', len(boundary_cond_p2), boundary_cond_p2[0:10])
    print('Length and first 10 elements of the first time slice of the solute transport model output, after iterative loop: ', len(readvance_cs[:, 0]), readvance_cs[0, 0:10])

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(boundary_condition_profile_analytical[-1, :], z, marker='o', markersize=15, linestyle='', mfc='lemonchiffon', mec='k', label='Input profile')
    ax.plot(boundary_cond_p2, z, marker=None, linestyle='-', linewidth=5, color='w', label='Hand-off profile')
    ax.plot(readvance_cs[0, :], z, marker=None, linestyle='--', linewidth=1, color='k', label='Initial condition output')
    ax.set(title='Hand-off Test - All plots should be same!', xlabel='Conductivity ($\mu S cm^{-1} @ 25^°C$)', ylabel='Depth, cm')
    plt.legend()
    ax.yaxis.set_inverted(True)

    return fig, ax




