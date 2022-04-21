#!/usr/bin/env python
# coding: utf-8
# %%

# # Linear Audio Visual Reaction Time Simulation
# This notebook use the decision making model by Wang and colleagues to simulate linear audio-visual reaction time task.
# 

# %%


#from platform import python_version

#print(python_version())


# ## Material and Methods

# ### Calling Library Fuctions

# %%


# LIBRARY
## DDM LIBRARY
import ddm.plot
from ddm import Model, Fittable
from ddm.models import DriftConstant, NoiseConstant, BoundConstant, OverlayChain,OverlayNonDecision,OverlayPoissonMixture,BoundCollapsingExponential
from ddm.functions import fit_adjust_model, display_model
from ddm import Sample
from ddm.plot import model_gui
from ddm.models import LossRobustBIC
from ddm.functions import fit_adjust_model, display_model


## LIBRARIES
import numpy as np # vector manipulation
import math  # math functions
import sys
import pandas as pd
from scipy.stats import ttest_ind
# THIS IS FOR PLOTTING
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt # side-stepping mpl backend
import warnings
warnings.filterwarnings("ignore")

# get ANOVA table as R like output
import statsmodels.api as sm
from statsmodels.formula.api import ols

import ptitprince as pt
import seaborn as sns


# ## The Reduced Network Model
# 
# The firing rate function, the input-output, 
# $$ H(x)=\frac{ax-b}{1-e^{-d(ax-b)}},$$
# a=270, b=108 and d=0.154.

# %%


def H(x):
    a=270 # Hz/nA
    b=108 # Hz
    d=.154 # seconds
    f=(a*x-b)/(1-np.exp(-d*(a*x-b)))
    return f
x=np.arange(-1,1,0.01)


# ## Neural Circuit
# 
# ### Unisensory version
# $$ x_{1}=J_{11}S_1-J_{12}S_2+I_{0}+I_{1}+I_{noise,1}$$
# $$ x_{2}=J_{22}S_2-J_{21}S_1+I_{0}+I_{2}+I_{noise,1}$$
# 
# where the synaptic couplings are $J_{11}=0.2609$, $J_{22}=0.2609$, $J_{12}=0.0497$ and $J_{21}=0.0497$.
# $I_{0}=0.3255 nA$ represents external input and $S_1$ and $S_2$ are the auditory activity $A_1$ and $A_2$ or visual activity $V_1$ and $V_2$.
# Where 1 is HIT and 2 in MISS
# 
# 
# ### Multisensory version
# TO BE TESTED?
# $$ x_{HIT}=J_{11}(A_1+V_1)-J_{12}(A_2+V_2)+I_{0}+I_{1}+I_{noise,1}$$
# $$ x_{MISS}=J_{22}(A_2+V_2)-J_{21}(A_1+V_1)+I_{0}+I_{2}+I_{noise,1}$$
# 
# 

# %%


def total_synaptic_current(S_1,S_2,I_1,I_2,I_noise_1,I_noise_2):
    # Synaptic coupling
    J_11=0.2609 # nA
    J_22=0.2609 # nA
    J_13=0*0.000497/np.sqrt(2) # nA
    J_24=0*0.000497/np.sqrt(2) # nA
  
    J_12=0.0497 # nA
    J_21=0.0497 # nA
    I_0=0.3255  # nA
    x_1=J_11*S_1-J_12*S_2+I_0+I_1+I_noise_1
    x_2=J_22*S_2-J_21*S_1+I_0+I_2+I_noise_2
    return x_1, x_2


def MULTISENSORY_total_synaptic_current(A_1,A_2,V_1,V_2,I_1,I_2,I_noise_1,I_noise_2):
    # Synaptic coupling
    J_11=0.2609 # nA
    J_22=0.2609 # nA
    J_13=0*0.000497/np.sqrt(2) # nA
    J_24=0*0.000497/np.sqrt(2) # nA
  
    J_12=0.0497 # nA
    J_21=0.0497 # nA
    I_0=0.3255  # nA
    x_1=J_11*(A_1+V_1)-J_12*(A_2+V_2)+I_0+I_1+I_noise_1
    x_2=J_22*(A_2+V_2)-J_21*(A_1+V_1)+I_0+I_2+I_noise_2
    return x_1, x_2


# ### Background activity
# $$ \tau_{AMPA}\frac{d I_{noise,i}(t)}{dt} =-I_{noise,i}(t)+\eta_i(t)\sqrt{\tau_{AMPA}}\sigma_{noise}$$

# %%


def Background_Activity(I_noise):
    h=0.1
    sigma_noise=0.02 # nA
    tau_AMPA=2 #ms
    eta_noise=np.random.normal(0,1,1)
    k=0#(-(I_noise)+eta_noise*np.sqrt(tau_AMPA)*sigma_noise)
    I_noise_new=I_noise+h/tau_AMPA*(-(I_noise+h/2*k)+eta_noise
                                *np.sqrt(tau_AMPA)*sigma_noise)
    return I_noise_new


# ### Network Dynamics
# $$ \frac{d S_{i}}{dt} =-\frac{S_{i}}{\tau_S}+(1-S_{i})\gamma H_{i}$$

# %%


def Network_Dynamics_VIS(S,x,tau_S=0.1):
    h=0.1/1000 #ms
    gamma=0.641
    tau_S=.100 #s
    k=(-S/tau_S+(1-S)*gamma*H(x)/1)
    S_new=S+h*(-(S+h/2*k)/tau_S+(1-S+h/2*k)*gamma*H(x))
    return S_new

def Network_Dynamics_AUDIO(S,x,tau_S=0.1):
    h=0.1/1000 #ms
    gamma=0.641
    #tau_S=.10 #s
    tau_S=.100 #s
    k=(-S/tau_S+(1-S)*gamma*H(x)/1)
    S_new=S+h*(-(S+h/2*k)/tau_S+(1-S+h/2*k)*gamma*H(x))
    return S_new

def Network_Dynamics_AV(S,x,tau_S=0.1):
    h=0.1/1000 #ms
    gamma=0.641
    #tau_S=.10 #s
    tau_S=.100 #s
    k=(-S/tau_S+(1-S)*gamma*H(x)/1)
    S_new=S+h*(-(S+h/2*k)/tau_S+(1-S+h/2*k)*gamma*H(x))
    return S_new


# ### Input Current Target
# $c'$ is coherence in this formula but we use $c'$ as strenght of stimulus
# $$ I_i=J_{A,ext}\mu_0\left(1+ \frac{c'}{100} \right) $$
# default at 10 is close to 100 % hits but not quiet

# %%


def I_input_1(c_dash):
    J_A_ext=5.2/10000 # nA/Hz
    mu_0=30 # Hz
    I_motion=J_A_ext*mu_0*(1+(c_dash)/100)
    return I_motion


# $$ I_2=J_{A,ext}\mu_0\left(1- \frac{c'}{100} \right) $$

# %%


def I_input_2(c_dash):
    J_A_ext=0.00052 # nA/Hz
    mu_0=30 # Hz
    I_motion=J_A_ext*mu_0*(1-(c_dash)/100)
    return I_motion


# ## Reaction Time Function
# This function detects when firing rate goes above Threshold.
# It takes in Firing rate for HIT and MISS, threshold and time, it returns ANSWER, Reation Time (RT) and count.
# If count =1 there has been a response if count =0 no response.
# 

# %%


def Reaction_Time_UNI(Firing_Rate_1,Firing_Rate_2,Threshold,time):
    ANSWER=0
    RT=0
    count=0
    if (Firing_Rate_1>=Threshold ): 
        ANSWER=1
        RT=time
        count=1
    elif (Firing_Rate_2>=Threshold):
        ANSWER=0
        RT=time
        count=1
    return ANSWER,RT,count     


# ## Multisensory Winner Take All
# The function takes in both Audio and Visual activity and checks which pass threshold.

# %%


def Reaction_Time_MULT(FR_Audio_HIT,Firing_Rate_2,FR_Video_HIT,Firing_Rate_4,Threshold,time):
    ANSWER=0
    RT=0
    count=0
    if (FR_Audio_HIT>=Threshold )| (FR_Video_HIT >=Threshold): 
        ANSWER=1
        RT=time
        count=1
    elif (Firing_Rate_2>=Threshold)|(Firing_Rate_4 >=Threshold):
        ANSWER=0
        #RT=time
        count=1
    return ANSWER,RT,count


# ## Setting up time
# Each epoch (trial) is between -100 ms and 1500ms.
# The Threshold is set to 20 Hz.

# %%


h=0.1
time=np.arange(-100,1500,h)
J_A_ext=0.00052 # nA/Hz
mu_0=30 # Hz
STIMULUS=[15.0]#,7.5,10.0,15.0]
Threshold=20


# # Parameters
# * K is number of "participants"
# * N is number of trials

# %%


K=50 #51
N=501

RT_AUDIO_coh_hit=[]
RT_AUDIO_coh_miss=[]#np.zeros(len(Vector_coherence))
Prob_AUDIO=[]#np.zeros(len(Vector_coherence))
RT_VIS_coh_hit=[]#np.zeros(len(Vector_coherence))
RT_VIS_coh_miss=[]#np.zeros(len(Vector_coherence))
Prob_VIS=[]#np.zeros(len(Vector_coherence))


GROUP_RT=np.zeros((3,K))
GROUP_ACC=np.zeros((3,K))
PRIMED_RT=np.zeros((3,K))
PRIMED_ACC=np.zeros((3,K))
NOT_PRIMED_RT=np.zeros((3,K))
NOT_PRIMED_ACC=np.zeros((3,K))
PRIMED_Drift=np.zeros((3,K))
NOT_PRIMED_Drift=np.zeros((3,K))
PRIMED_time_delay=np.zeros((3,K))
NOT_PRIMED_time_delay=np.zeros((3,K))

ALL_F_1=0.2*np.ones((N,len(time)))
ALL_F_2=0.2*np.ones((N,len(time)))
I_VIS_HIT=0.0*np.ones(len(time)) 
I_VIS_MISS=0.0*np.ones(len(time)) 
I_AUDIO_HIT=0.0*np.ones(len(time)) 
I_AUDIO_MISS=0.0*np.ones(len(time)) 

Firing_target_VIS_HIT=0*time # np.zeros((1,len(time)))
Firing_target_VIS_MISS=0*time # np.zeros((1,len(time)))
Firing_target_AUDIO_HIT=0*time # np.zeros((1,len(time)))
Firing_target_AUDIO_MISS=0*time # np.zeros((1,len(time)))



# %%


TRIAL_TYPE=np.random.randint(3, size=N) 
TRIAL_REPEAT=np.random.randint(2, size=N) 
AUDIO_REPEAT=np.random.randint(2, size=N) 
VISUAL_REPEAT=np.random.randint(2, size=N) 


# %%


AUDIO_REPEAT=np.zeros(N) 
VISUAL_REPEAT=np.zeros(N) 


# %%


TRIAL_TYPE=np.random.randint(3, size=N) 
AUDIO_REPEAT[((TRIAL_TYPE==1) | (TRIAL_TYPE==2))]=1
VISUAL_REPEAT[((TRIAL_TYPE==0) | (TRIAL_TYPE==2))]=1


# Setting up group data

# %%


AV_Drift=np.zeros(K)
V_Drift=np.zeros(K)
A_Drift=np.zeros(K)
Pred_Drift=np.zeros(K)


AV_time_delay=np.zeros(K)
V_time_delay=np.zeros(K)
A_time_delay=np.zeros(K)


# ### Participant Variability
# Here each participant gets their own audio time constant $\tau_A$ which is chosen randomly from a flat distribution from $(0.007, 0.013)$.
# For simplicities sake $\tau_V=0.009$ and held constant for all participants.

# %%


TAU_AUDIO=np.random.uniform(low=0.007, high=0.013, size=K)
TAU_VIDEO=np.random.uniform(low=0.007, high=0.013, size=K)


df_TAU=pd.DataFrame({'Audio': TAU_AUDIO, 'Visual': TAU_VIDEO})


df_TAU.to_csv('DATA_PY/TAU.csv') 


# ### The DDM function
# "Participants" Audio, Visual and Audio-Visual Reaction time and accuracy data are each submitted inturn to the general drift diffusion model to fit a drift rate (k) and time delay ($\tau_{r}$).
# * Drift range is between 5 and 14.
# * Noise is set to 1.5 (standard)
# * Bound is set to 2.5
# For a simple reaction time the drift rate for combination of sense is predicted by 
# $$ \hat{k}_{AV}=\sqrt{k_A^2+k_V^2},$$
# where $k_A$ and $k_V$ are the audio and visual drift rates and $\hat{k}_{AV}$ is the predicted drift rate.

# %%


def DDM_FIT(RT,ANSWER):
    df=[]
    
    # RT is scalles to seconds, the function takes seconds
    df=pd.DataFrame({'RT': RT/1000, 'correct': ANSWER})
    df.head()
    sample = Sample.from_pandas_dataframe(df, rt_column_name="RT",
                                                  correct_column_name="correct")
    model = Model(name='Model',
                      drift=DriftConstant(drift=Fittable(minval=6, maxval=25)),
                      noise=NoiseConstant(noise=1.5),#(noise=Fittable(minval=0.5, maxval=2.5)),
                      bound=BoundConstant(B=2.5),
                      overlay=OverlayChain(overlays=[OverlayNonDecision(nondectime=Fittable(minval=0, maxval=.8)),
                                                    OverlayPoissonMixture(pmixturecoef=.02,
                                                                          rate=1)]),
                      dx=.001, dt=.01, T_dur=2)

    # Fitting this will also be fast because PyDDM can automatically
    # determine that DriftCoherence will allow an analytical solution.
    fit_model = fit_adjust_model(sample=sample, model=model,fitting_method="differential_evolution",
                     lossfunction=LossRobustBIC,verbose=False)

    param=fit_model.get_model_parameters()

    Drift=np.asarray(param[0])
    Delay=np.asarray(param[1])
  
    return Drift,Delay


# ## Main Function
# Three loops, Participant (k and K), Trials (n and N) and time (i).

# %%


## PARTICIPANT LOOP
for k in range(0,K):
  # SETTING UP INDIVIDUAL RTS and ANSWERS (HIT 1 and MISS 0)
    ANSWER_VIS=np.zeros(N)
    RT_VIS=np.zeros(N)
    ANSWER_AUDIO=np.zeros(N)
    RT_AUDIO=np.zeros(N)
    ANSWER_AV=np.zeros(N)
    RT_AV=np.zeros(N)


    tau_audio=TAU_AUDIO[k] ## PARTICIPANT TAU
    tau_video=TAU_VIDEO[k] ## PARTICIPANT TAU

    
    for n in range(0,N): # TRIAL LOOP
        if n%50==0:
            print('k: %d of %d n: %d of %d' %(k,K-1,n,N-1))
        I_noise_VIS_HIT=0.001*np.random.normal(0,1,len(time))
        I_noise_VIS_MISS=0.001*np.random.normal(0,1,len(time))

        I_noise_AUDIO_HIT=0.001*np.random.normal(0,1,len(time))
        I_noise_AUDIO_MISS=0.001*np.random.normal(0,1,len(time))


        x_VIS_HIT=J_A_ext*mu_0*np.random.uniform(0,1,len(time))
        x_VIS_MISS=J_A_ext*mu_0*np.random.uniform(0,1,len(time))
        x_AUDIO_HIT=J_A_ext*mu_0*np.random.uniform(0,1,len(time))
        x_AUDIO_MISS=J_A_ext*mu_0*np.random.uniform(0,1,len(time))
        
        S_VIS_HIT=0.2*np.ones(len(time))+0.01*np.random.normal(0,1,len(time))
        S_AUDIO_HIT=0.2*np.ones(len(time))+0.01*np.random.normal(0,1,len(time))
        S_VIS_MISS=0.2*np.ones(len(time))+0.01*np.random.normal(0,1,len(time)) 
        S_AUDIO_MISS=0.2*np.ones(len(time))+0.01*np.random.normal(0,1,len(time)) 


        # INITIAL CONDITIONS
        if AUDIO_REPEAT[n]==1:
            S_AUDIO_HIT=0.2*np.ones(len(time))+0.01*np.random.normal(0,1,len(time))+0.01 # PRIME
        if VISUAL_REPEAT[n]==1:
            S_VIS_HIT=0.2*np.ones(len(time))+0.01*np.random.normal(0,1,len(time))+0.01 # PRIME
       


        
 

        x_VIS_HIT,x_VIS_MISS=total_synaptic_current(S_VIS_HIT,S_VIS_MISS,
                                                                      I_VIS_HIT,I_VIS_MISS,
                                                                      I_noise_VIS_HIT,
                                                                      I_noise_VIS_MISS)
        x_AUDIO_HIT,x_AUDIO_MISS=total_synaptic_current(S_AUDIO_HIT,
                                                                          S_AUDIO_MISS,
                                                                          I_AUDIO_HIT,
                                                                          I_AUDIO_MISS,
                                                                          I_noise_AUDIO_HIT,
                                                                          I_noise_AUDIO_MISS)

        count_AUDIO=0
        count_VIS=0
        count_AV=0

        Firing_target_VIS_HIT[0]=H(x_VIS_HIT[0])
        Firing_target_VIS_MISS[0]=H(x_VIS_MISS[0])
        Firing_target_AUDIO_HIT[0]=H(x_VIS_HIT[0])
        Firing_target_AUDIO_MISS[0]=H(x_VIS_MISS[0])

   

        
        # TIME LOOP
        for i in range (0,len(time)-1):
            if time[i] >=0 and time[i]<1000:
                c_dash=STIMULUS[0]
            else:
                c_dash=0.0

        
            I_noise_VIS_HIT[i+1]=Background_Activity(I_noise_VIS_HIT[i])
            I_noise_VIS_MISS[i+1]=Background_Activity(I_noise_VIS_MISS[i])
            I_VIS_HIT[i+1]=I_input_1(c_dash) # VISUAL HIT INPUT
            I_VIS_MISS[i+1]=I_input_1(-c_dash) # VISUAL MISS INPUT
         
         
            S_VIS_HIT[i+1]=Network_Dynamics_VIS(S_VIS_HIT[i],x_VIS_HIT[i],tau_video)
            S_VIS_MISS[i+1]=Network_Dynamics_VIS(S_VIS_MISS[i],x_VIS_MISS[i],tau_video)
            
            
            
            x_VIS_HIT[i+1],x_VIS_MISS[i+1]=total_synaptic_current(S_VIS_HIT[i+1],S_VIS_MISS[i+1],
                                                                      I_VIS_HIT[i+1],I_VIS_MISS[i+1],
                                                                      I_noise_VIS_HIT[i+1],
                                                                      I_noise_VIS_MISS[i+1])
            
            I_noise_AUDIO_HIT[i+1]=Background_Activity(I_noise_AUDIO_HIT[i])
            I_noise_AUDIO_MISS[i+1]=Background_Activity(I_noise_AUDIO_MISS[i])
            
            I_AUDIO_HIT[i+1]=I_input_1(c_dash) # AUDITORY HIT INPUT
            I_AUDIO_MISS[i+1]=I_input_1(-c_dash) # AUDITORY MISS INPUT

            S_AUDIO_HIT[i+1]=Network_Dynamics_AUDIO(S_AUDIO_HIT[i],x_AUDIO_HIT[i],tau_audio)
            S_AUDIO_MISS[i+1]=Network_Dynamics_AUDIO(S_AUDIO_MISS[i],x_AUDIO_MISS[i],tau_audio)
            
            x_AUDIO_HIT[i+1],x_AUDIO_MISS[i+1]=total_synaptic_current(S_AUDIO_HIT[i+1],
                                                                          S_AUDIO_MISS[i+1],
                                                                          I_AUDIO_HIT[i+1],
                                                                          I_AUDIO_MISS[i+1],
                                                                          I_noise_AUDIO_HIT[i+1],
                                                                          I_noise_AUDIO_MISS[i+1])
            
            
            Firing_target_AUDIO_HIT[i+1]=H(x_AUDIO_HIT[i+1])
            Firing_target_AUDIO_MISS[i+1]=H(x_AUDIO_MISS[i+1])
            Firing_target_VIS_HIT[i+1]=H(x_VIS_HIT[i+1])
            Firing_target_VIS_MISS[i+1]=H(x_VIS_MISS[i+1])
           
            # AV RACE MODEL REACTION TIME
            if count_AV <0.5:
                ANSWER_AV[n],RT_AV[n],count_AV=Reaction_Time_MULT(Firing_target_VIS_HIT[i],Firing_target_VIS_MISS[i],Firing_target_AUDIO_HIT[i],Firing_target_AUDIO_MISS[i],Threshold,time[i])
            
            # VISUAL REACTION TIME THRESHOLD
            if count_VIS <0.5:
                ANSWER_VIS[n],RT_VIS[n],count_VIS=Reaction_Time_UNI(Firing_target_VIS_HIT[i],Firing_target_VIS_MISS[i],Threshold,time[i])
            
            # AUDITORY REACTION TIME THRESHOLD
            if count_AUDIO <0.5:
                ANSWER_AUDIO[n],RT_AUDIO[n],count_AUDIO=Reaction_Time_UNI(Firing_target_AUDIO_HIT[i],Firing_target_AUDIO_MISS[i],Threshold,time[i])
    
    
    # GENERATES GROUP DATA BY AVERAGING PARTICIPANTS TRIALS            
    GROUP_RT[0,k]=np.mean(RT_AUDIO[ANSWER_AUDIO==1])
    GROUP_RT[1,k]=np.mean(RT_VIS[ANSWER_VIS==1])
    GROUP_RT[2,k]=np.mean(RT_AV[ANSWER_AV==1])
    GROUP_ACC[0,k]=np.mean(ANSWER_AUDIO)
    GROUP_ACC[1,k]=np.mean(ANSWER_VIS)
    GROUP_ACC[2,k]=np.mean(ANSWER_AV)
    
    
    PRIMED_RT[0,k]=np.mean(RT_AUDIO[(ANSWER_AUDIO==1) & (AUDIO_REPEAT==1)])
    PRIMED_RT[1,k]=np.mean(RT_VIS[(ANSWER_VIS==1) & (VISUAL_REPEAT==1)])
    PRIMED_RT[2,k]=np.mean(RT_AV[ANSWER_AV==1])
    PRIMED_ACC[0,k]=np.mean(ANSWER_AUDIO[TRIAL_REPEAT==1])
    PRIMED_ACC[1,k]=np.mean(ANSWER_VIS[TRIAL_REPEAT==1])
    PRIMED_ACC[2,k]=np.mean(ANSWER_AV[TRIAL_REPEAT==1])
    NOT_PRIMED_RT[0,k]=np.mean(RT_AUDIO[(ANSWER_AUDIO==1) & (AUDIO_REPEAT==0)])
    NOT_PRIMED_RT[1,k]=np.mean(RT_VIS[(ANSWER_VIS==1) & (VISUAL_REPEAT==0)])
    NOT_PRIMED_RT[2,k]=np.mean(RT_AV[ANSWER_AV==1])
    NOT_PRIMED_ACC[0,k]=np.mean(ANSWER_AUDIO[AUDIO_REPEAT==0])
    NOT_PRIMED_ACC[1,k]=np.mean(ANSWER_VIS[VISUAL_REPEAT==0])
    NOT_PRIMED_ACC[2,k]=np.mean(ANSWER_AV[VISUAL_REPEAT==0])    
    ## FITTING THE OUTPUTS
    A_Drift[k],A_time_delay[k]=DDM_FIT(RT_AUDIO,ANSWER_AUDIO)
    V_Drift[k],V_time_delay[k]=DDM_FIT(RT_VIS,ANSWER_VIS)
    AV_Drift[k],AV_time_delay[k]=DDM_FIT(RT_AV,ANSWER_AV)
    
    Pred_Drift[k]=np.sqrt(A_Drift[k]*A_Drift[k]+V_Drift[k]*V_Drift[k])
    PRIMED_Drift[0,k],PRIMED_time_delay[0,k]=DDM_FIT(RT_AUDIO[AUDIO_REPEAT==1],ANSWER_AUDIO[AUDIO_REPEAT==1])
    PRIMED_Drift[1,k],PRIMED_time_delay[1,k]=DDM_FIT(RT_VIS[VISUAL_REPEAT==1],ANSWER_VIS[VISUAL_REPEAT==1])
    PRIMED_Drift[2,k],PRIMED_time_delay[2,k]=DDM_FIT(RT_AV[(AUDIO_REPEAT==1) & (VISUAL_REPEAT==1)],ANSWER_AV[(AUDIO_REPEAT==1) & (VISUAL_REPEAT==1)])

    NOT_PRIMED_Drift[0,k],NOT_PRIMED_time_delay[0,k]=DDM_FIT(RT_AUDIO[AUDIO_REPEAT==0],ANSWER_AUDIO[AUDIO_REPEAT==0])
    NOT_PRIMED_Drift[1,k],NOT_PRIMED_time_delay[1,k]=DDM_FIT(RT_VIS[VISUAL_REPEAT==0],ANSWER_VIS[VISUAL_REPEAT==0])
 


# ## Results

# %%


fig = plt.figure(figsize=(14,4))
plt.subplot(131)
plt.plot(time,Firing_target_AUDIO_HIT.T,'y')
plt.plot(time,Firing_target_AUDIO_MISS.T,'r')

plt.hlines(Threshold,-100,1500,colors='grey',linestyles='dashed',label='Threshold')
plt.vlines(RT_AUDIO[-1],0,Threshold,colors='k',linestyles='dashed',label='Reaction Time')
plt.xlabel('time(ms)')
plt.ylabel('Firing Rate')

plt.title("Auditory Trial")
#plt.legend()
plt.subplot(132)
plt.plot(time,Firing_target_VIS_HIT.T,'b')
plt.plot(time,Firing_target_VIS_MISS.T,'r')
plt.hlines(Threshold,-100,1500,colors='grey',linestyles='dashed',label='Threshold')
plt.vlines(RT_VIS[-1],0,Threshold,colors='k',linestyles='dashed',label='Reaction Time')
#plt.legend()
plt.xlabel('time(ms)')
plt.title("Visual Trial")
plt.subplot(133)
plt.plot(time,Firing_target_VIS_HIT.T,'b',label='Visual')
plt.plot(time,Firing_target_AUDIO_HIT.T,'y',label='Audio')
#plt.plot(time,Firing_target_VIS_MISS.T)
plt.hlines(Threshold,-100,1500,colors='grey',linestyles='dashed',label='Threshold')
plt.vlines(RT_AV[-1],0,Threshold,colors='k',linestyles='dashed',label='Reaction Time')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel('time(ms)')
plt.title("Audio-Visual Trial")
plt.show()


# ### Reaction Time
# ### Individual Example

# %%


fig = plt.figure(figsize=(14,4))
plt.subplot(132)
plt.hist(RT_VIS[VISUAL_REPEAT==1],bins=20,facecolor='red', ec="k", alpha=0.5)
plt.vlines(np.mean(RT_VIS[VISUAL_REPEAT==1]),0,210,linestyles='dashed',color='red')
plt.xlim(0,1200)
plt.ylim(0,210)
plt.xlabel('Reaction Time',fontsize=15)
plt.title('Visual Only ',fontsize=20)


plt.subplot(131)
plt.hist(RT_AUDIO[AUDIO_REPEAT==1],bins=20,facecolor='blue', ec="k", alpha=0.5)
plt.xlabel('Reaction Time',fontsize=15)
plt.vlines(np.mean(RT_AUDIO[AUDIO_REPEAT==1]),0,210,linestyles='dashed',color='blue')
plt.xlim(0,1200)
plt.ylim(0,210)


plt.title('Audio Only',fontsize=20)


plt.subplot(133)
plt.hist(RT_AV[ANSWER_AV==1],bins=20,facecolor='grey', ec="k", alpha=0.5)
plt.vlines(np.mean(RT_AV[ANSWER_AV==1]),0,210,color='grey',linestyles='dashed',label="Mean Reaction Time")
plt.xlabel('Reaction Time',fontsize=15)
plt.title('Audio Visual',fontsize=20)
plt.xlim(0,1200)
plt.ylim(0,210)
plt.savefig('FIGURES/FigureRTPRIME.eps',dpi=300)

plt.savefig('FIGURES/FigureRTPRIME.png',dpi=300)



plt.show()


# %%


fig = plt.figure(figsize=(14,8))
plt.subplot(222)
plt.hist(RT_VIS[(ANSWER_VIS==1) & (VISUAL_REPEAT==1)],bins=20,facecolor='blue', ec="k", alpha=0.5)
plt.vlines(np.mean(RT_VIS[(ANSWER_VIS==1) & (VISUAL_REPEAT==1)]),0,50,linestyles='dashed',color='b')
plt.xlim(0,1000)

plt.xlabel('Reaction Time',fontsize=15)
#plt.title('Visual Only Primed',fontsize=20)


plt.subplot(221)
plt.hist(RT_AUDIO[(ANSWER_AUDIO==1) & (AUDIO_REPEAT==1)],bins=20,facecolor='yellow', ec="k", alpha=0.5)
plt.xlabel('Reaction Time',fontsize=15)
plt.vlines(np.mean(RT_AUDIO[(ANSWER_AUDIO==1) & (AUDIO_REPEAT==1)]),0,50,linestyles='dashed',color='y')
plt.xlim(0,1000)


plt.subplot(224)
plt.hist(RT_VIS[(ANSWER_VIS==1) & (VISUAL_REPEAT==0)],bins=20,facecolor='blue', ec="k", alpha=0.5)
plt.vlines(np.mean(RT_VIS[(ANSWER_VIS==1) & (VISUAL_REPEAT==0)]),0,50,linestyles='dashed',color='b')
plt.xlim(0,1000)

plt.xlabel('Reaction Time',fontsize=15)


plt.subplot(223)
plt.hist(RT_AUDIO[(ANSWER_AUDIO==1) & (AUDIO_REPEAT==0)],bins=20,facecolor='yellow', ec="k", alpha=0.5)
plt.xlabel('Reaction Time',fontsize=15)
plt.vlines(np.mean(RT_AUDIO[(ANSWER_AUDIO==1) & (AUDIO_REPEAT==0)]),0,50,linestyles='dashed',color='y')
plt.xlim(0,1000)




#plt.title('Audio Only Primed',fontsize=20)
plt.show


# ### Group Analysis

# %%


df_RT=pd.DataFrame({'Audio': GROUP_RT[0,:], 'Visual': GROUP_RT[1,:],'AV':GROUP_RT[2,:]})

fig = plt.figure(figsize=(6,6))
ax=df_RT.boxplot(grid=False, rot=45, fontsize=15)
#ax=df.plot.scatter()
#fig.set_ylabel('RT')
ax.set_ylabel('RT', fontsize=15)
ax.set_ylim((0,400))
#df.boxplot(grid=False, rot=45, fontsize=15)
plt.show()

df_RT.to_csv('DATA/GROUP_RT.csv') 


# %%


df_RT=pd.DataFrame({'Audio': PRIMED_RT[0,:], 'Visual': PRIMED_RT[1,:],'AV':PRIMED_RT[2,:]})

fig = plt.figure(figsize=(6,6))
ax=df_RT.boxplot(grid=False, rot=45, fontsize=15)
#ax=df.plot.scatter()
#fig.set_ylabel('RT')
ax.set_ylabel('RT', fontsize=15)
ax.set_ylim((0,400))
#df.boxplot(grid=False, rot=45, fontsize=15)
plt.show()

df_RT.to_csv('DATA/GROUP_RT_PRIMED.csv') 


# %%


df_RT=pd.DataFrame({'Audio': NOT_PRIMED_RT[0,:], 'Visual': NOT_PRIMED_RT[1,:],'AV':NOT_PRIMED_RT[2,:]})

fig = plt.figure(figsize=(6,6))
ax=df_RT.boxplot(grid=False, rot=45, fontsize=15)
#ax=df.plot.scatter()
#fig.set_ylabel('RT')
ax.set_ylabel('RT', fontsize=15)
ax.set_ylim((0,600))
#df.boxplot(grid=False, rot=45, fontsize=15)
plt.show()
df_RT.to_csv('DATA/GROUP_RT_NOT_PRIMED.csv') 


# ### ACCURACY 

# %%


df_ACC=pd.DataFrame({'Audio': GROUP_ACC[0,:], 'Visual': GROUP_ACC[1,:],'AV':GROUP_ACC[2,:]})
#from matplotlib import pyplot as plt

fig = plt.figure(figsize=(6,4))
ax=df_ACC.boxplot(grid=False, rot=45, fontsize=15)
#ax=df.plot.scatter()
#fig.set_ylabel('RT')
ax.set_ylabel('Accuracy', fontsize=15)
plt.ylim(0.8,1.1)
#df.boxplot(grid=False, rot=45, fontsize=15)
plt.show()
df_ACC.to_csv('DATA/GROUP_ACC.csv') 


# ## DDM Results
# ## Drift Rate

# %%


df_Drift=pd.DataFrame({'Audio': A_Drift, 'Visual': V_Drift,'AV':AV_Drift})
#from matplotlib import pyplot as plt

fig = plt.figure(figsize=(6,6))
ax=df_Drift.boxplot(grid=False, rot=45, fontsize=15)
#ax=df.plot.scatter()
#fig.set_ylabel('RT')
ax.set_ylabel(r'Drift Rate $\mu$', fontsize=15)
#df.boxplot(grid=False, rot=45, fontsize=15)
plt.show()
df_Drift.to_csv('DATA/GROUP_ACC.csv') 


# ### Time Delay

# %%



df_delay=pd.DataFrame({'Audio': A_time_delay, 'Visual': V_time_delay,'AV':AV_time_delay})
#from matplotlib import pyplot as plt

fig = plt.figure(figsize=(6,6))
ax=df_delay.boxplot(grid=False, rot=45, fontsize=15)
#ax=df.plot.scatter()
#fig.set_ylabel('RT')
ax.set_ylabel('non-decision time [s]', fontsize=15)
#df.boxplot(grid=False, rot=45, fontsize=15)
plt.show()


# ### Observed and Predicted

# %%


fig = plt.figure(figsize=(6,6))
plt.plot(Pred_Drift,AV_Drift,'ko',markeredgewidth=2,markerfacecolor='white', markersize=10)
plt.plot([8,14],[8,14],'k:')
plt.ylabel('Observed AV Drift Rate',fontsize=20)
plt.xlabel('Predicted Drift Rate',fontsize=20)
plt.xlim(8,14)
plt.ylim(8,14)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.axis('square')
plt.show()


# %%


RT_melt = pd.melt(df_RT.reset_index(), id_vars=['index'], value_vars=['Audio', 'Visual', 'AV'])
# replace column names
RT_melt.columns = ['index', 'condition', 'RT']


# Ordinary Least Squares (OLS) model
model = ols('RT ~ C(condition)', data=RT_melt).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
anova_table


# %%


ACC_melt = pd.melt(df_ACC.reset_index(), id_vars=['index'], value_vars=['Audio', 'Visual', 'AV'])
# replace column names
ACC_melt.columns = ['index', 'condition', 'ACC']


# Ordinary Least Squares (OLS) model
model = ols('ACC ~ C(condition)', data=ACC_melt).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
anova_table


# %%



Drift_melt = pd.melt(df_Drift.reset_index(), id_vars=['index'], value_vars=['Audio', 'Visual', 'AV'])
# replace column names
Drift_melt.columns = ['index', 'condition', 'Drift']

# Ordinary Least Squares (OLS) model
model = ols('Drift ~ C(condition)', data=Drift_melt).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
anova_table


# %%



Delay_melt = pd.melt(df_delay.reset_index(), id_vars=['index'], value_vars=['Audio', 'Visual', 'AV'])
# replace column names
Delay_melt.columns = ['index', 'condition', 'Delay']

# Ordinary Least Squares (OLS) model
model = ols('Delay ~ C(condition)', data=Delay_melt).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
anova_table


# ## Reference
# Fearon, C., Butler, J. S., Newman, L., Lynch, T., & Reilly, R. B. (2015). Audiovisual processing is abnormal in Parkinson’s disease and correlates with freezing of gait and disease duration. Journal of Parkinson's disease, 5(4), 925-936.

# %%





# %%


fig, ax = plt.subplots(figsize=(6,6))
pal = "Set2"
ax=sns.stripplot( x = "condition", y = "RT", data = RT_melt, palette = pal,
edgecolor = "white", size = 4, jitter = 1, zorder = 0)
ax=sns.boxplot( x = "condition", y = "RT", data = RT_melt, color = "black",
width = .15, zorder = 10, showcaps = True,
boxprops = {'facecolor':'none', "zorder":10}, showfliers=True, whiskerprops = {'linewidth':4, "zorder":10},
saturation = 1)
ax.set_ylim(150,300)
plt.savefig('FIGURES/FigureRTGroupPRIMED.eps',dpi=300)

plt.savefig('FIGURES/FigureRTGroupPRIMED.png',dpi=300)


plt.show()


# %%


fig, ax = plt.subplots(figsize=(6,6))
pal = "Set2"
ax=sns.stripplot( x = "condition", y = "Drift", data = Drift_melt, palette = pal,
edgecolor = "white", size = 10, jitter = 1, zorder = 0)
ax=sns.boxplot( x = "condition", y = "Drift", data = Drift_melt, color = "black",
width = .15, zorder = 10, showcaps = True,
boxprops = {'facecolor':'none', "zorder":10}, showfliers=True, whiskerprops = {'linewidth':4, "zorder":10},
saturation = 1)
ax.set_ylim(0,22)
plt.savefig('FIGURES/FigureDriftGroupPRIMED.eps',dpi=300)

plt.savefig('FIGURES/FigureDriftGroupPRIMED.png',dpi=300)


plt.show()


# %%


PRIMED_Pred_Drift=np.sqrt(PRIMED_Drift[0,:]*PRIMED_Drift[0,:]+PRIMED_Drift[1,:]*PRIMED_Drift[1,:])


# %%


fig = plt.figure(figsize=(6,6))
plt.plot(Pred_Drift,PRIMED_Drift[2,:],'ko',markeredgewidth=2,markerfacecolor='white', markersize=10)
plt.plot([8,25],[8,25],'k:')
plt.ylabel('Observed AV Drift Rate',fontsize=20)
plt.xlabel('Predicted Drift Rate',fontsize=20)
plt.xlim(8,14)
plt.ylim(8,14)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.axis('square')
plt.savefig('FIGURES/FigurePredictedPRIMED1.eps',dpi=300)

plt.savefig('FIGURES/FigurePredictedPRIMED1.png',dpi=300)


plt.show()


# %%


fig = plt.figure(figsize=(6,6))
plt.plot(PRIMED_Pred_Drift,PRIMED_Drift[2,:],'ko',markeredgewidth=2,markerfacecolor='white', markersize=10)
plt.plot([8,25],[8,25],'k:')
plt.ylabel('Observed AV Drift Rate',fontsize=20)
plt.xlabel('Predicted Drift Rate',fontsize=20)
plt.xlim(8,14)
plt.ylim(8,14)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.axis('square')
plt.savefig('FIGURES/FigurePredictedPRIMED.eps',dpi=300)

plt.savefig('FIGURES/FigurePredictedPRIMED.png',dpi=300)


plt.show()


# %%


np.sum(AUDIO_REPEAT==1)


# %%




