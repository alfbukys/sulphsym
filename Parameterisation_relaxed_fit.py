"""This code runs the parameter fitting of the model. It outputs the fitted 
parameters as a 2D numpy array and saves each as .txt file"""
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution


#All relevant rate equations are described below-------------------------------

"""Irreversible Henri-Michaelis Menten kinetics"""
def Irr_Michaelis_Menten (S, Vmax, Km):
    return(Vmax*S / (Km+S))

"""Irreversible specific activation kinetics"""
def Irr_Spec_Action (S, A, Vmax, Km, ka):
    return(Vmax*S*A / (Km*ka + (Km+S) * A))

"""Irreversible non-competitive inhibition"""
def Irr_Noncomp_Inhib (S, I, Vmax, Km, ki):
    return(Vmax*S / ((Km+S)*(1+I/ki) + S))

"""Irreversible competitive inhibition"""
def Irr_Comp_Inhib (S, I, Vmax, Km, ki):
    return(Vmax*S / (Km*(1+I/ki) + S))

"""Reversible Bi-bi kinetics"""
def Rev_BiBi (S, ATP, aps, ppi, Vf, Km_S, Km_ATP, Km_aps, Km_ppi, keq):
    return(Vf/Km_S/Km_ATP * (S*ATP - (aps*ppi/keq)) / (1 + S/Km_S + ATP/Km_ATP + S*ATP/Km_S/Km_ATP + aps/Km_aps + ppi/Km_ppi + aps*ppi/Km_aps/Km_ppi))

"""Mass action kinetics. Used for consumption of molecules"""
def Consume (S, k):
    return(k*S)

"""Constant flux. Used for production of molecules"""
def Produce (v):
    return(v)

"""Reversible Henri-Michaelis-Menten kinetics """
def Rev_Michaelis_Menten (S, P, Vf, Vr, Km_S, Km_P):
    return((Vf*S/Km_S - Vr*P/Km_P) / (1+S/Km_S+P/Km_P))

"""Reversible Ter kinetics for APRp"""
def APRp (aps, GSH, SO3, GSSG, AMP, Vmax, Km_aps, Km_GSH, Km_SO3, Km_GSSG, Km_AMP, keq):
    return (Vmax/Km_aps/Km_GSH**2 * (aps*GSH**2 - SO3*GSSG*AMP/keq) / ( (1+aps/Km_aps)*(1+GSH/Km_GSH)**2 + (1+SO3/Km_SO3)*(1+GSSG/Km_GSSG)*(1+AMP/Km_AMP) -1 ))

"""Reversible Bi kinetics for GRp"""
def GRp (GSSG, NADPH, GSH, NADP, Vmax, Km_GSSG, Km_NADPH, Km_GSH, Km_NADP, keq):
    return(Vmax/Km_GSSG/Km_NADPH * (GSSG*NADPH - GSH**2 * NADP/keq) / ( (1+GSSG/Km_GSSG)*(1+NADPH/Km_NADPH) + (1+NADP/Km_NADP)*(1+GSH/Km_GSH)**2 - 1))

"""Reversible Henri-Michaelis-Menten kinetics for SiR"""
def SiR (S, P, Vmax, Km_S, Km_P, keq):
    return(Vmax/Km_S * (S-P/keq) / (1+S/Km_S+P/Km_P))

"""Reversible Michaelis-Menten kinetics with three regulatory terms"""
def Rev_Michaelis_Menten_3reg (Ser, OAS, S, cys, Vf, Vr, Km_S, Km_P, ki, ka, n):
    return((1+S**n/(S**n + ka**n))*(ki/(ki+OAS))*((Vf*Km_P*Ser - Vr*Km_S*OAS) / ((1+cys/ki)*(Km_S*Km_P + Km_S*OAS + Km_P*Ser))))

"""Reversible Michaelis-Menten kinetics with two regulatory terms """
def Rev_Michaelis_Menten_2reg (Ser, OAS, S, Vf, Vr, Km_S, Km_P, ki, ka, n):
    return((1+S**n/(S**n + ka**n))*(ki/(ki+OAS))*((Vf*Km_P*Ser - Vr*Km_S*OAS) / (Km_S*Km_P + Km_S*OAS + Km_P*Ser)))

"""Reversible Bi Uni kinetics + two regulatory terms"""
def Rev_Bi_Uni_2reg (OAS, S, cys, Vf, Vr, Km_S, Km_OAS, Km_cys, ki, ka, n):
    return((ki/(S+ki)) * (1+OAS**n/(ka+OAS**n)) * ((Vf*S*OAS/Km_S/Km_OAS - Vr*cys/Km_cys) / (1 + S/Km_S + OAS/Km_OAS + S*OAS/Km_S/Km_OAS + cys/Km_cys)))

"""Reversible Ter kinetics + non-competitive inhibition """
def Rev_Ter_NC_Inhib (cys, glu, ATP, gEC, ADP, Pi, GSH, Vmax, Km_cys, Km_glu, Km_ATP, Km_gEC, Km_ADP, Km_Pi, ki, keq):
    return(Vmax/Km_cys/Km_glu/Km_ATP * (cys*glu*ATP - gEC*ADP*Pi/keq) / ( (1+GSH/ki)*(1+cys/Km_cys)*(1+glu/Km_glu)*(1+ATP/Km_ATP) + (1+gEC/Km_gEC)*(1+ADP/Km_ADP)*(1+Pi/Km_Pi) - 1))

"""Irreversible Ter kinetics + product inhibition"""
def Irr_Ter_Prod_Inhib (a, b, c, p, q, r, Vmax, Kma, Kmb, Kmc, Kipa, Kipb, Kipc, Kiqa, Kiqb, Kiqc, Kira, Kirb, Kirc):
    return(Vmax*a*b*c/((Kmc*(1+p/Kipc)*(1+q/Kiqc)*(1+r/Kirc)*a*(1+p/Kipa)*(1+q/Kiqa)*(1+r/Kira)*b*(1+p/Kipb)*(1+q/Kiqb)*(1+r/Kirb)+Kmb*(1+p/Kipb)*(1+q/Kiqb)*(1+r/Kirb)*a*(1+p/Kipa)*(1+q/Kiqa)*(1+r/Kira)*c*(1+p/Kipc)*(1+r/Kirc)+Kma*(1+p/Kipa)*(1+q/Kiqa)*(1+r/Kira)*b*(1+p/Kipb)*(1+q/Kiqb)*(1+r/Kirb)*c*(1+p/Kipc)*(1+r/Kirc) + a*(1+p/Kipa)*(1+q/Kiqa)*(1+r/Kira)*b*(1+p/Kipb)*(1+q/Kiqb)*(1+r/Kirb)*c*(1+p/Kipc)*(1+r/Kirc))))

"""Ordered Bi-Bi, substrate inhibition"""
def Ord_Bi_Bi_Sub_Inhib (A, B, P, Q, Vmax, Km_a, Km_b, Km_p, Km_q, ki_a, KI_B, ki_p, ki_q, K__IB, k__iq):
    return(Vmax*A*B / ( ki_a*Km_b + Km_b*A + Km_a*B + A*B + A*B**2/KI_B + Km_b*ki_a*Km_q*P/ki_q/Km_p + Km_b*A*P/ki_p + Km_b*Km_q*A*P/ki_q/Km_p + A*B*P/ki_p + Km_b*Km_q*A*P**2/ki_q/Km_p/ki_p + Km_b*ki_a*Q/ki_q + Km_a*B*Q/k__iq + ki_a*B**2*Q/ki_q*K__IB) )

#------------------------------------------------------------------------------

#Main kinetics function which is used to calculate the metabolite fluxes-------

"""This is used for modelling the kinetics of sulphate metabolism. 
Returns numpy array with all metabolite fluxes which is then integrated
via the SciPy function solve_ivp"""
def Kinetics (time, starting_states, parameters):

    #Compartment volumes in litres 
    V_c = 5.4e-8 
    V_p = 2.54e-7
    V_v = 6.8e-7 

    #Defining the initial states
    (SO4ic, SO4iv, SO4ip, APSc, PPic, APSp, PPip,
    SO3p, Sp, PAPSc, PAPSp, ADPc, ADPp, AMPp) = starting_states

    #Defining the parameters
    (Vm_Sultr1_1, Km_Sultr1_1, 
     Vm_Sultr1_2, Km_Sultr1_2, 
     Vm_Sultr4cv, Km_Sultr4cv, ka_Sultr4cv, 
     Vm_Sultr4vc, Km_Sultr4vc, ki_Sultr4vc, 
     Vf_ATPSc, KmS_ATPSc, Kmatp_ATPSc, Kmaps_ATPSc, Kmppi_ATPSc, keq_ATPSc, 
     k_APSc, k_PPic,
     Vf_Sultr3cp, Vr_Sultr3cp, KmS_Sultr3cp, KmP_Sultr3cp,
     Vf_ATPSp, KmS_ATPSp, Kmatp_ATPSp, Kmaps_ATPSp, Kmppi_ATPSp, keq_ATPSp,
     k_APSp, k_PPip,
     Vm_APRp, Kmaps_APRp, Kmgsh_APRp, KmSO3_APRp, Kmgssg_APRp, Kmamp_APRp, keq_APRp,
     k_SO3p, 
     Vm_SiR, KmS_SiR, KmP_SiR, keq_SiR,
     ATP_c, ATP_p, SO4_ex, GSHp, GSSGp, 
     Vm_APK_c, Km_a_APK_c, Km_b_APK_c, Km_p_APK_c, Km_q_APK_c, ki_a_APK_c, KI_B_APK_c, ki_p_APK_c, ki_q_APK_c, K__IB_APK_c, k__iq_APK_c,
     Vm_APK_p, Km_a_APK_p, Km_b_APK_p, Km_p_APK_p, Km_q_APK_p, ki_a_APK_p, KI_B_APK_p, ki_p_APK_p, ki_q_APK_p, K__IB_APK_p, k__iq_APK_p,
     k_ADPc, k_ADPp, k_AMPp, 
     Vm_PAPST1, Km_PAPST1, ki_PAPST1, 
     k_PAPSc, k_Sp) = parameters
    

    #Rate equations stored in:
    dydt = np.empty(len(starting_states))

    #Change of SO4ic. 6 reactions: +Sultr1;1, +Sultr1;2, -Sultr3cp, -Sultr4cv, +Sultr4vc, -ATPSc
    dydt[0] = ( 
                + Irr_Michaelis_Menten(SO4_ex, Vm_Sultr1_1, Km_Sultr1_1) / V_c
                + Irr_Michaelis_Menten(SO4_ex, Vm_Sultr1_2, Km_Sultr1_2) / V_c
                - Rev_Michaelis_Menten(SO4ic, SO4ip, Vf_Sultr3cp, Vr_Sultr3cp, KmS_Sultr3cp, KmP_Sultr3cp) / V_c
                - Irr_Spec_Action(SO4ic, SO4ic, Vm_Sultr4cv, Km_Sultr4cv, ka_Sultr4cv)/ V_c
                + Irr_Noncomp_Inhib(SO4iv, SO4ic, Vm_Sultr4vc, Km_Sultr4vc, ki_Sultr4vc)/ V_c
                - Rev_BiBi(SO4ic, ATP_c, APSc, PPic, Vf_ATPSc, KmS_ATPSc, Kmatp_ATPSc, Kmaps_ATPSc, Kmppi_ATPSc, keq_ATPSc)
    )

    #Change of SO4iv.  2 reactions: +Sultr4cv,  -Sultr4vc,
    dydt[1] = ( 
                + Irr_Spec_Action(SO4ic, SO4ic, Vm_Sultr4cv, Km_Sultr4cv, ka_Sultr4cv) / V_v
                - Irr_Noncomp_Inhib(SO4iv, SO4ic, Vm_Sultr4vc, Km_Sultr4vc, ki_Sultr4vc) / V_v
    )
    
    #Change of SO4ip. 2 reactions: -ATPSp, +Sultr3cp
    dydt[2] = (
                - Rev_BiBi(SO4ip, ATP_p, APSp, PPip, Vf_ATPSp, KmS_ATPSp, Kmatp_ATPSp, Kmaps_ATPSp, Kmppi_ATPSp, keq_ATPSp)
                + Rev_Michaelis_Menten(SO4ic, SO4ip, Vf_Sultr3cp, Vr_Sultr3cp, KmS_Sultr3cp, KmP_Sultr3cp) / V_p
    )

    #Change of APSc. 2 Reactions: +ATPSc, -APK
    dydt[3] = (
                + Rev_BiBi(SO4ic, ATP_c, APSc, PPic, Vf_ATPSc, KmS_ATPSc, Kmatp_ATPSc, Kmaps_ATPSc, Kmppi_ATPSc, keq_ATPSc)
                - Ord_Bi_Bi_Sub_Inhib(ATP_c, APSc, PAPSc, ADPc, Vm_APK_c, Km_a_APK_c, Km_b_APK_c, Km_p_APK_c, Km_q_APK_c, ki_a_APK_c, KI_B_APK_c, ki_p_APK_c, ki_q_APK_c, K__IB_APK_c, k__iq_APK_c)
    )
    
    #Change of PPic. 2 reactions: +ATPSc, -kPPic
    dydt[4] = (
                + Rev_BiBi(SO4ic, ATP_c, APSc, PPic, Vf_ATPSc, KmS_ATPSc, Kmatp_ATPSc, Kmaps_ATPSc, Kmppi_ATPSc, keq_ATPSc)
                - Consume(PPic, k_PPic)
    )
    
    #Change of APSp. 3 reactions: +ATPSp, -APRp, -APK. 
    dydt[5] = (
                 + Rev_BiBi(SO4ip, ATP_p, APSp, PPip, Vf_ATPSp, KmS_ATPSp, Kmatp_ATPSp, Kmaps_ATPSp, Kmppi_ATPSp, keq_ATPSp)
                 - APRp(APSp, GSHp, SO3p, GSSGp, AMPp, Vm_APRp, Kmaps_APRp, Kmgsh_APRp, KmSO3_APRp, Kmgssg_APRp, Kmamp_APRp, keq_APRp)
                 - Ord_Bi_Bi_Sub_Inhib(ATP_p, APSp, PAPSp, ADPp, Vm_APK_p, Km_a_APK_p, Km_b_APK_p, Km_p_APK_p, Km_q_APK_p, ki_a_APK_p, KI_B_APK_p, ki_p_APK_p, ki_q_APK_p, K__IB_APK_p, k__iq_APK_p)
    )
    
    #Change of PPip. 2 reactions: +ATPSp, -kPPip
    dydt[6] = (
                 + Rev_BiBi(SO4ip, ATP_p, APSp, PPip, Vf_ATPSp, KmS_ATPSp, Kmatp_ATPSp, Kmaps_ATPSp, Kmppi_ATPSp, keq_ATPSp)
                 - Consume(PPip, k_PPip)
    )

    #Change of SO3p. 3 reactions: +APRp, -SiRp, -kSO3
    dydt[7] = (
                + APRp(APSp, GSHp, SO3p, GSSGp, AMPp, Vm_APRp, Kmaps_APRp, Kmgsh_APRp, KmSO3_APRp, Kmgssg_APRp, Kmamp_APRp, keq_APRp)
                - SiR(SO3p, Sp, Vm_SiR, KmS_SiR, KmP_SiR, keq_SiR)
                - Consume(SO3p, k_SO3p)
    )

    #Change of Sp. 2 reactions: +SiRp, -kSp
    dydt[8] = (
                + SiR(SO3p, Sp, Vm_SiR, KmS_SiR, KmP_SiR, keq_SiR)
                - Consume(Sp, k_Sp)
    )

    #Change of PAPSc. 3 reactions: +APKc, +PAPST1, -kPAPSc
    dydt[9] = (
                + Ord_Bi_Bi_Sub_Inhib(ATP_c, APSc, PAPSc, ADPc, Vm_APK_c, Km_a_APK_c, Km_b_APK_c, Km_p_APK_c, Km_q_APK_c, ki_a_APK_c, KI_B_APK_c, ki_p_APK_c, ki_q_APK_c, K__IB_APK_c, k__iq_APK_c)
                + Irr_Comp_Inhib(PAPSp, ATP_p, Vm_PAPST1, Km_PAPST1, ki_PAPST1) / V_c
                - Consume(PAPSc, k_PAPSc)
    )

    #Change of PAPSp. 2 reactions: +APKp, -PAPST1
    dydt[10] = (
                + Ord_Bi_Bi_Sub_Inhib(ATP_p, APSp, PAPSp, ADPp, Vm_APK_p, Km_a_APK_p, Km_b_APK_p, Km_p_APK_p, Km_q_APK_p, ki_a_APK_p, KI_B_APK_p, ki_p_APK_p, ki_q_APK_p, K__IB_APK_p, k__iq_APK_p)                
                - Irr_Comp_Inhib(PAPSp, ATP_p, Vm_PAPST1, Km_PAPST1, ki_PAPST1) / V_p
    )
    #Change of ADPc. 2 reactions: +APKc, -kADPc
    dydt[11] = (
                + Ord_Bi_Bi_Sub_Inhib(ATP_c, APSc, PAPSc, ADPc, Vm_APK_c, Km_a_APK_c, Km_b_APK_c, Km_p_APK_c, Km_q_APK_c, ki_a_APK_c, KI_B_APK_c, ki_p_APK_c, ki_q_APK_c, K__IB_APK_c, k__iq_APK_c)
                - Consume(ADPc, k_ADPc)
    )
    
    #Change of ADPp. 2 reactions: +APKp, -kADPp
    dydt[12] = (
                + Ord_Bi_Bi_Sub_Inhib(ATP_p, APSp, PAPSp, ADPp, Vm_APK_p, Km_a_APK_p, Km_b_APK_p, Km_p_APK_p, Km_q_APK_p, ki_a_APK_p, KI_B_APK_p, ki_p_APK_p, ki_q_APK_p, K__IB_APK_p, k__iq_APK_p)
                - Consume(ADPp, k_ADPp)                
    )
    #Change of AMPp. 2 reactions: +APRp, -kAMPp
    dydt[13] = (
                + APRp(APSp, GSHp, SO3p, GSSGp, AMPp, Vm_APRp, Kmaps_APRp, Kmgsh_APRp, KmSO3_APRp, Kmgssg_APRp, Kmamp_APRp, keq_APRp)
                - Consume(AMPp, k_AMPp)
    )

    return dydt
#------------------------------------------------------------------------------

#Cost function described below-------------------------------------------------
"""Weighted Residual Sum of Squares cost function.
Compares experimental steady-state data to model output (see paper). The higher
the difference between the real vs the modeled values, the bigger the cost. 
During parameter fitting, the goal is to reduce the cost function to 0.
This is a 'loose fit' implementation of the cost function: if model output is 
within the experimentally determined steady-state range, the cost is set to 0.
This allows greater flexibility within the parameter search space and prevents
overfitting of the parameters."""
def wRSS(params, time, starting_state, steady_states, steady_min, steady_max, weights):
    
    #Run model and store the states in y
    model = solve_ivp(Kinetics, time, starting_state, args=(params,) ,method='LSODA', rtol=1e-12, atol=1e-6)
    y = model['y']

    #model_states picks out the last value from the simulation for relevant metabolites to be compared to steady state values in literature
    model_states = np.array([y[0,-1], y[1,-1], y[2,-1], y[7,-1], y[8,-1], y[3,-1], y[9,-1], y[5,-1], y[10,-1]])
    cost = np.empty(len(model_states))
    
    #'Loose fit' implementation
    for i in range(len(model_states)):
        #Checks if model output is within steady-state boundaries
        if steady_min[i] < model_states[i] and model_states[i] < steady_max[i]:
            cost[i] = 0
        #If model output is outside boundaries, calculate cost
        else:
            cost[i] = steady_states[i]-model_states[i]
    
    #Returns the sum of RSS values for each metabolite steady-state
    return(np.sum(weights*np.square(cost)))
#------------------------------------------------------------------------------

#Read all the files needed to run the parameterisation-------------------------

Ordered_states_DF = pd.read_csv('States_order.csv')
Ordered_params_DF = pd.read_csv('Ordered_params.csv')
Model_states_DF = pd.read_csv('Model states.csv')
Parameter_bound_DF = pd.read_csv('Parameter bounds.csv')
Steady_states_DF=pd.read_csv('Steady_simple.csv')

#Rearranging the starting states for easier readability
Ordered_states = np.array(Ordered_states_DF['Names'])
Model_states_names = np.array(Model_states_DF['Name'])
Model_states_values = np.array(Model_states_DF['Simple']) #keep in mind to update this according to parameter set used
Model_States_Dict = dict(zip(Model_states_names, Model_states_values))

Starting_states = np.empty(len(Ordered_states))
for i in range(len(Ordered_states)):
    Starting_states[i] = Model_States_Dict[Ordered_states[i]]

#Doing the same for parameters
Ordered_params = np.array(Ordered_params_DF['Name'])
Model_params_names = np.array(Parameter_bound_DF['Parameter'])
Model_params_values = np.array(Parameter_bound_DF['Model 2'])
Model_Params_Dict = dict(zip(Model_params_names, Model_params_values))

Model_parameters = np.empty(len(Ordered_params))
for i in range(len(Ordered_params)):
    Model_parameters[i] = Model_Params_Dict[Ordered_params[i]]

#Getting steady state boundaries
Lower_steady_state = np.array(Steady_states_DF['Lower concentration'])
Upper_steady_state = np.array(Steady_states_DF['Upper concentration'])
Paired_steady_state = np.array(Steady_states_DF[['Lower concentration', 'Upper concentration']])
#Taking average of steady states
Average_steady_state = np.array((Lower_steady_state+Upper_steady_state)/2)

#Setting weights
stds = np.std(Paired_steady_state, axis=1)
weights = (1/stds)
normalized = ((weights - min(weights) +1e-7) / (max(weights) - min(weights)))

#Getting the parameter search bounds
All_bounds = np.array(Parameter_bound_DF[['Lower bound', 'Upper bound']])
Bound_Dict = dict(zip(Model_params_names, All_bounds))

Bounds = np.empty((len(Ordered_params),2))
for i in range(len(Ordered_params)):
    Bounds[i] = Bound_Dict[Ordered_params[i]]
bnd = [tuple(x) for x in Bounds]

time = (0, 1e6) #sets the time integration range
#------------------------------------------------------------------------------

#Run the parameterisation------------------------------------------------------
#Loop number determines how many parameter sets are generated
for item in range(50):
    
    Fitting = differential_evolution(wRSS, bnd, args=(time, Starting_states, Average_steady_state, Lower_steady_state, Upper_steady_state, normalized), maxiter=50, disp=1, updating='immediate', popsize=1, mutation=(0.2, 0.6), polish=False)
    Fit = np.array(Fitting['x'])    
    np.savetxt('Model_{}.txt'.format(item+1), Fit, delimiter=',')    
    print(item+1, '/10 steps complete')
#------------------------------------------------------------------------------
print("THE CODE HAS FINISHED RUNNING")