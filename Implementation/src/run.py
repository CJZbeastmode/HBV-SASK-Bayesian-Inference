import time
import numpy as np
import sys
import json

sys.path.append('/Users/jay/Desktop/Bachelorarbeit/Implementation')
from dependencies.hbv_sask.model import HBVSASKModel as hbvmodel
from dependencies.PyDREAM.pydream.convergence import Gelman_Rubin
from src.run_mcmc.run_dream import run_mcmc_dream
from src.run_mcmc.run_gpmh import run_mcmc_gpmh
from src.run_mcmc.run_mh import run_mcmc_mh
from src.construct_model import get_model

runConfigPath = '/Users/jay/Desktop/Bachelorarbeit/run_config.json'
with open(runConfigPath, 'r') as file:
    run_config = json.load(file)

configPath = run_config['configPath']
basis = run_config['basis']
model = get_model(configPath, basis)

if __name__ == "__main__": 
    start = time.time()
    separate_chain = False
    run_mcmc = run_mcmc_dream
    sampled_params, total_iterations = run_mcmc(niterations=20, nchains=4)
    end = time.time()
    print("Time needed: " + str(end - start))

    # Check convergence and continue sampling if not converged

    #GR = Gelman_Rubin(sampled_params)
    #print('At iteration: ', total_iterations, ' GR = ', GR)
    #np.savetxt('robertson_nopysb_dreamzs_5chain_GelmanRubin_iteration_' + str(total_iterations) + '.txt', GR)

    old_samples = sampled_params

    #if np.any(GR > 1.2):
    #    starts = [sampled_params[chain][-1, :] for chain in range(nchains)]

        # while not converged:
        #     total_iterations += niterations

        #     sampled_params, log_ps = run_dream(sampled_parameter_names, likelihood, niterations=niterations,
        #                                        nchains=nchains, multitry=False, gamma_levels=4, adapt_gamma=True,
        #                                        history_thin=1, model_name='robertson_nopysb_dreamzs_5chain', verbose=True, restart=True)

        #     for chain in range(len(sampled_params)):
        #         np.save('robertson_nopysb_dreamzs_5chain_sampled_params_chain_' + str(chain) + '_' + str(total_iterations),
        #                     sampled_params[chain])
        #         np.save('robertson_nopysb_dreamzs_5chain_logps_chain_' + str(chain) + '_' + str(total_iterations),
        #                     log_ps[chain])

        #     old_samples = [np.concatenate((old_samples[chain], sampled_params[chain])) for chain in range(nchains)]
        #     GR = Gelman_Rubin(old_samples)
        #     print('At iteration: ', total_iterations, ' GR = ', GR)
        #     np.savetxt('robertson_nopysb_dreamzs_5chain_GelmanRubin_iteration_' + str(total_iterations)+'.txt', GR)

        #     if np.all(GR < 1.2):
        #         converged = True

        # Plot output
    burnin = int(total_iterations / 5)
    #old_samples = np.array(old_samples)[burnin:]
    #old_samples = old_samples[::3]

    if separate_chain:
        samples = np.hstack((old_samples[0][burnin:, :], old_samples[1][burnin:, :], old_samples[2][burnin:, :],
                                        old_samples[3][burnin:, :], old_samples[4][burnin:, :]))
        header = 'TT_1,C0_1,beta_1,ETF_1,FC_1,FRAC_1,K2_1,TT_2,C0_2,beta_2,ETF_2,FC_2,FRAC_2,K2_2,TT_3,C0_3,beta_3,ETF_3,FC_3,FRAC_3,K2_3,TT_4,C0_4,beta_4,ETF_4,FC_4,FRAC_4,K2_4,TT_5,C0_5,beta_5,ETF_5,FC_5,FRAC_5,K2_5'
        np.savetxt('mcmc_sep_data.out', samples, delimiter=',', header=header, comments='') 
    else:
        #if mode == 'MH':
        #np.savetxt('mcmc_data.out', old_samples, delimiter=',', header='TT,C0,beta,ETF,FC,FRAC,K2', comments='') 
        #else:
        samples = np.concatenate((old_samples[0][burnin:, :], old_samples[1][burnin:, :], old_samples[2][burnin:, :],
                                            old_samples[3][burnin:, :], old_samples[4][burnin:, :]))
        np.savetxt('mcmc_data.out', samples, delimiter=',', header='TT,C0,beta,ETF,FC,FRAC,K2') 

else:
    pass
    #run_kwargs = {'parameters':sampled_parameter_names, 'likelihood':likelihood, 'niterations':10000, 'nchains':nchains, 'multitry':False, 'gamma_levels':4, 'adapt_gamma':True, 'history_thin':1, 'model_name':'robertson_nopysb_dreamzs_5chain', 'verbose':True}
