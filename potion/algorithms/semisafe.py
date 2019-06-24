#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semi-Safe Policy Gradient (SSPG)
@author: Matteo Papini
"""

from potion.simulation.trajectory_generators import generate_batch
from potion.common.misc_utils import performance, avg_horizon, mean_sum_info, clip, seed_all_agent, returns, separator
from potion.estimation.gradients import gpomdp_estimator, reinforce_estimator
from potion.common.logger import Logger
import torch
from potion.estimation.eigenvalues import power
import time
import scipy.stats as sts
from scipy.sparse.linalg import eigsh
import math


def semisafepg(env, policy, horizon, *,
                    conf = 0.05,
                    min_batchsize = 32,
                    max_batchsize = 5000,
                    iterations = float('inf'),
                    max_samples = 1e6,
                    disc = 0.9,
                    forget = 0.1,
                    action_filter = None,
                    estimator = 'gpomdp',
                    baseline = 'peters',
                    logger = Logger(name='SSPG'),
                    shallow = True,
                    pow_step = 0.01,
                    pow_decay = 0.1,
                    pow_it = 20,
                    pow_epochs = 5,
                    pow_tol = 0.1,
                    pow_clip = 0.2,
                    fast = False,
                    meta_conf = 0.05,
                    seed = None,
                    test_batchsize = False,
                    info_key = 'danger',
                    save_params = 100,
                    log_params = True,
                    log_grad = False,
                    parallel = False,
                    render = False,
                    verbose = 1):
    """
    Semi-safe PG algorithm from "Smoothing Policies and Safe Policy Gradients,
                                    Papini et al., 2019
        
    env: environment
    policy: the one to improve
    horizon: maximum task horizon
    conf: probability of unsafety (per update)
    min_batchsize: minimum number of trajectories used to estimate policy gradient
    max_batchsize: maximum number of trajectories used to estimate policy gradient
    iterations: maximum number of learning iterations
    max_samples: maximum number of total trajectories 
    disc: discount factor
    forget: decay of the (estimated) global gradient Lipscthiz constant
    action_filter: function to apply to the agent's action before feeding it to 
        the environment, not considered in gradient estimation. By default,
        the action is clipped to satisfy evironmental boundaries
    estimator: either 'reinforce' or 'gpomdp' (default). The latter typically
        suffers from less variance
    baseline: control variate to be used in the gradient estimator. Either
        'avg' (average reward, default), 'peters' (variance-minimizing) or
        'zero' (no baseline)
    logger: for human-readable logs (standard output, csv, tensorboard)
    shallow: whether to employ pre-computed score functions (only available for
        shallow policies)
    pow_step: step size of the power method
    pow_decay: initial decay parameter of the power method
    pow_it: maximum number of iterations (per epoch) of the power method
    pow_epochs: maximum number of epochs of the power method
    pow_tol: relative-error tolerance of the power method
    pow_clip: importance-weight clipping parameter for the power method (default 0.2)
    seed: random seed (None for random behavior)
    fast: whether to pursue maximum convergence speed under safety constraints
    meta_conf: confidence level of safe update test (for evaluation)
    test_batchsize: number of test trajectories used to evaluate the 
        corresponding deterministic policy at each iteration. If False, no 
        test is performed
    info_key: name of the environment info to log
    save_params: how often (every x iterations) to save the policy 
        parameters to disk. Final parameters are always saved for 
        x>0. If False, they are never saved.
    log_params: whether to include policy parameters in the human-readable logs
    log_grad: whether to include gradients in the human-readable logs
    parallel: number of parallel jobs for simulation. If False, 
        sequential simulation is performed.
    render: how often (every x iterations) to render the agent's behavior
        on a sample trajectory. If False, no rendering happens
    verbose: level of verbosity
    """
    #Defaults
    if action_filter is None:
        action_filter = clip(env)
    
    #Seed agent
    if seed is not None:
        seed_all_agent(seed)
    
    #Prepare logger
    algo_info = {'Algorithm': 'SSPG',
                   'Estimator': estimator,
                   'Baseline': baseline,
                   'Env': str(env), 
                   'Horizon': horizon,
                   'Discount': disc,
                   'Confidence': conf,
                   'ConfidenceParam': conf,
                   'Seed': seed,
                   'MinBatchSize': min_batchsize,
                   'MaxBatchSize': max_batchsize,
                   'ForgetParam': forget,
                   'PowerStep': pow_step,
                   'PowerDecay': pow_decay,
                   'PowerIters': pow_it,
                   'PowerEpochs': pow_epochs,
                   'PowerTolerance': pow_tol,
                   'Fast': fast
                   }
    logger.write_info({**algo_info, **policy.info()})
    log_keys = ['Perf', 
                'UPerf', 
                'AvgHorizon', 
                'StepSize', 
                'GradNorm', 
                'Time',
                'StepSize',
                'BatchSize',
                'LipConst',
                'ErrBound',
                'SampleVar',
                'Info',
                'TotSamples',
                'Safety',
                'TScore']
    if log_params:
        log_keys += ['param%d' % i for i in range(policy.num_params())]
    if log_grad:
        log_keys += ['grad%d' % i for i in range(policy.num_params())]
    if test_batchsize:
        log_keys += ['TestPerf', 'TestPerf', 'TestInfo']
    log_row = dict.fromkeys(log_keys)
    logger.open(log_row.keys())
    
    #Initializations
    it = 0
    updated = False
    updates = 0
    unsafe_updates = 0
    safety = 1.
    tot_samples = 0
    optimal_batchsize = min_batchsize
    min_safe_batchsize = min_batchsize
    _estimator = reinforce_estimator if estimator=='reinforce' else gpomdp_estimator
    old_lip_const = 0.
    
    #Learning loop
    while(it < iterations and tot_samples < max_samples):
        start = time.time()
        if verbose:
            print('\n* Iteration %d *' % it)
        params = policy.get_flat()
        
        #Test the corresponding deterministic policy
        if test_batchsize:
            test_batch = generate_batch(env, policy, horizon, 
                                        episodes=test_batchsize, 
                                        action_filter=action_filter,
                                        n_jobs=parallel,
                                        deterministic=True,
                                        key=info_key)
            log_row['TestPerf'] = performance(test_batch, disc)
            log_row['UTestPerf'] = performance(test_batch, 1)
            log_row['TestInfo'] = mean_sum_info(test_batch).item()
        
        #Render the agent's behavior
        if render and it % render==0:
            generate_batch(env, policy, horizon,
                           episodes=1,
                           action_filter=action_filter, 
                           render=True)
    
    
        #Experience loop
        _conf = conf
        target_batchsize = min_safe_batchsize if fast else optimal_batchsize
        #Collect trajectories according to target batch size
        batch = generate_batch(env, policy, horizon, 
                                episodes=max(min_batchsize, min(max_batchsize, target_batchsize)), 
                                action_filter=action_filter,
                                n_jobs=parallel,
                                key=info_key)
        batchsize = len(batch)
        
        do = True
        while do or batchsize < min_safe_batchsize:
            do = False
            #Collect more trajectories to match minimum safe batch size
            batch += generate_batch(env, policy, horizon, 
                        episodes=min(max_batchsize, min_safe_batchsize) - batchsize, 
                        action_filter=action_filter,
                        n_jobs=parallel,
                        key=info_key)
            batchsize = len(batch)
            
            #Estimate policy gradient
            grad_samples = _estimator(batch, disc, policy, 
                                        baselinekind=baseline, 
                                        shallow=shallow,
                                        result='samples')
            grad = torch.mean(grad_samples, 0)
                
            #Compute estimation error with ellipsoid confidence region
            centered = grad_samples - grad.unsqueeze(0)
            grad_cov = batchsize/(batchsize - 1) * torch.mean(torch.bmm(centered.unsqueeze(2), centered.unsqueeze(1)),0)
            grad_var = torch.sum(torch.diag(grad_cov)).item() #only for human-readable logs
            max_eigv = eigsh(grad_cov.numpy(), 1)[0][0]
            dfn = grad.shape[0]
            quant = sts.f.ppf(1 - _conf, dfn, batchsize - dfn)
            eps = math.sqrt(max_eigv * dfn * quant)
            
            #Optimal batch size
            optimal_batchsize = torch.ceil(4 * eps**2 / 
                                   (torch.norm(grad)**2) + dfn).item()
            min_safe_batchsize = torch.ceil(eps**2 / torch.norm(grad)**2 + dfn).item()
            if verbose and optimal_batchsize < max_batchsize:
                print('Collected %d / %d trajectories' % (batchsize, optimal_batchsize))
            elif verbose:
                print('Collected %d / %d trajectories' % (batchsize, min(max_batchsize, min_safe_batchsize)))
            
            #Adjust confidence before collecting more data for the same update
            _conf /= 2
            if batchsize >= max_batchsize:
                break
        
        if verbose:
            print('Optimal batch size: %d' % optimal_batchsize if optimal_batchsize < float('inf') else -1)
            print('Minimum safe batch size: %d' % min_safe_batchsize if min_safe_batchsize < float('inf') else -1)
            if batchsize >= min_safe_batchsize and batchsize < optimal_batchsize:
                print('Low sample regime')
                
        #Update safety measure
        if updates == 0:
            old_rets= returns(batch, disc)
        elif updated:
            new_rets = returns(batch, disc)
            tscore, pval = sts.ttest_ind(old_rets, new_rets)
            if pval / 2 < meta_conf and tscore > 0:
                unsafe_updates += 1
                if verbose:
                    print('The previous update was unsafe! (p-value = %f)' % (pval / 2))
            old_rets = new_rets
            safety = 1 - unsafe_updates / updates

        #Update long-term quantities
        tot_samples += batchsize
        
        #Log
        log_row['SampleVar'] = grad_var
        log_row['TScore'] = torch.norm(grad).item() / math.sqrt(grad_var / batchsize)
        log_row['Safety'] = safety
        log_row['ErrBound'] = eps
        log_row['Perf'] = performance(batch, disc)
        log_row['Info'] = mean_sum_info(batch).item()
        log_row['UPerf'] = performance(batch, disc=1.)
        log_row['AvgHorizon'] = avg_horizon(batch)
        log_row['GradNorm'] = torch.norm(grad).item()
        log_row['BatchSize'] = batchsize
        log_row['TotSamples'] = tot_samples
        if log_params:
            for i in range(policy.num_params()):
                log_row['param%d' % i] = params[i].item()
        if log_grad:
            for i in range(policy.num_params()):
                log_row['grad%d' % i] = grad[i].item()
                
        #Check if number of samples is sufficient to perform update
        if batchsize < min_safe_batchsize:
            updated = False
            if verbose:
                print('No update, would require more samples than allowed by memory constraints')
            #Log
            log_row['LipConst'] = old_lip_const
            log_row['StepSize'] = 0.
            log_row['Time'] = time.time() - start
            if verbose:
                print(separator)
            logger.write_row(log_row, it)
            if verbose:
                print(separator)
            
            #Adjust confidence before collecting new data for the same update
            _conf /= 2
            
            #Skip to next iteration (current trajectories are discarded)
            it += 1
            continue
        
        #Estimate (local) gradient Lipschitz constant with off-policy Power Method
        lip_const = power(policy, batch, grad, disc, 
              step=pow_step, 
              decay=pow_decay,
              tol=pow_tol, 
              max_it=pow_it, 
              max_ep=pow_epochs, 
              estimator=_estimator, 
              baseline=baseline, 
              shallow=shallow, 
              clip=pow_clip,
              verbose=verbose)
        
        #Update global lipschitz constant
        if it > 0:
            lip_const = (1 - forget) * max(lip_const, old_lip_const) + forget * lip_const
        old_lip_const = lip_const
        log_row['LipConst'] = lip_const
        
        #Select step size
        stepsize = 1. / lip_const * (1 - eps / (torch.norm(grad) * math.sqrt(batchsize - dfn)).item())
        if fast:
            stepsize *= 2
        log_row['StepSize'] = stepsize
                
        #Update policy parameters
        new_params = params + stepsize * grad
        policy.set_from_flat(new_params)
        updated = True
        updates += 1
        
        #Save parameters
        if save_params and it % save_params == 0:
            logger.save_params(params, it)
        
        #Next iteration
        log_row['Time'] = time.time() - start
        if verbose:
            print(separator)
        logger.write_row(log_row, it)
        if verbose:
            print(separator)
        it += 1
    
    #Save final parameters
    if save_params:
        logger.save_params(params, it)
    
    #Cleanup
    logger.close()
    
    
def adabatch2(env, policy, horizon, pen_coeff, var_bound, *,
                    conf = 0.05,
                    min_batchsize = 32,
                    max_batchsize = 5000,
                    iterations = float('inf'),
                    max_samples = 1e6,
                    disc = 0.9,
                    action_filter = None,
                    estimator = 'gpomdp',
                    baseline = 'peters',
                    bound = 'student',
                    logger = Logger(name='SSPG'),
                    shallow = True,
                    fast = False,
                    meta_conf = 0.05,
                    seed = None,
                    test_batchsize = False,
                    info_key = 'danger',
                    save_params = 100,
                    log_params = True,
                    log_grad = False,
                    parallel = False,
                    render = False,
                    verbose = 1):
    """
    Semi-safe PG algorithm from "Adaptive Batch Size for Safe Policy Gradients",
                                Papini et al., 2017.
    Only for Gaussian policies.
        
    env: environment
    policy: the one to improve
    horizon: maximum task horizon
    pen_coeff: penalty coefficient for policy update
    var_bound: upper bound on the variance of the PG estimator
    horizon: maximum task horizon
    conf: probability of unsafety (per update)
    min_batchsize: minimum number of trajectories used to estimate policy gradient
    max_batchsize: maximum number of trajectories used to estimate policy gradient
    iterations: maximum number of learning iterations
    max_samples: maximum number of total trajectories 
    disc: discount factor
    forget: decay of the (estimated) global gradient Lipscthiz constant
    action_filter: function to apply to the agent's action before feeding it to 
        the environment, not considered in gradient estimation. By default,
        the action is clipped to satisfy evironmental boundaries
    estimator: either 'reinforce' or 'gpomdp' (default). The latter typically
        suffers from less variance
    baseline: control variate to be used in the gradient estimator. Either
        'avg' (average reward, default), 'peters' (variance-minimizing) or
        'zero' (no baseline)
    logger: for human-readable logs (standard output, csv, tensorboard)
    shallow: whether to employ pre-computed score functions (only available for
        shallow policies)
    seed: random seed (None for random behavior)
    fast: whether to pursue maximum convergence speed under safety constraints
    meta_conf: confidence level of safe update test (for evaluation)
    test_batchsize: number of test trajectories used to evaluate the 
        corresponding deterministic policy at each iteration. If False, no 
        test is performed
    info_key: name of the environment info to log
    save_params: how often (every x iterations) to save the policy 
        parameters to disk. Final parameters are always saved for 
        x>0. If False, they are never saved.
    log_params: whether to include policy parameters in the human-readable logs
    log_grad: whether to include gradients in the human-readable logs
    parallel: number of parallel jobs for simulation. If False, 
        sequential simulation is performed.
    render: how often (every x iterations) to render the agent's behavior
        on a sample trajectory. If False, no rendering happens
    verbose: level of verbosity
    """
    #Defaults
    if action_filter is None:
        action_filter = clip(env)
    
    #Seed agent
    if seed is not None:
        seed_all_agent(seed)
    
    #Prepare logger
    algo_info = {'Algorithm': 'SSPG',
                   'Estimator': estimator,
                   'Baseline': baseline,
                   'Env': str(env), 
                   'Horizon': horizon,
                   'Discount': disc,
                   'Confidence': conf,
                   'ConfidenceParam': conf,
                   'Seed': seed,
                   'MinBatchSize': min_batchsize,
                   'MaxBatchSize': max_batchsize,
                   'PenalizationCoefficient': pen_coeff,
                   'VarianceBound': var_bound,
                   'Fast': fast
                   }
    logger.write_info({**algo_info, **policy.info()})
    log_keys = ['Perf', 
                'UPerf', 
                'AvgHorizon', 
                'StepSize', 
                'GradNorm', 
                'Time',
                'StepSize',
                'BatchSize',
                'LipConst',
                'ErrBound',
                'SampleVar',
                'Info',
                'TotSamples',
                'Safety']
    if log_params:
        log_keys += ['param%d' % i for i in range(policy.num_params())]
    if log_grad:
        log_keys += ['grad%d' % i for i in range(policy.num_params())]
    if test_batchsize:
        log_keys += ['TestPerf', 'TestPerf', 'TestInfo']
    log_row = dict.fromkeys(log_keys)
    logger.open(log_row.keys())
    
    #Initializations
    it = 0
    updated = False
    updates = 0
    unsafe_updates = 0
    safety = 1.
    tot_samples = 0
    optimal_batchsize = min_batchsize
    min_safe_batchsize = min_batchsize
    _estimator = reinforce_estimator if estimator=='reinforce' else gpomdp_estimator
    old_lip_const = 0.
    
    #Learning loop
    while(it < iterations and tot_samples < max_samples):
        start = time.time()
        if verbose:
            print('\n* Iteration %d *' % it)
        params = policy.get_flat()
        
        #Test the corresponding deterministic policy
        if test_batchsize:
            test_batch = generate_batch(env, policy, horizon, 
                                        episodes=test_batchsize, 
                                        action_filter=action_filter,
                                        n_jobs=parallel,
                                        deterministic=True,
                                        key=info_key)
            log_row['TestPerf'] = performance(test_batch, disc)
            log_row['UTestPerf'] = performance(test_batch, 1)
            log_row['TestInfo'] = mean_sum_info(test_batch).item()
        
        #Render the agent's behavior
        if render and it % render==0:
            generate_batch(env, policy, horizon,
                           episodes=1,
                           action_filter=action_filter, 
                           render=True)
    
    
        #Experience loop
        _conf = conf
        target_batchsize = min_safe_batchsize if fast else optimal_batchsize
        #Collect trajectories according to target batch size
        batch = generate_batch(env, policy, horizon, 
                                episodes=max(min_batchsize, min(max_batchsize, target_batchsize)), 
                                action_filter=action_filter,
                                n_jobs=parallel,
                                key=info_key)
        batchsize = len(batch)
        
        do = True
        while do or batchsize < min_safe_batchsize:
            do = False
            #Collect more trajectories to match minimum safe batch size
            batch += generate_batch(env, policy, horizon, 
                        episodes=min(max_batchsize, min_safe_batchsize) - batchsize, 
                        action_filter=action_filter,
                        n_jobs=parallel,
                        key=info_key)
            batchsize = len(batch)
            
            #Estimate policy gradient
            grad_samples = _estimator(batch, disc, policy, 
                                        baselinekind=baseline, 
                                        shallow=shallow,
                                        result='samples')
            grad = torch.mean(grad_samples, 0)
            grad_infnorm = torch.max(torch.abs(grad))
            coordinate = torch.min(torch.argmax(torch.abs(grad))).item()
                
            #Compute estimation error with ellipsoid confidence region
            centered = grad_samples - grad.unsqueeze(0)
            grad_cov = batchsize/(batchsize - 1) * torch.mean(torch.bmm(centered.unsqueeze(2), centered.unsqueeze(1)),0)
            grad_var = torch.max(torch.diag(grad_cov)).item()
            quant = sts.t.ppf(1 - _conf, batchsize) 
            eps = quant * math.sqrt(grad_var)
            
            #Optimal batch size
            optimal_batchsize = math.ceil(((13 + 3 * math.sqrt(17)) * eps**2 / (2 * grad_infnorm**2)).item())
            min_safe_batchsize = math.ceil((eps**2 / grad_infnorm**2).item())
            if verbose and optimal_batchsize < max_batchsize:
                print('Collected %d / %d trajectories' % (batchsize, optimal_batchsize))
            elif verbose:
                print('Collected %d / %d trajectories' % (batchsize, min(max_batchsize, min_safe_batchsize)))
            
            #Adjust confidence before collecting more data for the same update
            _conf /= 2
            if batchsize >= max_batchsize:
                break
        
        if verbose:
            print('Optimal batch size: %d' % optimal_batchsize if optimal_batchsize < float('inf') else -1)
            print('Minimum safe batch size: %d' % min_safe_batchsize if min_safe_batchsize < float('inf') else -1)
            if batchsize >= min_safe_batchsize and batchsize < optimal_batchsize:
                print('Low sample regime')
                
        #Update safety measure
        if updates == 0:
            old_rets= returns(batch, disc)
        elif updated:
            new_rets = returns(batch, disc)
            tscore, pval = sts.ttest_ind(old_rets, new_rets)
            if pval / 2 < meta_conf and tscore > 0:
                unsafe_updates += 1
                if verbose:
                    print('The previous update was unsafe! (p-value = %f)' % (pval / 2))
            old_rets = new_rets
            safety = 1 - unsafe_updates / updates

        #Update long-term quantities
        tot_samples += batchsize
        
        #Log
        log_row['SampleVar'] = grad_var
        log_row['Safety'] = safety
        log_row['ErrBound'] = eps
        log_row['Perf'] = performance(batch, disc)
        log_row['Info'] = mean_sum_info(batch).item()
        log_row['UPerf'] = performance(batch, disc=1.)
        log_row['AvgHorizon'] = avg_horizon(batch)
        log_row['GradNorm'] = torch.norm(grad).item()
        log_row['BatchSize'] = batchsize
        log_row['TotSamples'] = tot_samples
        if log_params:
            for i in range(policy.num_params()):
                log_row['param%d' % i] = params[i].item()
        if log_grad:
            for i in range(policy.num_params()):
                log_row['grad%d' % i] = grad[i].item()
                
        #Check if number of samples is sufficient to perform update
        if batchsize < min_safe_batchsize:
            updated = False
            if verbose:
                print('No update, would require more samples than allowed by memory constraints')
            #Log
            log_row['StepSize'] = 0.
            log_row['Time'] = time.time() - start
            if verbose:
                print(separator)
            logger.write_row(log_row, it)
            if verbose:
                print(separator)
            
            #Adjust confidence before collecting new data for the same update
            _conf /= 2
            
            #Skip to next iteration (current trajectories are discarded)
            it += 1
            continue
        
        #Select step size
        stepsize = ((grad_infnorm - eps / math.sqrt(batchsize))**2 \
                    /(2 * pen_coeff * (grad_infnorm + eps / math.sqrt(batchsize))**2)).item()
        if fast:
            stepsize *= 2
        log_row['StepSize'] = stepsize
                
        #Update policy parameters
        new_params = params
        new_params[coordinate] = params[coordinate] + stepsize * grad[coordinate]
        policy.set_from_flat(new_params)
        updated = True
        updates += 1
        
        #Save parameters
        if save_params and it % save_params == 0:
            logger.save_params(params, it)
        
        #Next iteration
        log_row['Time'] = time.time() - start
        if verbose:
            print(separator)
        logger.write_row(log_row, it)
        if verbose:
            print(separator)
        it += 1
    
    #Save final parameters
    if save_params:
        logger.save_params(params, it)
    
    #Cleanup
    logger.close()