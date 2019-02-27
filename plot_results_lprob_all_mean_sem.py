import os
import argparse

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import misc

import inference.mcmc as mcmc
import experiment_descriptor as ed
import pdfs

import util.io
import util.plot

root = misc.get_root()
rng = np.random.RandomState(42)

# for mcmc
thin = 10
n_mcmc_samples = 5000
burnin = 100


def get_samples_nde(exp_desc, sim, seed):
    """
    Generates MCMC samples for a given NDE experiment.
    """

    prior = sim.Prior()
    true_ps, obs_xs = sim.get_ground_truth()

    assert isinstance(exp_desc.inf, ed.NDE_Descriptor)
    res_file = os.path.join(root, 'results/seed_'+str(seed), exp_desc.get_dir(), 'samples')

    if os.path.exists(res_file + '.pkl'):
        samples = util.io.load(res_file)

    else:
        exp_dir = os.path.join(root, 'experiments/seed_'+str(seed), exp_desc.get_dir(), '0')

        if not os.path.exists(exp_dir):
            raise misc.NonExistentExperiment(exp_desc)

        net = util.io.load(os.path.join(exp_dir, 'model'))
        log_posterior = lambda t: net.eval([t, obs_xs]) + prior.eval(t)

        sampler = mcmc.SliceSampler(true_ps, log_posterior, thin=thin)
        sampler.gen(burnin, rng=rng)  # burn in
        samples = sampler.gen(n_mcmc_samples, rng=rng)

        util.io.save(samples, res_file)

    return samples


def get_samples_snl(exp_desc, sim, seed):
    """
    Generates MCMC samples for a given SNL experiment.
    """

    prior = sim.Prior()
    true_ps, obs_xs = sim.get_ground_truth()

    assert isinstance(exp_desc.inf, ed.SNL_Descriptor)
    res_file = os.path.join(root, 'results/seed_'+str(seed), exp_desc.get_dir(), 'samples')

    if os.path.exists(res_file + '.pkl'):
        all_samples = util.io.load(res_file)

    else:
        exp_dir = os.path.join(root, 'experiments/seed_'+str(seed), exp_desc.get_dir(), '0')

        if not os.path.exists(exp_dir):
            raise misc.NonExistentExperiment(exp_desc)

        _, _, all_nets = util.io.load(os.path.join(exp_dir, 'results'))
        all_samples = []

        for net in all_nets:

            net.reset_theano_functions()
            log_posterior = lambda t: net.eval([t, obs_xs]) + prior.eval(t)
            sampler = mcmc.SliceSampler(true_ps, log_posterior, thin=thin)
            sampler.gen(burnin, rng=rng)  # burn in
            samples = sampler.gen(n_mcmc_samples, rng=rng)

            all_samples.append(samples)

        util.io.save(all_samples, res_file)

    return all_samples


def view_samples_nde(sim_name, seed):
    """
    Plots MCMC samples for all NDE experiments.
    """

    sim = get_sim(sim_name)

    true_ps, _ = sim.get_ground_truth()

    for exp_desc in ed.parse(util.io.load_txt('exps/{0}_nl.txt'.format(sim_name))):

        samples = get_samples_nde(exp_desc, sim, seed)

        fig = util.plot.plot_hist_marginals(samples, lims=sim.get_disp_lims(), gt=true_ps)
        fig.suptitle('NDE, sims = {0}'.format(exp_desc.inf.n_samples))

    plt.plot()


def view_samples_snl(sim_name, seed):
    """
    Plots MCMC samples for all SNL experiments.
    """

    sim = get_sim(sim_name)

    true_ps, _ = sim.get_ground_truth()

    for exp_desc in ed.parse(util.io.load_txt('exps/{0}_seq.txt'.format(sim_name))):

        if isinstance(exp_desc.inf, ed.SNL_Descriptor):

            all_samples = get_samples_snl(exp_desc, sim, seed)

            for i, samples in enumerate(all_samples):
                fig = util.plot.plot_hist_marginals(samples, lims=sim.get_disp_lims(), gt=true_ps)
                fig.suptitle('SNL, round = {0}'.format(i + 1))

    plt.plot()


def view_samples_sl(sim_name, seed):
    """
    Plots samples from all the synth likelihood experiments.
    """

    sim = get_sim(sim_name)

    true_ps, _ = sim.get_ground_truth()

    for exp_desc in ed.parse(util.io.load_txt('exps/{0}_sl.txt'.format(sim_name))):

        exp_dir = os.path.join(root, 'experiments/seed_'+str(seed), exp_desc.get_dir(), '0')
        samples, _ = util.io.load(os.path.join(exp_dir, 'results'))

        fig = util.plot.plot_hist_marginals(samples, lims=sim.get_disp_lims(), gt=true_ps)
        fig.suptitle('Synth Lik, sims = {0}'.format(exp_desc.inf.n_sims))

    plt.plot()


def calc_err(true_ps, samples, weights=None):
    """
    Calculates error (neg log prob of truth) for a set of possibly weighted samples.
    """

    std = n_mcmc_samples ** (-1.0 / (len(true_ps) + 4))

    return -pdfs.gaussian_kde(samples, weights, std).eval(true_ps)


def get_err_nde(exp_desc, sim, seed):
    """
    Calculates the error for a given NDE experiment.
    """

    assert isinstance(exp_desc.inf, ed.NDE_Descriptor)
    res_file = os.path.join(root, 'results/seed_'+str(seed), exp_desc.get_dir(), 'err')

    if os.path.exists(res_file + '.pkl'):
        err = util.io.load(res_file)

    else:
        true_ps, _ = sim.get_ground_truth()
        err = calc_err(true_ps, get_samples_nde(exp_desc, sim, seed))
        util.io.save(err, res_file)

    return err


def get_err_smc(exp_desc, sim, seed):
    """
    Calculates the error for a given SMC ABC experiment.
    """

    assert isinstance(exp_desc.inf, ed.SMC_ABC_Descriptor)
    res_file = os.path.join(root, 'results/seed_'+str(seed), exp_desc.get_dir(), 'err')

    if os.path.exists(res_file + '.pkl'):
        all_errs, all_n_sims = util.io.load(res_file)

    else:
        exp_dir = os.path.join(root, 'experiments/seed_'+str(seed), exp_desc.get_dir(), '0')

        if not os.path.exists(exp_dir):
            raise misc.NonExistentExperiment(exp_desc)

        true_ps, _ = sim.get_ground_truth()

        all_samples, all_log_weights, _, _, all_n_sims = util.io.load(os.path.join(exp_dir, 'results'))
        all_errs = []

        for samples, log_weights in zip(all_samples, all_log_weights):

            weights = np.exp(log_weights)
            err = calc_err(true_ps, samples, weights)
            all_errs.append(err)

        util.io.save((all_errs, all_n_sims), res_file)

    return all_errs, all_n_sims


def get_err_sl(exp_desc, sim, seed):
    """
    Calculates the error for a given synth likelihood experiment.
    """

    assert isinstance(exp_desc.inf, ed.SynthLik_Descriptor)
    res_file = os.path.join(root, 'results/seed_'+str(seed), exp_desc.get_dir(), 'err')

    if os.path.exists(res_file + '.pkl'):
        err, n_sims = util.io.load(res_file)

    else:
        exp_dir = os.path.join(root, 'experiments/seed_'+str(seed), exp_desc.get_dir(), '0')

        if not os.path.exists(exp_dir):
            raise misc.NonExistentExperiment(exp_desc)

        samples, n_sims = util.io.load(os.path.join(exp_dir, 'results'))

        true_ps, _ = sim.get_ground_truth()
        err = calc_err(true_ps, samples)

        util.io.save((err, n_sims), res_file)

    return err, n_sims


def get_err_postprop(exp_desc, sim, seed):
    """
    Calculates the error for a given Post Prop experiment.
    """

    assert isinstance(exp_desc.inf, ed.PostProp_Descriptor)
    res_file = os.path.join(root, 'results/seed_'+str(seed), exp_desc.get_dir(), 'err')

    if os.path.exists(res_file + '.pkl'):
        all_prop_errs, post_err = util.io.load(res_file)

    else:
        exp_dir = os.path.join(root, 'experiments/seed_'+str(seed), exp_desc.get_dir(), '0')

        if not os.path.exists(exp_dir):
            raise misc.NonExistentExperiment(exp_desc)

        true_ps, _ = sim.get_ground_truth()

        all_proposals, posterior, _, _ = util.io.load(os.path.join(exp_dir, 'results'))
        all_prop_errs = []

        for i, proposal in enumerate(all_proposals[1:]):
            samples = proposal.gen(n_mcmc_samples, rng=rng)
            prop_err = calc_err(true_ps, samples)
            all_prop_errs.append(prop_err)

        samples = posterior.gen(n_mcmc_samples, rng=rng)
        post_err = calc_err(true_ps, samples)

        util.io.save((all_prop_errs, post_err), res_file)

    return all_prop_errs, post_err


def get_err_snpe(exp_desc, sim, seed):
    """
    Calculates the error for a given SNPE experiment.
    """

    assert isinstance(exp_desc.inf, ed.SNPE_MDN_Descriptor)
    res_file = os.path.join(root, 'results/seed_'+str(seed), exp_desc.get_dir(), 'err')

    if os.path.exists(res_file + '.pkl'):
        all_errs = util.io.load(res_file)

    else:
        exp_dir = os.path.join(root, 'experiments/seed_'+str(seed), exp_desc.get_dir(), '0')

        if not os.path.exists(exp_dir):
            raise misc.NonExistentExperiment(exp_desc)

        true_ps, _ = sim.get_ground_truth()

        all_posteriors, _, _, _ = util.io.load(os.path.join(exp_dir, 'results'))
        all_errs = []

        for posterior in all_posteriors[1:]:
            samples = posterior.gen(n_mcmc_samples, rng=rng)
            err = calc_err(true_ps, samples)
            all_errs.append(err)

        util.io.save(all_errs, res_file)

    return all_errs


def get_err_snl(exp_desc, sim, seed):
    """
    Calculates the error for a given SNL experiment.
    """

    assert isinstance(exp_desc.inf, ed.SNL_Descriptor)
    res_file = os.path.join(root, 'results/seed_'+str(seed), exp_desc.get_dir(), 'err')

    if os.path.exists(res_file + '.pkl'):
        all_errs = util.io.load(res_file)

    else:
        all_samples = get_samples_snl(exp_desc, sim, seed)

        all_errs = []

        true_ps, _ = sim.get_ground_truth()

        for samples in all_samples:
            err = calc_err(true_ps, samples)
            all_errs.append(err)

        util.io.save(all_errs, res_file)

    return all_errs


def get_sim(sim_name):
    """
    Returns the simulator object for a given simulator name.
    """

    if sim_name == 'lv':
        return misc.get_simulator('lotka_volterra')

    elif sim_name == 'hh':
        return misc.get_simulator('hodgkin_huxley')

    else:
        return misc.get_simulator(sim_name)


def plot_results(sim_name, run_name=''):
    """
    Plots all results for a given simulator.
    """

    sim = get_sim(sim_name)

    # SMC
    #exp_desc = ed.parse(util.io.load_txt('exps/{0}_smc.txt'.format(sim_name)))[0]
    #all_err_smc, all_n_sims_smc = get_err_smc(exp_desc, sim)

    # SL
    #all_err_slk = []
    #all_n_sims_slk = []
    #for exp_desc in ed.parse(util.io.load_txt('exps/{0}_sl.txt'.format(sim_name))):
    #    err, n_sims = get_err_sl(exp_desc, sim)
    #    all_err_slk.append(err)
    #    all_n_sims_slk.append(n_sims)

    # NDE
    #all_err_nde = []
    #all_n_sims_nde = []
    #for exp_desc in ed.parse(util.io.load_txt('exps/{0}_nl.txt'.format(sim_name))):
    #    all_err_nde.append(get_err_nde(exp_desc, sim))
    #    all_n_sims_nde.append(exp_desc.inf.n_samples)

    print 'sim: ' + str(sim_name)
    #print 'seeds: ' + str(seeds)

    if run_name == '.':
        run_name = ''

    fig, ax = plt.subplots(1, 1)
    seeds = np.arange(42, 52)
    all_errs_ppr, all_errs_snp, all_errs_snl, all_errs_snpc = [],[],[],[]

    ct = 0
    for i in range(len(seeds)):

        seed = seeds[i]

        all_err_ppr = None
        all_n_sims_ppr = None

        all_err_snp = None
        all_n_sims_snp = None

        all_err_snl = None
        all_n_sims_snl = None


        """
        # SMC
        exp_desc = ed.parse(util.io.load_txt('exps/{0}_smc.txt'.format(sim_name)))[0]
        if i==1:
            all_err_smc = np.nan*np.ones_like(all_err_smc)
            all_n_sims_smc = all_n_sims_smc
        else:
            all_err_smc, all_n_sims_smc = get_err_smc(exp_desc, sim, seed) 
            ct += 1
        if i == 0:
            ax.semilogx(all_n_sims_smc, all_err_smc, 'v:', color='y', label='SMC ABC')
        """

        for exp_desc in ed.parse(util.io.load_txt('exps/{0}_seq.txt'.format(sim_name))):

            # Post Prop
            if isinstance(exp_desc.inf, ed.PostProp_Descriptor):
                all_prop_err, post_err = get_err_postprop(exp_desc, sim, seed)
                all_err_ppr = all_prop_err + [post_err]
                all_n_sims_ppr = [(i + 1) * exp_desc.inf.n_samples_p for i in xrange(len(all_prop_err))]
                all_n_sims_ppr.append(all_n_sims_ppr[-1] + exp_desc.inf.n_samples_f)
                all_errs_ppr.append(all_err_ppr)

            # SNPE
            if isinstance(exp_desc.inf, ed.SNPE_MDN_Descriptor):
                all_err_snp = get_err_snpe(exp_desc, sim, seed)
                all_n_sims_snp = [(i + 1) * exp_desc.inf.n_samples for i in xrange(exp_desc.inf.n_rounds)]
                all_errs_snp.append(all_err_snp)

            # SNL
            if isinstance(exp_desc.inf, ed.SNL_Descriptor):
                all_err_snl = get_err_snl(exp_desc, sim, seed)
                all_n_sims_snl = [(i + 1) * exp_desc.inf.n_samples for i in xrange(exp_desc.inf.n_rounds)]
                all_errs_snl.append(all_err_snl)




        all_n_sims = np.concatenate([all_n_sims_ppr, all_n_sims_snp, all_n_sims_snl])
        min_n_sims = np.min(all_n_sims)
        max_n_sims = np.max(all_n_sims)



        # SNPE-C
        try:
            all_err_snpc = np.load('../lfi-experiments/snpec/notebooks_apt/results/'+sim_name+run_name+'/seed'+str(seed)+'/all_prop_errs_N5000.npy')
            all_n_sims_snpc = [(i + 1) * exp_desc.inf.n_samples for i in xrange(all_err_snpc.size)]
            all_errs_snpc.append(all_err_snpc)
        except:
            print ' could not load SNPE-C results, seed ' + str(seed)


    mean_err_snp = np.mean(np.vstack(all_errs_snp), axis=0)
    mean_err_snl = np.mean(np.vstack(all_errs_snl), axis=0)
    mean_err_snpc = np.nanmean(np.vstack(all_errs_snpc), axis=0)
    sd_err_snp = np.std(np.vstack(all_errs_snp), axis=0)
    sd_err_snl = np.std(np.vstack(all_errs_snl), axis=0)
    sd_err_snpc = np.nanstd(np.vstack(all_errs_snpc), axis=0)


    all_errs_ppr = [np.pad(all_errs_ppr[i], 
                    pad_width=(0, np.max( [mean_err_snl.size +1 - len(all_errs_ppr[i]),0])),
                    mode='constant', constant_values=np.nan) for i in range(len(all_errs_ppr))]


    mean_err_ppr = np.nanmean(np.vstack(all_errs_ppr), axis=0)

    sd_err_ppr = np.nanstd(np.vstack(all_errs_ppr), axis=0)

    #print('nan-counts', np.count_nonzero(~np.isnan(all_errs_snp), axis=0))
    #print('nan-counts', np.count_nonzero(~np.isnan(all_errs_snl), axis=0))
    #print('nan-counts', np.count_nonzero(~np.isnan(all_errs_snpc), axis=0))
    #print('nan-counts', np.count_nonzero(~np.isnan(np.vstack(all_errs_ppr)), axis=0))

    sd_err_snp /= np.sqrt(np.count_nonzero(~np.isnan(all_errs_snp), axis=0))
    sd_err_snl /= np.sqrt(np.count_nonzero(~np.isnan(all_errs_snl), axis=0))
    sd_err_snpc /= np.sqrt(np.count_nonzero(~np.isnan(all_errs_snpc), axis=0))
    sd_err_ppr /= np.sqrt(np.count_nonzero(~np.isnan(np.vstack(all_errs_ppr)), axis=0))

    matplotlib.rc('text', usetex=True)
    matplotlib.rc('font', size=16)

    if sim_name=='lv':
        all_n_sims_ppr = np.arange(1,12) * 1000
    elif sim_name=='gauss':
        all_n_sims_ppr = np.arange(1,42) * 1000
    else:
        all_n_sims_ppr = all_n_sims_snl if len(all_n_sims_ppr) < len(all_n_sims_snl) else all_n_sims_ppr
    print('mean_err_ppr', mean_err_ppr)

    #ax.semilogx(all_n_sims_smc, all_err_smc, 'v:', color='y', label='SMC ABC')
    #ax.semilogx(all_n_sims_slk, all_err_slk, 'D:', color='maroon', label='SL')
    ax.semilogx(all_n_sims_ppr, mean_err_ppr, '>:', color='c', label='SNPE-A')
    ax.semilogx(all_n_sims_snp, mean_err_snp, 'p:', color='g', label='SNPE-B')
    #ax.semilogx(all_n_sims_nde, all_err_nde, 's:', color='b', label='NL')
    ax.semilogx(all_n_sims_snl, mean_err_snl, 'o:', color='r', label='SNL')
    ax.semilogx(all_n_sims_snpc, mean_err_snpc, 'd-', color='k', label='APT')
    ax.fill_between(all_n_sims_ppr, mean_err_ppr-sd_err_ppr, mean_err_ppr+sd_err_ppr, color='c', alpha=0.3)
    ax.fill_between(all_n_sims_snp, mean_err_snp-sd_err_snp, mean_err_snp+sd_err_snp, color='g', alpha=0.3)
    ax.fill_between(all_n_sims_snl, mean_err_snl-sd_err_snl, mean_err_snl+sd_err_snl, color='r', alpha=0.3)
    ax.fill_between(all_n_sims_snpc, mean_err_snpc-sd_err_snpc, mean_err_snpc+sd_err_snpc, color='k', alpha=0.3)
    ax.set_xlabel('Number of simulations (log scale)')
    #ax.set_ticks([100, 1000], fontsize=14)
    ax.set_ylabel('$-$log probability of true parameters')
    ax.set_xlim([min_n_sims * 10 ** (-0.2), max_n_sims * 10 ** 0.2])
    #ax.legend(fontsize=14)
    plt.xticks([1000, 10000], fontsize=16)
    plt.show()

    print('ct', ct)

def main():

    parser = argparse.ArgumentParser(description='Plotting the results for the log likelihood experiment.')
    parser.add_argument('sim', type=str, choices=['gauss', 'mg1', 'lv', 'hh'], help='simulator')
    parser.add_argument('run', type=str, help='fitting options')
    #parser.add_argument('seed', type=int, help='seed')
    args = parser.parse_args()

    plot_results(args.sim, args.run)

if __name__ == '__main__':
    main()
