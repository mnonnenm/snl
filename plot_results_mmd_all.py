import os
import argparse

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import misc

import inference.mcmc as mcmc
import inference.diagnostics.two_sample as two_sample
import simulators.gaussian as sim
import experiment_descriptor as ed

import util.io
import util.math
import util.plot

root = misc.get_root()
rng = np.random.RandomState(42)

prior = sim.Prior()
model = sim.Model()
true_ps, obs_xs = sim.get_ground_truth()

# for mcmc
thin = 10
n_mcmc_samples = 5000
burnin = 100


def get_true_samples(seed):
    """
    Generates MCMC samples from the true posterior.
    """

    res_file = os.path.join(root, 'results/seed_'+str(seed), 'gauss', 'true_samples')

    if os.path.exists(res_file + '.pkl'):
        samples = util.io.load(res_file)

    else:
        log_posterior = lambda t: model.eval([t, obs_xs]) + prior.eval(t)

        sampler = mcmc.SliceSampler(true_ps, log_posterior, thin=thin)
        sampler.gen(burnin, rng=rng)  # burn in
        samples = sampler.gen(n_mcmc_samples, rng=rng)

        util.io.save(samples, res_file)

    return samples


def get_samples_nde(exp_desc, seed):
    """
    Generates MCMC samples for a given NDE experiment.
    """

    assert isinstance(exp_desc.inf, ed.NDE_Descriptor)
    res_file = os.path.join(root, 'results/seed_'+str(seed), exp_desc.get_dir(), 'mmd')

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


def get_samples_snl(exp_desc, seed):
    """
    Generates MCMC samples for a given SNL experiment.
    """

    assert isinstance(exp_desc.inf, ed.SNL_Descriptor)
    res_file = os.path.join(root, 'results/seed_'+str(seed), exp_desc.get_dir(), 'mmd')

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


def view_true_samples(seed):
    """
    Plots MCMC samples from the true posterior.
    """

    samples = get_true_samples(seed)

    fig = util.plot.plot_hist_marginals(samples, lims=sim.get_disp_lims(), gt=true_ps)
    fig.suptitle('true samples')

    plt.plot()


def view_samples_nde(seed):
    """
    Plots MCMC samples for all NDE experiments.
    """

    for exp_desc in ed.parse(util.io.load_txt('exps/gauss_nl.txt')):

        samples = get_samples_nde(exp_desc, seed)

        fig = util.plot.plot_hist_marginals(samples, lims=sim.get_disp_lims(), gt=true_ps)
        fig.suptitle('NDE, sims = {0}'.format(exp_desc.inf.n_samples))

    plt.plot()


def view_samples_snl(seed):
    """
    Plots MCMC samples for all SNL experiments.
    """

    for exp_desc in ed.parse(util.io.load_txt('exps/gauss_seq.txt')):

        if isinstance(exp_desc.inf, ed.SNL_Descriptor):

            all_samples = get_samples_snl(exp_desc, seed)

            for i, samples in enumerate(all_samples):
                fig = util.plot.plot_hist_marginals(samples, lims=sim.get_disp_lims(), gt=true_ps)
                fig.suptitle('SNL, round = {0}'.format(i + 1))

    plt.plot()


def view_samples_sl():
    """
    Plots MCMC samples for all synth likelihood experiments.
    """

    for exp_desc in ed.parse(util.io.load_txt('exps/gauss_sl.txt')):

        exp_dir = os.path.join(root, 'experiments', exp_desc.get_dir(), '0')
        samples, _ = util.io.load(os.path.join(exp_dir, 'results'))

        fig = util.plot.plot_hist_marginals(samples, lims=sim.get_disp_lims(), gt=true_ps)
        fig.suptitle('Synth Lik, sims = {0}'.format(exp_desc.inf.n_sims))

    plt.plot()


def get_mmd_nde(exp_desc, seed):
    """
    Calculates the MMD for a given NDE experiment.
    """

    assert isinstance(exp_desc.inf, ed.NDE_Descriptor)
    res_file = os.path.join(root, 'results/seed_'+str(seed), exp_desc.get_dir(), 'mmd')

    if os.path.exists(res_file + '.pkl'):
        err = util.io.load(res_file)

    else:
        samples = get_samples_nde(exp_desc, seed)
        true_samples = get_true_samples(seed)
        scale = util.math.median_distance(true_samples)
        err = two_sample.sq_maximum_mean_discrepancy(samples, true_samples, scale=scale)

        util.io.save(err, res_file)

    return err


def get_mmd_smc(exp_desc, seed):
    """
    Calculates the MMD for a given SMC ABC experiment.
    """

    assert isinstance(exp_desc.inf, ed.SMC_ABC_Descriptor)
    res_file = os.path.join(root, 'results/seed_'+str(seed), exp_desc.get_dir(), 'mmd')

    if os.path.exists(res_file + '.pkl'):
        all_errs, all_n_sims = util.io.load(res_file)

    else:
        exp_dir = os.path.join(root, 'experiments/seed_'+str(seed), exp_desc.get_dir(), '0')

        if not os.path.exists(exp_dir):
            raise misc.NonExistentExperiment(exp_desc)

        true_samples = get_true_samples(seed)
        scale = util.math.median_distance(true_samples)

        all_samples, all_log_weights, _, _, all_n_sims = util.io.load(os.path.join(exp_dir, 'results'))
        all_errs = []

        for samples, log_weights in zip(all_samples, all_log_weights):

            weights = np.exp(log_weights)
            err = two_sample.sq_maximum_mean_discrepancy(xs=samples, ys=true_samples, wxs=weights, scale=scale)
            all_errs.append(err)

        util.io.save((all_errs, all_n_sims), res_file)

    return all_errs, all_n_sims


def get_mmd_sl(exp_desc, seed):
    """
    Calculates the MMD for a given synth likelihood experiment.
    """

    assert isinstance(exp_desc.inf, ed.SynthLik_Descriptor)
    res_file = os.path.join(root, 'results/seed_'+str(seed), exp_desc.get_dir(), 'mmd')

    if os.path.exists(res_file + '.pkl'):
        err, n_sims = util.io.load(res_file)

    else:
        exp_dir = os.path.join(root, 'experiments/seed_'+str(seed), exp_desc.get_dir(), '0')

        if not os.path.exists(exp_dir):
            raise misc.NonExistentExperiment(exp_desc)

        samples, n_sims = util.io.load(os.path.join(exp_dir, 'results'))
        true_samples = get_true_samples(seed)
        scale = util.math.median_distance(true_samples)

        err = two_sample.sq_maximum_mean_discrepancy(samples, true_samples, scale=scale)

        util.io.save((err, n_sims), res_file)

    return err, n_sims


def get_mmd_postprop(exp_desc, seed):
    """
    Calculates the MMD for a given Post Prop experiment.
    """

    assert isinstance(exp_desc.inf, ed.PostProp_Descriptor)
    res_file = os.path.join(root, 'results/seed_'+str(seed), exp_desc.get_dir(), 'mmd')

    if os.path.exists(res_file + '.pkl'):
        all_prop_errs, post_err = util.io.load(res_file)

    else:
        exp_dir = os.path.join(root, 'experiments/seed_'+str(seed), exp_desc.get_dir(), '0')

        print(exp_dir)
        if not os.path.exists(exp_dir):
            raise misc.NonExistentExperiment(exp_desc)

        true_samples = get_true_samples(seed)
        scale = util.math.median_distance(true_samples)

        all_proposals, posterior, _, _ = util.io.load(os.path.join(exp_dir, 'results'))
        all_prop_errs = []

        for i, proposal in enumerate(all_proposals[1:]):
            samples = proposal.gen(n_mcmc_samples, rng=rng)
            prop_err = two_sample.sq_maximum_mean_discrepancy(samples, true_samples, scale=scale)
            all_prop_errs.append(prop_err)

        samples = posterior.gen(n_mcmc_samples, rng=rng)
        post_err = two_sample.sq_maximum_mean_discrepancy(samples, true_samples, scale=scale)

        util.io.save((all_prop_errs, post_err), res_file)

    return all_prop_errs, post_err


def get_mmd_snpe(exp_desc, seed):
    """
    Calculates the MMD for a given SNPE experiment.
    """

    assert isinstance(exp_desc.inf, ed.SNPE_MDN_Descriptor)
    res_file = os.path.join(root, 'results/seed_'+str(seed), exp_desc.get_dir(), 'mmd')

    if os.path.exists(res_file + '.pkl'):
        all_errs = util.io.load(res_file)

    else:
        exp_dir = os.path.join(root, 'experiments/seed_'+str(seed), exp_desc.get_dir(), '0')

        if not os.path.exists(exp_dir):
            raise misc.NonExistentExperiment(exp_desc)

        true_samples = get_true_samples(seed)
        scale = util.math.median_distance(true_samples)

        all_posteriors, _, _, _ = util.io.load(os.path.join(exp_dir, 'results'))
        all_errs = []

        for posterior in all_posteriors[1:]:
            samples = posterior.gen(n_mcmc_samples, rng=rng)
            err = two_sample.sq_maximum_mean_discrepancy(samples, true_samples, scale=scale)
            all_errs.append(err)

        util.io.save(all_errs, res_file)

    return all_errs


def get_mmd_snl(exp_desc, seed):
    """
    Calculates the MMD for a given SNL experiment.
    """

    assert isinstance(exp_desc.inf, ed.SNL_Descriptor)
    res_file = os.path.join(root, 'results/seed_'+str(seed), exp_desc.get_dir(), 'mmd')

    if os.path.exists(res_file + '.pkl'):
        all_errs = util.io.load(res_file)

    else:
        true_samples = get_true_samples(seed)
        scale = util.math.median_distance(true_samples)
        all_samples = get_samples_snl(exp_desc, seed)

        all_errs = []

        for samples in all_samples:
            err = two_sample.sq_maximum_mean_discrepancy(samples, true_samples, scale=scale)
            all_errs.append(err)

        util.io.save(all_errs, res_file)

    return all_errs


def plot_results(run_name=''):

    """
    # SMC
    exp_desc = ed.parse(util.io.load_txt('exps/gauss_smc.txt'))[0]
    all_mmd_smc, all_n_sims_smc = get_mmd_smc(exp_desc)

    # SL
    all_mmd_slk = []
    all_n_sims_slk = []
    for exp_desc in ed.parse(util.io.load_txt('exps/gauss_sl.txt')):
        mmd, n_sims = get_mmd_sl(exp_desc)
        all_mmd_slk.append(mmd)
        all_n_sims_slk.append(n_sims)

    # NDE
    all_mmd_nde = []
    all_n_sims_nde = []
    for exp_desc in ed.parse(util.io.load_txt('exps/gauss_nl.txt')):
        all_mmd_nde.append(get_mmd_nde(exp_desc))
        all_n_sims_nde.append(exp_desc.inf.n_samples)
    """

    #print 'seeds: ' + str(seeds)

    fig, ax = plt.subplots(1, 1)
    seeds = np.arange(42, 52)
    for seed in seeds:

        all_mmd_ppr = None
        all_n_sims_ppr = None

        all_mmd_snp = None
        all_n_sims_snp = None

        all_mmd_snl = None
        all_n_sims_snl = None

        for exp_desc in ed.parse(util.io.load_txt('exps/gauss_seq.txt')):

            # Post Prop
            if isinstance(exp_desc.inf, ed.PostProp_Descriptor):
                all_prop_mmd, post_mmd = get_mmd_postprop(exp_desc, seed)
                all_mmd_ppr = all_prop_mmd + [post_mmd]
                all_n_sims_ppr = [(i + 1) * exp_desc.inf.n_samples_p for i in xrange(len(all_prop_mmd))]
                all_n_sims_ppr.append(all_n_sims_ppr[-1] + exp_desc.inf.n_samples_f)

            # SNPE
            if isinstance(exp_desc.inf, ed.SNPE_MDN_Descriptor):
                all_mmd_snp = get_mmd_snpe(exp_desc, seed)
                all_n_sims_snp = [(i + 1) * exp_desc.inf.n_samples for i in xrange(exp_desc.inf.n_rounds)]

            # SNL
            if isinstance(exp_desc.inf, ed.SNL_Descriptor):
                all_mmd_snl = get_mmd_snl(exp_desc, seed)
                all_n_sims_snl = [(i + 1) * exp_desc.inf.n_samples for i in xrange(exp_desc.inf.n_rounds)]

        matplotlib.rc('text', usetex=True)
        matplotlib.rc('font', size=16)

        all_n_sims = np.concatenate([all_n_sims_ppr, all_n_sims_snp, all_n_sims_snl])
        min_n_sims = np.min(all_n_sims)
        max_n_sims = np.max(all_n_sims)
        
        all_mmd_snl = np.array(all_mmd_snl)

        #print all_mmd_snl        
        print np.array(all_mmd_snl).shape

        #ax.semilogx(all_n_sims_smc, np.sqrt(all_mmd_smc), 'v:', color='y', label='SMC ABC')
        #ax.semilogx(all_n_sims_slk, np.sqrt(all_mmd_slk), 'D:', color='maroon', label='SL')
        ax.semilogx(all_n_sims_ppr, np.sqrt(all_mmd_ppr), '>:', color='c', label='SNPE-A')
        ax.semilogx(all_n_sims_snp, np.sqrt(all_mmd_snp), 'p:', color='g', label='SNPE-B')
        #ax.semilogx(all_n_sims_nde, np.sqrt(all_mmd_nde), 's:', color='b', label='NL')
        ax.semilogx(all_n_sims_snl, np.sqrt(all_mmd_snl), 'o:', color='r', label='SNL')

        # SNPE-C
        try:
            all_mmd_snpc = np.load('../lfi_experiments/snpec/results/gauss_'+run_name + '/seed'+str(seed)+'/all_mmds_N5000.npy')
            all_n_sims_snpc = [(i + 1) * exp_desc.inf.n_samples for i in xrange(all_mmd_snpc.size)]
            ax.semilogx(all_n_sims_snpc, np.sqrt(all_mmd_snpc), 'd-', color='k', label='SNPE-C')
        except:
            print ' could not load SNPE-C results, seed ' + str(seed)


        ax.set_xlabel('Number of simulations (log scale)')
        ax.set_ylabel('Maximum Mean Discrepancy')
        ax.set_xlim([min_n_sims * 10 ** (-0.2), max_n_sims * 10 ** 0.2])
        ax.set_ylim([0.0, ax.get_ylim()[1]])
        if seed == seeds[0]:    
            ax.legend(fontsize=14)


        #samples = get_true_samples(seed)
        #np.save('/home/mackelab/Desktop/Projects/Biophysicality/code/lfi_experiments/snpec/results/gauss/seed'+str(seed)+'/samples', samples)

    plt.show()


def main():

    parser = argparse.ArgumentParser(description='Plotting the results for the MMD experiment.')
    parser.add_argument('sim', type=str, choices=['gauss'], help='simulator')
    parser.add_argument('run', type=str, help='fitting options')

    args = parser.parse_args()
    plot_results(args.run)

if __name__ == '__main__':
    main()
