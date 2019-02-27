import os
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import experiment_descriptor as ed
import misc

import util.io

root = misc.get_root()


def get_dist(exp_desc, average, seed):
    """
    Get the average distance from observed data in every round.
    """

    if average == 'mean':
        fname = 'dist_obs'
        avg_f = np.mean

    elif average == 'median':
        fname = 'dist_obs_median'
        avg_f = np.median

    else:
        raise ValueError('unknown average: {0}'.format(average))

    res_file = os.path.join(root, 'results/seed_'+str(seed), exp_desc.get_dir(), fname)

    if os.path.exists(res_file + '.pkl'):
        avg_dist = util.io.load(res_file)

    else:
        exp_dir = os.path.join(root, 'experiments/seed_'+str(seed), exp_desc.get_dir(), '0')

        _, obs_xs = util.io.load(os.path.join(exp_dir, 'gt'))
        results = util.io.load(os.path.join(exp_dir, 'results'))

        if isinstance(exp_desc.inf, ed.PostProp_Descriptor):
            _, _, _, all_xs = results

        elif isinstance(exp_desc.inf, ed.SNPE_MDN_Descriptor):
            _, _, all_xs, _ = results

        elif isinstance(exp_desc.inf, ed.SNL_Descriptor):
            _, all_xs, _ = results

        else:
            raise TypeError('unsupported experiment descriptor')

        avg_dist = []

        for xs in all_xs:
            dist = np.sqrt(np.sum((xs - obs_xs) ** 2, axis=1))
            dist = filter(lambda x: not np.isnan(x), dist)
            avg_dist.append(avg_f(dist))

        util.io.save(avg_dist, res_file)

    return avg_dist


def plot_results(sim_name, run_name, average):
    """
    Plots all results for a given simulator.
    """

    fig, ax = plt.subplots(1, 1)
    seeds = np.arange(42, 52)
    all_dists_ppr, all_dists_snp, all_dists_snl, all_dists_snpc = [],[],[],[]
    for i in range(len(seeds)):

        seed = seeds[i]
        all_dist_ppr = None
        all_dist_snp = None
        all_dist_snl = None

        for exp_desc in ed.parse(util.io.load_txt('exps/{0}_seq.txt'.format(sim_name))):

            # Post Prop
            if isinstance(exp_desc.inf, ed.PostProp_Descriptor):
                all_dists_ppr.append(get_dist(exp_desc, average, seed))

            # SNPE
            if isinstance(exp_desc.inf, ed.SNPE_MDN_Descriptor):
                all_dists_snp.append(get_dist(exp_desc, average, seed))

            # SNL
            if isinstance(exp_desc.inf, ed.SNL_Descriptor):
                all_dists_snl.append(get_dist(exp_desc, average, seed))

        # SNPE-C
        try:
            all_dists_snpc.append(np.load('../lfi-experiments/snpec/notebooks_apt/results/'+sim_name+run_name+'/seed'+str(seed)+'/avg_dist.npy'))
        except:
            print ' could not load SNPE-C results, seed ' + str(seed)

    all_dists_snpc = [all_dists_snpc[i][:len(all_dists_snl[i])] for i in range(len(all_dists_snpc))]

    mean_dist_snp = np.mean(np.vstack(all_dists_snp), axis=0)
    mean_dist_snl = np.mean(np.vstack(all_dists_snl), axis=0)
    mean_dist_snpc = np.mean(np.vstack(all_dists_snpc), axis=0)
    sd_dist_snp = np.std(np.vstack(all_dists_snp), axis=0)
    sd_dist_snl = np.std(np.vstack(all_dists_snl), axis=0)
    sd_dist_snpc = np.std(np.vstack(all_dists_snpc), axis=0)

    all_dists_ppr = [np.pad(all_dists_ppr[i], 
                    pad_width=(0, np.max( [mean_dist_snl.size +1 - len(all_dists_ppr[i]),0])),
                    mode='constant', constant_values=np.nan) for i in range(len(all_dists_ppr))]
    mean_dist_ppr = np.nanmean(np.vstack(all_dists_ppr), axis=0)
    sd_dist_ppr = np.nanstd(np.vstack(all_dists_ppr), axis=0)

    #print('nan-counts', np.count_nonzero(~np.isnan(all_dists_snp), axis=0))
    #print('nan-counts', np.count_nonzero(~np.isnan(all_dists_snl), axis=0))
    #print('nan-counts', np.count_nonzero(~np.isnan(all_dists_snpc), axis=0))
    #print('nan-counts', np.count_nonzero(~np.isnan(np.vstack(all_dists_ppr)), axis=0))

    sd_dist_snp /= np.sqrt(np.count_nonzero(~np.isnan(all_dists_snp), axis=0))
    sd_dist_snl /= np.sqrt(np.count_nonzero(~np.isnan(all_dists_snl), axis=0))
    sd_dist_snpc /= np.sqrt(np.count_nonzero(~np.isnan(all_dists_snpc), axis=0))
    sd_dist_ppr /= np.sqrt(np.count_nonzero(~np.isnan(np.vstack(all_dists_ppr)), axis=0))

    matplotlib.rc('text', usetex=True)
    matplotlib.rc('font', size=16)

    ax.plot(np.arange(len(all_dists_ppr[-1])) + 1, mean_dist_ppr, '>:', color='c', label='SNPE-A')
    ax.plot(np.arange(len(all_dists_snp[-1])) + 1, mean_dist_snp, 'p:', color='g', label='SNPE-B')
    ax.plot(np.arange(len(all_dists_snl[-1])) + 1, mean_dist_snl, 'o:', color='r', label='SNL')
    ax.plot(np.arange(len(all_dists_snpc[-1])) + 1, mean_dist_snpc, 'd-', color='k', label='APT')

    ax.fill_between(np.arange(len(all_dists_ppr[-1])) + 1, mean_dist_ppr-sd_dist_ppr, mean_dist_ppr+sd_dist_ppr, color='c', alpha=0.3)
    ax.fill_between(np.arange(len(all_dists_snp[-1])) + 1, mean_dist_snp-sd_dist_snp, mean_dist_snp+sd_dist_snp, color='g', alpha=0.3)
    ax.fill_between(np.arange(len(all_dists_snl[-1])) + 1, mean_dist_snl-sd_dist_snl, mean_dist_snl+sd_dist_snl, color='r', alpha=0.3)
    ax.fill_between(np.arange(len(all_dists_snpc[-1])) + 1, mean_dist_snpc-sd_dist_snpc, mean_dist_snpc+sd_dist_snpc, color='k', alpha=0.3)


    ax.set_xlabel('Round')
    ax.set_ylabel('{0} distance'.format(average[0].upper() + average[1:]))
    ax.set_ylim([0.0, ax.get_ylim()[1]])

    ax.legend(fontsize=14)

    plt.show()


def main():

    parser = argparse.ArgumentParser(description='Plotting distance vs time for the attention-focusing experiments.')
    parser.add_argument('sim', type=str, choices=['gauss', 'mg1', 'lv', 'hh'], help='simulator')
    parser.add_argument('run', type=str, help='fitting options')
    parser.add_argument('-a', '--average', type=str, choices=['mean', 'median'], default='median', help='average type')
    args = parser.parse_args()

    plot_results(args.sim, args.run, args.average)


if __name__ == '__main__':
    main()
