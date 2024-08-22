import argparse
import math
import time
import msprime
import statistics
import numpy as np
from matplotlib import pyplot as plt


class SNPSimulation:


    def __init__(self, inds, seq_len=200000, rate_map=None):
        self.inds     = inds
        self.seq_len  = seq_len
        self.sim_tree = None
        self.rate_map = rate_map

        self.dem_model, self.samples_dic, self.admixed_pop_id = SNPSimulation.__create_admixture_model(self.inds)


    def simulate_model(self):
        if self.rate_map:
            map = msprime.RateMap.read_hapmap(self.rate_map, sequence_length=self.seq_len)
        else:
            map = 2e-8

        tree = msprime.sim_ancestry(samples=self.samples_dic, demography=self.dem_model, ploidy=2,
                                    sequence_length=self.seq_len, recombination_rate=map)
        self.sim_tree  = msprime.sim_mutations(tree, rate=2e-8)
        self.positions = [x.site.position for x in self.sim_tree.variants()]

        #self.__set_admixed_list()


  ###################
  # LAI TRUE LABELS #
  ###################


  # Calculate ancestries for admixed individuals through GNN
  # Phyloseminar #96: Wilder Wohns
  # https://youtu.be/YrZTKjLzZY0?t=2788
    def __set_admixed_list(self):
        admixed_samples = self.sim_tree.samples(population=self.admixed_pop_id)

        self.admix_anc_list = []
        # For logging
        print('ADMIXTURE PREPROCESSING')
        num_of_trees = self.sim_tree.num_trees
        output_num   = num_of_trees // 10
        for i, tree in enumerate(self.sim_tree.trees()):
            if i % output_num == 0:
                print(f'Processing tree no. {i} / {num_of_trees}')
            sample_pop_dict = {}
            for sample in admixed_samples:
                sample_pop_dict[sample] = self.__get_gnn(tree, sample, tree.parent(sample))

            self.admix_anc_list.append((tree.interval.left, tree.interval.right, sample_pop_dict))


    def __get_gnn(self, tree, sample, parent):
        stk = list(tree.children(parent))
        gnn = []

        # DFS
        while stk:
            curr_node = stk.pop()
            curr_pop  = tree.population(curr_node)

            if curr_pop == tree.population(sample):
                continue

            children = tree.children(curr_node)

            if not children:
                gnn.append(curr_pop)
            else:
                stk.extend(children)

        if gnn:
            return statistics.mode(gnn)
        elif parent != tree.root:
            return self.__get_gnn(tree, sample, tree.parent(parent))
        else:
            assert False


    def __label_admixed_snps(self):
        self.admixed_snp_matrix = np.full((self.inds * 2, len(self.positions)), -1)

        tree_idx = 0
        for col, pos in enumerate(self.positions):
            while pos >= self.admix_anc_list[tree_idx][1]:
                tree_idx += 1

            assert(pos >= self.admix_anc_list[tree_idx][0] and pos < self.admix_anc_list[tree_idx][1])

            for row, ind in enumerate(self.sim_tree.samples(population=self.admixed_pop_id)):
                anc = self.admix_anc_list[tree_idx][2][ind]
                self.admixed_snp_matrix[row, col] = anc

        windows = []
        for row in self.admixed_snp_matrix:
            curr_window = 0
            for col in row:
                if col == 1:
                    curr_window += 1
                if col == 0 and curr_window != 0:
                    windows.append(curr_window)
                    curr_window = 0

            if curr_window != 0:
                windows.append(curr_window)

        bins = np.linspace(math.ceil(min(windows)), math.floor(max(windows)), 200)
        plt.xlim([min(windows)-5, max(windows)+5])

        plt.hist(windows, bins=bins)
        plt.title('Admixed tract length distribution')
        plt.xlabel('Tract length (SNPs, bin size = 5)')
        plt.ylabel('count')

        plt.savefig('window_distribution.png')

        print(statistics.median(windows))
        print(statistics.stdev(windows))
        print(statistics.mode(windows))
        print(len(windows))


    def __get_max_pop(self, idx, size, ind):
        return statistics.mode(self.admixed_snp_matrix[ind - (2 * 2 * self.inds), idx : idx + size])


  ###############
  # GENERIC I/O #
  ###############


    def __generate_windows(self, inds, snps, size, step, num):
        ind_pops = []
        for pop in range(2):
            for ind in self.sim_tree.samples(population=pop):
                ind_pops.append((ind, pop))

        for ind, pop in ind_pops:
            idx       = 0
            generated = 0
            while idx + size <= snps and generated != num:

                yield (ind, idx, size, pop, generated)

                if step == 0:
                    break

                idx       += step
                generated += 1


    def __generate_admixed_windows(self, snps, size, step, num):
        for ind in self.sim_tree.samples(population=2):
            idx       = 0
            generated = 0
            while idx + size <= snps and generated != num:
                pop = self.__get_max_pop(idx, size, ind)

                yield (ind, idx, size, pop, generated)

                if step == 0:
                    break

                idx       += step
                generated += 1


    def save_windows(self, size, step, num, win_pfx, snp_pfx, pos_fp, vcf_fp, cont=False):
        if not self.sim_tree:
            raise Exception('Simulation tree has not been created yet')

        '''
        gt_matrix  = self.__get_gt_matrix()
        total_snps = step * (num - 1) + size

        if len(gt_matrix[0]) < total_snps:
            raise Exception(f'Not enough SNPs ({len(gt_matrix[0])} < {total_snps})')

        print(f'Saving {len(gt_matrix[0])} SNPs from {len(gt_matrix)} individuals')
        print(f'Win. size = {size}, win. step = {step}, Num. of win. = {num}')

        # Write standard window data
        win_fos = [open(f'{win_pfx}_{i}.csv', 'w') for i in range(num)]
        for i, (ind, idx, size, pop, w_idx) in \
            enumerate(self.__generate_windows(len(gt_matrix), len(gt_matrix[0]), size, step, num)):
            win_fos[w_idx].write(f'{i},{ind},{idx},{size},{pop}\n')

        for fo in win_fos:
            fo.close()

        # Write admixed window data
        self.__label_admixed_snps()
        win_fos = [open(f'{win_pfx}_admix_{i}.csv', 'w') for i in range(num)]
        for i, (ind, idx, size, pop, w_idx) in \
            enumerate(self.__generate_admixed_windows(len(gt_matrix[0]), size, step, num)):
            win_fos[w_idx].write(f'{i},{ind},{idx},{size},{pop}\n')

        for fo in win_fos:
            fo.close()

        # Write SNP matrices
        for i in range(num):
            np.savetxt(f'{snp_pfx}_{i}.csv', gt_matrix[:self.inds * 2 * 2, i * step : (i * step) + size],
                fmt='%i', delimiter=',')
            np.savetxt(f'{snp_pfx}_admixed_{i}.csv', gt_matrix[self.inds * 2 * 2:, i * step : (i * step) + size],
                fmt='%i', delimiter=',')
        np.savetxt(f'{snp_pfx}_admixed_full.csv', self.admixed_snp_matrix, fmt='%i', delimiter=',')

        # Write position data
        with open(pos_fp, 'w') as pos_f:
            pos = list(map(str, self.__get_positions(cont)))
            pos_f.write(f'{",".join(pos)}\n')

        '''
        # Write VCF data
        ind_ids = []
        for pop in ['A', 'B']:
            for i in range(self.inds):
                ind_ids.append(f'{pop}_{i}')
        with open(vcf_fp, 'w') as out_f:
            self.sim_tree.write_vcf(out_f, contig_id='1', individual_names=ind_ids)


    def __get_gt_matrix(self):
        if not self.sim_tree:
            raise Exception('Simulation tree has not beeen created yet')

        return self.sim_tree.genotype_matrix().transpose()


    def __get_positions(self, cont=False):
        if not self.sim_tree:
            raise Exception('Simulation tree has not beeen created yet')

        if cont:
            return [x / self.seq_len for x in self.positions]
        else:
            return self.positions


  ######################
  # DEMOGRAPHIC MODELS #
  ######################


    def __create_admixture_model(inds):
        ret_model = msprime.Demography()

        ret_model.add_population(name='A',     initial_size=10000)
        ret_model.add_population(name='B',     initial_size=10000)
        ret_model.add_population(name='ADMIX', initial_size=10000)
        ret_model.add_population(name='AB',    initial_size=10000)

        ret_model.add_admixture(200, derived='ADMIX', ancestral=['A', 'B'], proportions=[9 / 10, 1 / 10])
        ret_model.add_population_split(500, derived=['A', 'B'], ancestral='AB')

        samples_dic = {'A': inds, 'B': inds}

        return ret_model, samples_dic, 2


def main():
    # Arguments
    parser = argparse.ArgumentParser(description='Data simulation for training.')
    parser.add_argument('--size',
        help='Window size (in SNPs)', default=1000, type=int)
    parser.add_argument('--windows',
        help='Maximum amount of windows to generate', default=100, type=int)
    parser.add_argument('--step',
        help='Window step (in SNPs)', default=1000, type=int)
    parser.add_argument('--inds',
        help='Number of individuals', default=1000, type=int)
    parser.add_argument('--chrom',
        help='Size of chromosome to simulate', default=5_000_000, type=int)
    parser.add_argument('--map',
        help='Path to genetic map', required=False)
    parser.add_argument('--path',
        help='Directory for simulated results', required=True)
    args = vars(parser.parse_args())

    win_size = args['size']
    win_step = args['step']
    win_amnt = args['windows']
    num_inds = args['inds']
    chr_size = args['chrom']
    rate_map = args['map']
    out_path = args['path']

    # Simulate data
    start_t = time.time()

    basic_sim = SNPSimulation(num_inds, chr_size, rate_map)
    basic_sim.simulate_model()

    elapsed_t = time.strftime('%H:%M:%S', time.gmtime(time.time() - start_t))
    print(f'Done with msprime simulation. Elapsed time = {elapsed_t}.')

    # Save results
    start_t = time.time()

    win_pfx = f'{out_path}/win'
    snp_pfx = f'{out_path}/snp'
    pos_fp  = f'{out_path}/pos.csv'
    vcf_fp  = f'{out_path}/all.vcf'
    basic_sim.save_windows(win_size, win_step, win_amnt, win_pfx, snp_pfx, pos_fp, vcf_fp)

    elapsed_t = time.strftime('%H:%M:%S', time.gmtime(time.time() - start_t))
    print(f'Done writing results. Elapsed time = {elapsed_t}.')


if __name__ == '__main__':
    main()
