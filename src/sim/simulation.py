import msprime
import statistics
import numpy as np
from collections import deque

class SNPSimulation:

  def __init__(self, inds, seq_len=200000, dem_fp=None, admixed=False):
    self.inds     = inds
    self.dem_fp   = dem_fp
    self.seq_len  = seq_len
    self.sim_tree = None
    self.admixed  = admixed

    if not self.dem_fp:
      if not self.admixed:
        self.dem_model, self.samples_dic = \
          SNPSimulation.__create_basic_model(self.inds)
      else:
        self.dem_model, self.samples_dic = \
          SNPSimulation.__create_admixture_model(self.inds)

  def simulate_model(self):
    tree = msprime.sim_ancestry(samples=self.samples_dic,
      demography=self.dem_model, ploidy=1,
      sequence_length=self.seq_len, recombination_rate=2e-8)
    self.sim_tree  = msprime.sim_mutations(tree, rate=2e-8)
    self.positions = [x.site.position for x in self.sim_tree.variants()]

    if not self.admixed:
      return

    # Calculate ancestries for admixed individuals through GNN
    # Phyloseminar #96: Wilder Wohns
    # https://youtu.be/YrZTKjLzZY0?t=2788
    admixed_samples = self.sim_tree.samples(population=3)

    self.admix_anc_list = []
    # For logging
    num_of_trees = self.sim_tree.num_trees
    output_num   = num_of_trees // 20
    for i, tree in enumerate(self.sim_tree.trees()):
      if i % output_num == 0:
        print(f'Processing tree no. {i}/{num_of_trees}')
      sample_pop_dict = {}
      for sample in admixed_samples:
        sample_pop_dict[sample] = self.__get_gnn(tree, sample,
                                                 tree.parent(sample))

      self.admix_anc_list.append((tree.interval.left, tree.interval.right,
                                  sample_pop_dict))

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
        stk += list(children)

    if gnn:
      return statistics.mode(gnn)
    else:
      return self.__get_gnn(tree, sample, tree.parent(parent))

  def __get_max_pop(self, idx, size, ind):
    left  = self.positions[idx]
    right = self.positions[idx + size - 1]

    ancestries = []
    for admix_info in self.admix_anc_list:
      tree_left  = admix_info[0]
      tree_right = admix_info[1]
      tree_anc   = admix_info[2][ind]

      # No overlap
      if right < tree_left or left > tree_right:
        continue

      # Get endpoints
      end_l = max(tree_left,  left)
      end_r = min(tree_right, right)
      ancestries.append((end_r - end_l, tree_anc))

    anc_sum = {}
    for ancestry in ancestries:
      anc_sum[ancestry[1]] = anc_sum.get(ancestry[1], 0) + ancestry[0]

    return max(anc_sum, key=anc_sum.get)

  def __generate_admixed_windows(self, snps, size, step, num):
    if snps < size:
      raise Exception('Not enough SNPs')

    for ind in self.sim_tree.samples(population=3):
      idx       = 0
      generated = 0
      while idx + size <= snps and generated != num:
        pop = self.__get_max_pop(idx, size, ind)

        yield ind, idx, size, pop, generated

        if step == 0:
          break

        idx       += step
        generated += 1

  def __generate_windows(self, inds, snps, size, step, num):
    pop_labels = self.sim_tree.individuals_population

    if snps < size:
      raise Exception('Not enough SNPs')

    for ind, pop in zip(range(inds), pop_labels):
      # TODO: fix this hacky thing
      if pop == 3:
        return

      idx       = 0
      generated = 0
      while idx + size <= snps and generated != num:
        yield ind, idx, size, pop, generated

        if step == 0:
          break

        idx       += step
        generated += 1

  def save_windows(self, size, step, num, win_pfx, snp_fp, pos_fp, cont=False):
    if not self.sim_tree:
      raise Exception('Simulation tree has not beeen created yet')

    gt_matrix = self.__get_gt_matrix()
    print(f'Saving {len(gt_matrix[0])} SNPs from {len(gt_matrix)} individuals')
    print(f'Win. size = {size}, win. step = {step}, Num. of win. = {num}')

    # Write window data
    win_fos = [open(f'{win_pfx}_{i}.csv', 'w') for i in range(num)]
    for i, (ind, idx, size, pop, w_idx) in enumerate(self.__generate_windows(
        len(gt_matrix), len(gt_matrix[0]), size, step, num)):
      win_fos[w_idx].write(f'{i},{ind},{idx},{size},{pop}\n')
    for fo in win_fos:
      fo.close()

    # Write admixed window data
    win_fos = [open(f'{win_pfx}_admix_{i}.csv', 'w') for i in range(num)]
    for i, (ind, idx, size, pop, w_idx) in enumerate(
      self.__generate_admixed_windows(len(gt_matrix[0]), size, step, num)):
      win_fos[w_idx].write(f'{i},{ind},{idx},{size},{pop}\n')
    for fo in win_fos:
      fo.close()

    # Write SNP matrix
    # TODO: put this in a sparse matrix
    np.savetxt(snp_fp, gt_matrix, fmt='%i', delimiter=',')

    # Write position data
    with open(pos_fp, 'w') as pos_f:
      pos = list(map(str, self.__get_positions(cont)))
      pos_f.write(f'{",".join(pos)}\n')

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

  def __create_basic_model(inds):
    ret_model = msprime.Demography()

    ret_model.add_population(name='AFR',
      description='African', initial_size=14474)
    ret_model.add_population(name='EUR',
      description='European', initial_size=34039, growth_rate=0.0038)
    ret_model.add_population(name='EAS',
      description='East Asian', initial_size=45852, growth_rate=0.0048)
    ret_model.add_population(name='OOA',
      description='Bottleneck OOA', initial_size=1861)
    ret_model.add_population( name='AMH',
      description='Anatomically modern humans', initial_size=14474)
    ret_model.add_population(name='ANC',
      description='Ancestral equilibrium', initial_size=7310)

    ret_model.set_symmetric_migration_rate(['AFR', 'EUR'], 2.5e-5)
    ret_model.set_symmetric_migration_rate(['AFR', 'EAS'], 0.78e-5)
    ret_model.set_symmetric_migration_rate(['EUR', 'EAS'], 3.11e-5)

    ret_model.add_population_split(920,
      derived=['EUR', 'EAS'], ancestral='OOA')
    ret_model.add_symmetric_migration_rate_change( time=920,
      populations=['AFR', 'OOA'], rate=15e-5)
    ret_model.add_population_split(2040,
      derived=['OOA', 'AFR'], ancestral='AMH')
    ret_model.add_population_split(5920, derived=['AMH'], ancestral='ANC')

    samples_dic = {'AFR': inds, 'EUR': inds, 'EAS': inds}

    return ret_model, samples_dic

  def __create_admixture_model(inds):
    T_OOA = 920

    ret_model = msprime.Demography()

    ret_model.add_population(name="AFR",
      description="African", initial_size=14474)
    ret_model.add_population(name="EUR",
      description="European", initial_size=34039, growth_rate=0.0038)
    ret_model.add_population(name="EAS",
      description="East Asian", initial_size=45852, growth_rate=0.0048)
    ret_model.add_population(name="ADMIX",
      description="Admixed America", initial_size=54664, growth_rate=0.05)
    ret_model.add_population(name="OOA",
      description="Bottleneck out-of-Africa", initial_size=1861)
    ret_model.add_population(name="AMH",
      description="Anatomically modern humans", initial_size=14474)
    ret_model.add_population(name="ANC",
      description="Ancestral equilibrium", initial_size=7310)

    ret_model.set_symmetric_migration_rate(["AFR", "EUR"], 2.5e-5)
    ret_model.set_symmetric_migration_rate(["AFR", "EAS"], 0.78e-5)
    ret_model.set_symmetric_migration_rate(["EUR", "EAS"], 3.11e-5)

    ret_model.add_admixture(12, derived="ADMIX",
      ancestral=["AFR", "EUR", "EAS"], proportions=[1 / 6, 2 / 6, 3 / 6])

    ret_model.add_population_split(T_OOA,
      derived=["EUR", "EAS"], ancestral="OOA")
    ret_model.add_symmetric_migration_rate_change(time=T_OOA,
      populations=["AFR", "OOA"], rate=15e-5)
    ret_model.add_population_split(2040,
      derived=["OOA", "AFR"], ancestral="AMH")
    ret_model.add_population_split(5920,
      derived=["AMH"], ancestral="ANC")

    samples_dic = {'AFR': inds, 'EUR': inds, 'EAS': inds, 'ADMIX': inds}

    return ret_model, samples_dic
