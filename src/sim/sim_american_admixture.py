import sys
import msprime
import demes

graph = demes.load(sys.argv[1])
demography = msprime.Demography.from_demes(graph)

sample_dic = {}
ind_ids    = []
pops = ['MXB', 'MXL', 'CLM', 'PEL', 'PUR']
inds = 500

for pop in pops:
  sample_dic[pop] = inds

  for i in range(inds):
    ind_ids.append(f'{pop}_{i}')


sam_pop_train = open('train_pop_samples.txt', 'w')
sam_pop_train_tab = open('train_pop_samples_tab.txt', 'w')
sam_train = open('train_samples.txt', 'w')
sam_pop_test = open('test_pop_samples.txt', 'w')
sam_pop_test_tab = open('test_pop_samples_tab.txt', 'w')
sam_test = open('test_samples.txt', 'w')

for ind in ind_ids:
  toks = ind.split('_')

  if int(toks[1]) < inds * .90:
    sam_pop_train.write(f'{ind} {toks[0]}\n')
    sam_pop_train_tab.write(f'{ind}\t{toks[0]}\n')
    sam_train.write(f'{ind}\n')
  else:
    sam_pop_test.write(f'{ind} {toks[0]}\n')
    sam_pop_test_tab.write(f'{ind}\t{toks[0]}\n')
    sam_test.write(f'{ind}\n')

rate_map = msprime.RateMap.read_hapmap(sys.argv[2], sequence_length=10000000)

trees = msprime.sim_ancestry(sample_dic, sequence_length=10_000_000,
  ploidy=2, demography=demography, recombination_rate=rate_map)
trees = msprime.sim_mutations(trees, rate=2e-08)

with open('sim.vcf', 'w') as out_f:
  trees.write_vcf(out_f, contig_id='1', individual_names=ind_ids)
