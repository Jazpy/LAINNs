import msprime
import argparse
import time
import json
from simulation import SNPSimulation

def main():
  # Arguments
  parser = argparse.ArgumentParser(description='Data simulation for training.')
  parser.add_argument('--size',
    help='Window size (in SNPs)', default=1000, type=int)
  parser.add_argument('--windows',
    help='Maximum amount of windows to generate', default=100, type=int)
  parser.add_argument('--step',
    help='Window step (in SNPs)', default=200, type=int)
  parser.add_argument('--inds',
    help='Number of individuals', default=3000, type=int)
  parser.add_argument('--chrom',
    help='Size of chromosome to simulate', default=5000000, type=int)
  parser.add_argument('--path',
    help='Directory for simulated results', required=True)
  args = vars(parser.parse_args())

  win_size = args['size']
  win_step = args['step']
  win_amnt = args['windows']
  num_inds = args['inds']
  chr_size = args['chrom']
  out_path = args['path']

  # Simulate data
  start_t = time.time()

  basic_sim = SNPSimulation(num_inds, chr_size, admixed=True)
  basic_sim.simulate_model()

  elapsed_t = time.strftime('%H:%M:%S', time.gmtime(time.time() - start_t))
  print(f'Done with msprime simulation. Elapsed time = {elapsed_t}.')

  # Save results
  start_t = time.time()

  win_pfx = f'{out_path}/win'
  snp_fp  = f'{out_path}/snp.csv'
  pos_fp  = f'{out_path}/pos.csv'
  con_fp  = f'{out_path}/config.txt'
  basic_sim.save_windows(win_size, win_step, win_amnt, win_pfx, snp_fp, pos_fp)

  # Simulation metadata
  config_dict = {
    'window_size'  : win_size,
    'window_step'  : win_step,
    'inds_per_pop' : num_inds,
    'window_num'   : win_amnt,
    'num_pops'     : 4
  }
  with open(f'{out_path}/config.txt', 'w') as out_f:
    json.dump(config_dict, out_f)

  elapsed_t = time.strftime('%H:%M:%S', time.gmtime(time.time() - start_t))
  print(f'Done writing results. Elapsed time = {elapsed_t}.')

if __name__ == '__main__':
  main()
