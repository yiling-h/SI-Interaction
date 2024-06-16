import time, sys, joblib, os
sys.path.append('/home/yilingh/SI-Interaction')
import multiprocessing as mp

from selectinf.Simulation.H1.nonlinear_H1_helpers import *

if __name__ == '__main__':
    # Get the script's directory
    script_directory = os.path.dirname(os.path.abspath(__file__))
    # Change the working directory to the script's directory
    os.chdir(script_directory)
    current_directory = os.getcwd()
    print("Current Working Directory:", current_directory)

    argv = sys.argv
    ## sys.argv: [something, start, end, ncores]
    #start, end, randomizer_scale = int(argv[1]), int(argv[2]), float(argv[3])
    #ncores = int(argv[4])
    start, end, randomizer_scale, ncores = 0, 8, 1.5, 4
    sim_per_process = int((end - start + 1) / ncores)
    print("start:", start, ", end:", end)
    print('nsim per process:', sim_per_process)

    # 2 sim per process, 4 processes in total
    args = [[start + i * sim_per_process,
             start + (i + 1) * sim_per_process,
             randomizer_scale] for i in range(ncores)]
    with mp.Pool(processes=ncores) as pool:
        results = pool.starmap(vary_main, args)

    dir = ('Results/main/results' + str(start) + '_' + str(end) + '_wh'
           + str(randomizer_scale) + '.pkl')
    joblib.dump(results, dir)
