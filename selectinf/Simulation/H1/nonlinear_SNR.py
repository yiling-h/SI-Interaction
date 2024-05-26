import time, sys, joblib, os
sys.path.append('/home/yilingh/SI-Interaction')
import multiprocessing as mp

from selectinf.Simulation.H1.nonlinear_H1_helpers import *

if __name__ == '__main__':
    # Get the script's directory
    script_directory = os.path.dirname(os.path.abspath(__file__))
    # Change the working directory to the script's directory
    os.chdir(script_directory)
    #argv = sys.argv
    ## sys.argv: [something, start, end, p_l, s_l, order, knots]
    #start, end = 0, 30#int(argv[1]), int(argv[2])
    #print("start:", start, ", end:", end)
    current_directory = os.getcwd()
    print("Current Working Directory:", current_directory)

    args = [(i * 2, (i + 1) * 2) for i in range(4)]
    with mp.Pool(processes=4) as pool:
        results = pool.starmap(vary_SNR, args)

    dir = 'Results/SNR/results.pkl'
    joblib.dump(results, dir)
