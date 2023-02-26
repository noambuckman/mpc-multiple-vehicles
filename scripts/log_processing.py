import argparse
import pickle, os, sys
import numpy as np
import matplotlib.pyplot as plt

from post_processing import load_data_from_logdir
from src.utils.plotting.car_plotting import plot_problem_call, plot_solver_return

def log_post_process(args):
    ''' Post process the logs'''
    for log in args.logdir:
        SOLVER_LOG_FILE = os.path.join(log, "mpc_solver_logger.p")
        
        with open(SOLVER_LOG_FILE, "rb") as fp:
            solver_logs = pickle.load(fp)
        
        # print(solver_logs)

        params, vehicles, world, x0, x, u = load_data_from_logdir(log, midrun=args.midrun)    

        SOLVER_PLOT_DIR_NAME = os.path.join(log, "solver_plots")
        os.makedirs(SOLVER_PLOT_DIR_NAME, exist_ok=True)
        for sidx, solver_log in enumerate(solver_logs):
            
            if args.i_mpc is not None and solver_log.i_mpc not in args.i_mpc:
                continue
            if args.i_ibr is not None and solver_log.i_ibr not in args.i_ibr:
                continue
            if args.agent_id is not None and solver_log.agent_id not in args.agent_id:
                continue
            if args.i_solve is not None and solver_log.solve_i not in args.i_solve:
                continue

            print("MPC RD %d    IBR RD: %d   Agent ID: %d   Solve Iteration: %d"%(solver_log.i_mpc, solver_log.i_ibr, solver_log.agent_id, solver_log.solve_i))
            sol = solver_log.solver_return
            # print(vars(sol))
            if sol.solved_status == False:
                print("FALSE!!!")

            t_start = solver_log.i_mpc * params["number_ctrl_pts_executed"]
            t_end = (solver_log.i_mpc + 1) * params["number_ctrl_pts_executed"]

            ego_veh = vehicles[solver_log.agent_id]
            if args.plot:
                fig, axs = plot_solver_return(sol, ego_veh, world, plot_controls=args.plot_controls, problem_call = solver_log.problem_call)
                solver_string_title = "ri%06d_mpc%03d_ibr%03d_ag%02d_s%02d"%(sidx, solver_log.i_mpc, solver_log.i_ibr, solver_log.agent_id, solver_log.solve_i)
                fig.suptitle(solver_string_title)
                plt.savefig(os.path.join(SOLVER_PLOT_DIR_NAME, "%s_traj.png"%(solver_string_title)))
                plt.close()
                # plt.show()

                fig1, axs1, fig2, axs2 = plot_problem_call(solver_log.problem_call)
                fig1.savefig(os.path.join(SOLVER_PLOT_DIR_NAME, "%s_initial.png"%(solver_string_title)))
                fig2.savefig(os.path.join(SOLVER_PLOT_DIR_NAME, "%s_problemcall.png"%(solver_string_title)))
                plt.close('all')
                # plt.show()
            print("Solve Status:  %s   Cost: %f   Max CPU Limit: %s   g_violation: %f"%(sol.solved_status, sol.current_cost, sol.max_cpu_limit, sol.g_violation))



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("logdir", type=str, nargs="+", help="log directory to analyze")
    parser.add_argument("--midrun", action="store_true", help="Allow analyzing midrun")

    parser.add_argument("--agent-id", type=int, nargs="+", default=None)
    parser.add_argument("--i-mpc", type=int, nargs="+", default=None)
    parser.add_argument("--i-ibr", type=int, nargs="+", default=None)    
    parser.add_argument("--i-solve", type=int, nargs="+", default=None, help="If it had to solve multiple times")

    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-controls", action="store_true")

    args = parser.parse_args()

    log_post_process(args)

    # /mnt/two_tb_hdd/mpc_results/supercloud/092922_ctrl1_longer_safetyc/results/144d-7aa6-20220929-173833ell/