from argparse import ArgumentParser, ArgumentTypeError
import numpy as np


class IBRParser(ArgumentParser):
    def __init__(self):
        ArgumentParser.__init__(self, description="Run iterative best response with SVO")

        self.add_argument('--load-log-dir', type=str, default=None, help="Load log")
        self.add_argument('--log-subdir',
                          type=str,
                          default=None,
                          help="If you'd like to specify the log subdir name, end without /")
        self.add_argument('--n-processors',
                          type=int,
                          default=15,
                          help="Number of processors used when solving a single mpc")

        ## Simulation Settings
        self.add_argument('--mpc-start-iteration',
                          type=int,
                          default=0,
                          help="At which mpc iteration should the simulation start")
        self.add_argument('--save-solver-input', action='store_true')
        self.add_argument('--seed', type=int, default=None)
        self.add_argument('--car-density', type=int, default=5000, help="Car density across all lanes, cars per hour")
        self.add_argument('--n-other', type=int, default=10, help="Number of ado vehicles")
        self.add_argument('--n-lanes', type=int, default=2, help="Number of lanes in the right direction")
        self.add_argument('--random-svo', type=int, default=1, help="Randomly assign svo to other vehicles")
        self.add_argument('--svo-theta',
                          type=float,
                          default=0.0,
                          help="Setting a homogeneous svo, random-svo MUST be set as 0")
        self.add_argument('--n-mpc', type=int, default=100, help="Rounds of MPC to execute in the simulation")

        self.add_argument('--plot-flag', action='store_true')
        self.add_argument('--print-flag', action='store_true')

        # MPC Settings
        self.add_argument('--T', type=float, default=5.0, help="Planning horizon for MPC")
        self.add_argument('--dt', type=float, default=0.2, help="Time discretization for MPC")
        self.add_argument('--p-exec',
                          type=float,
                          default=0.4,
                          help="Percent of control points executed at each iteration of MPC")

        # IBR Settings
        self.add_argument('--n-ibr',
                          type=int,
                          default=3,
                          help="Number of rounds of iterative best response before excuting mpc")
        self.add_argument('--n-cntrld', type=int, default=2, help="How many cars does the response control in planning")
        self.add_argument('--rnds-shrd-cntrl', type=int, default=2)
        self.add_argument('--shrd-cntrl-scheduler', type=str, default="constant")

        self.add_argument('--k-solve-amb-max-ibr',
                          type=int,
                          default=2,
                          help="Max number iterations where ado solves for ambulance controls, afterwards only ado")
        self.add_argument('--plan-fake-ambulance', action='store_true')
        self.add_argument('--save-ibr',
                          type=str2bool,
                          default=False,
                          help="Save the IBR control inputs, 1=True, 0=False")
        self.add_argument('--save-state',
                          type=str2bool,
                          default=False,
                          help="Save the states at each round of MPC inputs")

        # MPC Solver & Cost Settings
        self.add_argument('--default-n-warm-starts',
                          type=int,
                          default=16,
                          help="Number of warm starts it will try when solving the MPC")
        self.add_argument('--k-max-slack',
                          type=float,
                          default=0.01,
                          help="Maximum allowed collision slack/overlap between vehicles")
        self.add_argument('--k-max-solve-number',
                          type=int,
                          default=3,
                          help="Max re-solve attempts of the mpc for an individual vehicle")
        self.add_argument('--k-max-round-with-slack',
                          type=int,
                          default=np.infty,
                          help="Max rounds of ibr with slack variable used [SEMI DEPRECATED]")

        self.add_argument('--k-slack-d', type=float, default=1000, help="Default constant for slack collision costs")
        self.add_argument('--k-CA-d', type=float, default=0.05, help="Default collision avoidance cost")
        self.add_argument('--k-CA-power', type=float, default=1.0, help="Default collision avoidance power")
        self.add_argument('--wall-CA', type=int, default=1, help="Add collision avoidance cost for approaching walls")
        self.add_argument('--print-level', type=int, default=0, help="Print level for IPOPT solver")
        self.add_argument('--k-lat', type=float, default=None, help="lateral cost for vehicles")

        self.add_argument('--k-politeness', type=float, default=None, help="parameter for IDM")

        self.add_argument('--safety-constraint', type=str2bool, default=False, help="Stopping constraint")
        self.add_argument('--plot-initial-positions', type=str2bool, default=True, help="Plot & save initial conditions")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')
