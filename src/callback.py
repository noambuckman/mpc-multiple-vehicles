#
#     This file is part of CasADi.
#
#     CasADi -- A symbolic framework for dynamic optimization.
#     Copyright (C) 2010-2014 Joel Andersson, Joris Gillis, Moritz Diehl,
#                             K.U. Leuven. All rights reserved.
#     Copyright (C) 2011-2014 Greg Horn
#
#     CasADi is free software; you can redistribute it and/or
#     modify it under the terms of the GNU Lesser General Public
#     License as published by the Free Software Foundation; either
#     version 3 of the License, or (at your option) any later version.
#
#     CasADi is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#     Lesser General Public License for more details.
#
#     You should have received a copy of the GNU Lesser General Public
#     License along with CasADi; if not, write to the Free Software
#     Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
#
#
#! Callback
#! =====================
import casadi as cas
import numpy as np

#! In this example, we will demonstrate callback functionality for Ipopt.
#! Note that you need the fix https://github.com/casadi/casadi/wiki/enableIpoptCallback before this works
#!
#! We start with constructing the rosenbrock problem
# x=cas.SX.sym("x")
# y=cas.SX.sym("y")

# # f = (1-x)**2+100*(y-x**2)**2
# nlp={'x':cas.vertcat(x,y), 'f':f,'g':x+y}
# fcn = cas.Function('f', [x, y], [f])


class MyCallback(cas.Callback):
  def __init__(self, name, nx, ng, np, opts={}):
    cas.Callback.__init__(self)

    self.nx = nx
    self.ng = ng
    self.np = np

    self.x_sols = []
    self.y_sols = []
    self.lam_gs = []
    self.fs = []
    self.last_solution = None

    # Initialize internal objects
    self.construct(name, opts)

  def get_n_in(self): return cas.nlpsol_n_out()
  def get_n_out(self): return 1
  def get_name_in(self, i): return cas.nlpsol_out(i)
  def get_name_out(self, i): return "ret"

  def get_sparsity_in(self, i):
    n = cas.nlpsol_out(i)
    if n=='f':
      return cas.Sparsity. scalar()
    elif n in ('x', 'lam_x'):
      return cas.Sparsity.dense(self.nx)
    elif n in ('g', 'lam_g'):
      return cas.Sparsity.dense(self.ng)
    else:
      return cas.Sparsity(0,0)

  def eval(self, arg):
    # Create dictionary
    darg = {}
    for (i,s) in enumerate(cas.nlpsol_out()): darg[s] = arg[i]

    self.last_solution = darg
    # sol = darg['x']

    # self.x_sols.append(np.array(sol))
    # self.lam_gs.append(np.array(darg['lam_g']))
    # self.fs.append(np.array(darg['f']))

    return [0]

# mycallback = MyCallback('mycallback', 2, 1, 0)
# opts = {}
# opts['iteration_callback'] = mycallback
# opts['ipopt.tol'] = 1e-8
# opts['ipopt.max_iter'] = 50
# opts['ipopt.print_level'] = 5


# solver = cas.nlpsol('solver', 'ipopt', nlp, opts)
# sol = solver(lbx=-10, ubx=10, lbg=-10, ubg=10)

