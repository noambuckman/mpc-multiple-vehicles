import numpy as np
import casadi as cas


def minkowski_ellipse_collision_distance(ego_veh, ado_veh, x_ego, y_ego, phi_ego, x_ado, y_ado, phi_ado):
    """ Return the squared distance between the ego vehicle and ado vehicle
    for collision avoidance 
    Halder, A. (2019). On the Parameterized Computation of Minimum Volume Outer Ellipsoid of Minkowski Sum of Ellipsoids."
    """
    # if not numpy:
    shape_matrix_ego = np.array([[float(ego_veh.ax), 0.0], [0.0, float(ego_veh.by)]])
    shape_matrix_ado = np.array([[float(ado_veh.ax), 0.0], [0.0, float(ado_veh.by)]])

    rotation_matrix_ego = cas.vertcat(
        cas.horzcat(cas.cos(phi_ego), -cas.sin(phi_ego)),
        cas.horzcat(cas.sin(phi_ego), cas.cos(phi_ego)),
    )
    rotation_matrix_ado = cas.vertcat(
        cas.horzcat(cas.cos(phi_ado), -cas.sin(phi_ado)),
        cas.horzcat(cas.sin(phi_ado), cas.cos(phi_ado)),
    )

    # Compute the Minkowski Sum
    M_e_curr = cas.mtimes([rotation_matrix_ego, shape_matrix_ego])
    Q1 = cas.mtimes([M_e_curr, cas.transpose(M_e_curr)])

    M_a_curr = cas.mtimes([rotation_matrix_ado, shape_matrix_ado])
    Q2 = cas.mtimes([M_a_curr, cas.transpose(M_a_curr)])

    beta = cas.sqrt(cas.trace(Q1) / cas.trace(Q2))
    Q_minkowski = (1 + 1.0 / beta) * Q1 + (1.0 + beta) * Q2

    X_ego = cas.vertcat(x_ego, y_ego)
    X_ado = cas.vertcat(x_ado, y_ado)
    dist_squared = cas.mtimes([cas.transpose(X_ado - X_ego), cas.inv(Q_minkowski), (X_ado - X_ego)])

    return dist_squared
