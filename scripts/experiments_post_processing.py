import itertools
import os, pickle, json, glob, argparse
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

CONV_MS_TO_MPH = 2.23694


def post_process_experiments(args):
    ''' Load the directory and print the'''
    

    all_params, all_trajs, all_controls, all_vehicles, all_dirs, experiment_params = load_experiment_results(args.experiment_dir)

    describe_experiment_params(experiment_params)

    params = all_params[0]
    if args.n_mpc_end is None:
        args.n_mpc_end = all_trajs[0].shape[-1] - 1
    total_time = params["dt"] * args.n_mpc_end
    print("Total Sim Time %f     Rounds MPC: %d"%(total_time, args.n_mpc_end))

    summarize_sim_params(all_params, all_vehicles)
    summarize_sim_speeds(all_params, all_vehicles, "speed_summary_hist.png")
    # summarize_sim_params(all_params, all_vehicles, "speed_summary_hist.png")

    fixed_params, varying_params = describe_experiment_params(experiment_params)
    print("Fixed Params")
    print(fixed_params)
    print("Varying Params")
    print(varying_params)

    
    total_collisions, total_experiments, percentage_collisions_experiments, collisions_per_km, collisions_per_bkm = get_collision_stats(all_trajs)
    print("N_collisions: %d,  N_exp: %d, P_collisions: %.2f,  Collisions/Km: %.3e, Collisions/BKm %.1e"%(
                    total_collisions, total_experiments, percentage_collisions_experiments, collisions_per_km, collisions_per_bkm)) 


    if args.analyze_collisions:
        for exp_i in range(len(all_trajs)):
            if check_collision_experiment(all_trajs[exp_i]):
                print(all_dirs[exp_i])
                print("P", all_params[exp_i]["p_cooperative"])
                print("Density",  all_params[exp_i]["car_density"])
                print("Collision Index", get_collision_index(all_trajs[exp_i]))
                print("Seed", all_params[exp_i]["seed"])    

        
        collision_params, collision_expi = get_collision_parameters(all_params, all_trajs)
        summarize_sim_params(collision_params, all_vehicles, "collision_params_hist.png")
    
    if args.analyze_crazy_driving:
        for exp_i in range(len(all_trajs)):
            
            if check_out_of_y_bounds(all_trajs[exp_i], all_dirs[exp_i]):
                print("Crazy driver", exp_i, all_dirs[exp_i])

        out_of_y_bounds_params, out_of_y_bounds_expi = get_out_of_y_bounds_params(all_params, all_trajs, all_dirs)
        summarize_sim_params(out_of_y_bounds_params, all_vehicles, "outofbounds_params_hist.png")
        
        if args.analyze_collisions:
            collision_set = set(collision_expi)
            outofbounds_set = set(out_of_y_bounds_expi)    
            
            both_types = collision_set & outofbounds_set
            all_types = collision_set | outofbounds_set
            n_all_types = len(all_types)
            n_both = len(both_types)
            n_collision_only = len(collision_set) - n_both
            n_outofbounds_only = len(outofbounds_set) - n_both
            print("#Total: %d | # Collision Only %d,  # Out of Bounds Only %d,  # Both: %d"%(n_all_types, n_collision_only, n_outofbounds_only, n_both))

    if args.animate_by_density:
        animate_by_density(experiment_params, all_dirs, all_params)

    if args.animate_by is not None:
        param_name = args.animate_by
        assert param_name in experiment_params, "%s not in experiment.json"%param_name
        assert param_name in all_params[0], "%s not in params.json"%param_name

        animate_by(param_name, experiment_params, all_dirs, all_params)

    if args.animate_all:
        animate_by_all(experiment_params, all_dirs, all_params)

    if args.analyze_speed:
        save_fig = os.path.join(args.experiment_dir, "post_processing")
        analyze_speed(all_params, all_trajs, all_controls, all_dirs, args.aggregator, save_fig, p_baseline=args.p_baseline, remove_collision_seeds= args.remove_collision_seeds, ignore_out_of_y_bounds=args.ignore_out_of_y_bounds)


def analyze_speed(all_params, all_trajs, all_controls, all_dirs, aggregator="both", save_fig=True, p_baseline=0.0, remove_collision_seeds=False, ignore_out_of_y_bounds=True):
    ''' Compute the distance traveled for each experiment '''
    if save_fig:
        os.makedirs(save_fig, exist_ok=True)
        post_processing_fig_dir = save_fig

    if aggregator.lower() in ["all", "both", "median/mean"]:
        aggregators = ["mean", "median"]
    elif aggregator.lower() in ["median", "mean", "average", "avg", "med"]:
        aggregators = [aggregator]
    else:
        raise Exception("Invalid aggregator %s"%aggregator)

    collision_seeds = []
    collision_exp_i = []
    for exp_i in range(len(all_trajs)):
        if check_collision_experiment(all_trajs[exp_i]):
            collision_seeds.append(all_params[exp_i]["seed"])
            collision_exp_i.append(exp_i)
        if ignore_out_of_y_bounds and check_out_of_y_bounds(all_trajs[exp_i], all_dirs[exp_i]):
            print("Out of bounds:", all_params[exp_i]["seed"], all_params[exp_i]["p_cooperative"])
            collision_exp_i.append(exp_i)

    all_seeds = np.unique(np.array([all_params[exp_i]["seed"] for exp_i in range(len(all_params))]))
    all_p = np.unique(np.array([all_params[exp_i]["p_cooperative"] for exp_i in range(len(all_params))]))
    collision_seeds = np.unique(np.array(collision_seeds))
    print("Seeds with Collisions", collision_seeds)
    print("All Seeds", all_seeds)

    dt = all_params[0]["dt"]
    for average_agent in [True, False]:
        for aggregator in aggregators:

            controls = {}
            distances = {}
            total_time = dt * args.n_mpc_end
            
            highspeed_pseudoseeds = set([])
            lowspeed_pseudoseeds = set([])            

            cooperative_pseudoseeds = set([])
            noncooperative_pseudoseeds = set([])
            mixed_pseudoseed = set([])

            for exi in range(len(all_params)):
                if exi in collision_exp_i:
                    continue


                ds = (all_trajs[exi][:,0, args.n_mpc_end] - all_trajs[exi][:,0,0]) / total_time
                ctrl_sqrd = np.sum(np.sum(all_controls[exi]**2, axis=1),axis=1)
                
                p = all_params[exi]['p_cooperative']
                seed = all_params[exi]["seed"]

                current_log_dir = all_dirs[exi]
                
                agent_subset_type = None
                # agent_subset 
                num_agents = 6

                if agent_subset_type is None:
                    agent_subset = list(range(ds.shape[0])) # this will include all agents
                # elif agent_subset_type == "max_initial_speed":
                #     num_agents = 4
                #     agent_subset = get_max_initial_speed_agents(current_log_dir, num_agents)
                # elif agent_subset_type == "min_initial_speed":
                #     num_agents = 4
                #     agent_subset = get_max_initial_speed_agents(current_log_dir, num_agents)
                # elif agent_subset_type == "cooperative":
                #     agent_subset = get_cooperative_agents(current_log_dir)
                # elif agent_subset_type in ["egoistic", "noncooperative"]:
                #     agent_subset = get_noncooperative_agents(current_log_dir)
                # else:
                #     raise Exception("Invalid type of agent_subset_type")

                if agent_subset is not None:
                    ds = ds[agent_subset]

                if average_agent:
                    d = np.sum(ds) / ds.shape[0]
                    distances[(seed,p)] = d 

                    u = np.sum(ctrl_sqrd) / ctrl_sqrd.shape[0]
                    controls[(seed,p)] = u
                else:
                    cooperative_agents = get_cooperative_agents(current_log_dir)
                    noncooperative_agents = get_noncooperative_agents(current_log_dir)
                    lc = len(cooperative_agents)
                    lnc= len(noncooperative_agents)
                    for i in range(ds.shape[0]):
                        pseudoseed = int(seed + 1e9*i)
                        d = ds[i]
                        u = ctrl_sqrd[i]

                        distances[(pseudoseed,p)] = d
                        controls[(pseudoseed,p)] = u
                        if i in get_max_initial_speed_agents(current_log_dir, num_agents):
                            highspeed_pseudoseeds.add(pseudoseed)
                        if i in get_min_initial_speed_agents(current_log_dir, num_agents):
                            lowspeed_pseudoseeds.add(pseudoseed)

                        if i in mixed_pseudoseed:
                            pass
                        elif p>0.90 or p<0.10:
                            pass
                        else:
                            if i in cooperative_agents:
                                if pseudoseed in noncooperative_pseudoseeds:
                                    mixed_pseudoseed.add(pseudoseed) # this pseudoseed switched
                                    if pseudoseed in cooperative_pseudoseeds:
                                        cooperative_pseudoseeds.remove(pseudoseed)
                                else:
                                    cooperative_pseudoseeds.add(pseudoseed)
                            elif i in noncooperative_agents:
                                if pseudoseed in cooperative_pseudoseeds:
                                    mixed_pseudoseed.add(pseudoseed)
                                    if pseudoseed in noncooperative_pseudoseeds:
                                        noncooperative_pseudoseeds.remove(pseudoseed)
                                else:
                                    noncooperative_pseudoseeds.add(pseudoseed)

                        if seed in collision_seeds:
                            collision_seeds+=[pseudoseed]
            
            # No Baseline Comparison
            fig, ax = plt.subplots(1,1)
            p_populations = np.unique([k[1] for k in distances.keys()])
            average_distance_p = [np.mean([distances[k] for k in distances.keys() if k[1] == p]) for p in p_populations]
            
            std_error_distance_p = [1.96*np.std([distances[k] for k in distances.keys() if k[1] == p])/np.sqrt(len([distances[k] for k in distances.keys() if k[1] == p])) for p in p_populations]
                # plt.plot([p for _ in distances_p], [avg_d for avg_d in distances_p], '.')
            bottom_error = [average_distance_p[i] - std_error_distance_p[i] for i in range(len(average_distance_p))]
            top_error = [average_distance_p[i] + std_error_distance_p[i] for i in range(len(average_distance_p))]

            plt.fill_between(p_populations, bottom_error, top_error, alpha=0.25, color=args.plot_color, label="95% CI")
            plt.plot(p_populations, average_distance_p, '-o', color=args.plot_color, label="%s speed"%aggregator)
            plt.ylabel("Avg. Speed")
            plt.xlabel("% Cooperative")
            # perc_ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            perc_ticks = [0.0, 0.25, 0.50, 0.75, 1.0]

            plt.xticks(perc_ticks, ["%d%%"%(100*p) for p in perc_ticks])
            plt.legend()
            if save_fig:
                average_agent_str = "pop" if average_agent else "indiv"
                plt.savefig(os.path.join(post_processing_fig_dir, "distance_notbaselined_%s_%s.png"%(average_agent_str, aggregator)), dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.show()

            dist_baseline = {}
            controls_baseline = {}
            all_seeds = list(set([s for (s,p) in distances.keys()]))
            print("all seeds", all_seeds)
            for seed in all_seeds:
                if seed in collision_seeds and remove_collision_seeds:
                    continue

                if (seed, p_baseline) not in distances:
                    print("No p=%f in Seed: %d"%(p_baseline, seed))
                    for p in all_p:
                        if (seed,p) in distances:
                            print(seed, p)                    
                else:
                    for p in all_p:
                        if (seed,p) in distances:
                            dist_baseline[(seed,p)] = distances[(seed,p)] / (distances[(seed, p_baseline)] + 0.00000001)

                        if (seed,p) in controls:
                            controls_baseline[(seed,p)] = controls[(seed,p)] / (controls[(seed, p_baseline)] + 0.00000001)



            plt.figure()
            top_error = []
            average = []
            bottom_error = []

            for p in all_p:
                d_ratios = np.array([dist_baseline[(s,pi)] for (s,pi) in dist_baseline.keys() if pi==p])
                t_score = scipy.stats.t.ppf(q=0.975, df = (d_ratios.shape[0] - 1))
                print(p, d_ratios.shape)
                if aggregator.lower() in ["mean", "avg", "average"]:
                    d_ratios_mean = np.mean(d_ratios)
                    d_ratios_std = np.std(d_ratios)
                    d_ratios_stdE = t_score * d_ratios_std / np.sqrt(len(d_ratios))
                elif aggregator.lower() in ["median", "med"]:
                    d_ratios_mean = np.median(d_ratios)
                    iqr = np.subtract(*np.percentile(d_ratios, [75, 25]))
                    d_ratios_std = iqr
                    d_ratios_stdE = t_score*d_ratios_std / np.sqrt(len(d_ratios))                    
                else:
                    raise Exception("Not valid aggregator %s"%aggregator)


                top_error.append(d_ratios_mean + d_ratios_stdE)
                average.append(d_ratios_mean )
                bottom_error.append(d_ratios_mean - d_ratios_stdE)

                plt.plot([p]*len(d_ratios), d_ratios, '.', color='black')

            plt.plot(all_p, average, '-o', label="%s"%aggregator, color=args.plot_color)
            plt.fill_between(all_p, bottom_error, top_error, alpha=0.25, label="95% CI", color=args.plot_color)
            plt.ylabel("Speed Improvement")    
            plt.xlabel("% Cooperative")
            plt.ylim([0.98, 1.02])
            # perc_ticks = [0.0, 0.25, 0.50, 0.75, 1.0]
            plt.hlines(1.0, perc_ticks[0], perc_ticks[-1], linestyle='--', color='black')
            plt.xticks(perc_ticks, ["%d%%"%(100*p) for p in perc_ticks])
            plt.legend()            
            print("%s %s"%(average_agent_str, aggregator), average)

            if save_fig:
                plt.savefig(os.path.join(post_processing_fig_dir, "distance_performance_errorbars_%s_%s.png"%(average_agent_str, aggregator)), dpi=300)
                plt.savefig(os.path.join(post_processing_fig_dir, "distance_performance_errorbars_%s_%s.svg"%(average_agent_str, aggregator)))
                plt.close()
            else:
                plt.show()

            ######################## High Speed / Low Speed Plots
            if not average_agent:
                print(highspeed_pseudoseeds)
                print(lowspeed_pseudoseeds)
                plt.figure()
                top_error = []
                average = []
                bottom_error = []

                for p in all_p:
                    d_ratios = np.array([dist_baseline[(s,pi)] for (s,pi) in dist_baseline.keys() if pi==p and s in lowspeed_pseudoseeds])
                    print(p, d_ratios.shape)
                    t_score = scipy.stats.t.ppf(q=0.975, df = (d_ratios.shape[0] - 1))                    
                    if aggregator.lower() in ["mean", "avg", "average"]:
                        d_ratios_mean = np.mean(d_ratios)
                        d_ratios_std = np.std(d_ratios)
                        d_ratios_stdE = t_score * d_ratios_std / np.sqrt(len(d_ratios))
                    elif aggregator.lower() in ["median", "med"]:
                        d_ratios_mean = np.median(d_ratios)
                        iqr = np.subtract(*np.percentile(d_ratios, [75, 25]))
                        d_ratios_std = iqr
                        d_ratios_stdE = t_score*d_ratios_std / np.sqrt(len(d_ratios))                    
                    else:
                        raise Exception("Not valid aggregator %s"%aggregator)


                    top_error.append(d_ratios_mean + d_ratios_stdE)
                    average.append(d_ratios_mean )
                    bottom_error.append(d_ratios_mean - d_ratios_stdE)

                    plt.plot([p]*len(d_ratios), d_ratios, '.', color='black')

                plt.plot(all_p, average, '-o', label="Low Speed", color='orange')
                plt.fill_between(all_p, bottom_error, top_error, alpha=0.25, label="95% CI", color='orange')

                top_error = []
                average = []
                bottom_error = []
                for p in all_p:
                    d_ratios = np.array([dist_baseline[(s,pi)] for (s,pi) in dist_baseline.keys() if pi==p and s in highspeed_pseudoseeds])
                    t_score = scipy.stats.t.ppf(q=0.975, df = (d_ratios.shape[0] - 1))                    
                    if aggregator.lower() in ["mean", "avg", "average"]:
                        d_ratios_mean = np.mean(d_ratios)
                        d_ratios_std = np.std(d_ratios)
                        d_ratios_stdE = t_score * d_ratios_std / np.sqrt(len(d_ratios))
                    elif aggregator.lower() in ["median", "med"]:
                        d_ratios_mean = np.median(d_ratios)
                        iqr = np.subtract(*np.percentile(d_ratios, [75, 25]))
                        d_ratios_std = iqr
                        d_ratios_stdE = t_score*d_ratios_std / np.sqrt(len(d_ratios))                    
                    else:
                        raise Exception("Not valid aggregator %s"%aggregator)


                    top_error.append(d_ratios_mean + d_ratios_stdE)
                    average.append(d_ratios_mean )
                    bottom_error.append(d_ratios_mean - d_ratios_stdE)

                    plt.plot([p]*len(d_ratios), d_ratios, '.', color='black')

                plt.plot(all_p, average, '-o', label="High Speed", color='cyan')
                plt.fill_between(all_p, bottom_error, top_error, alpha=0.25, label="95% CI", color='cyan')


                plt.ylabel("Speed Improvement")    
                plt.xlabel("% Cooperative")
                plt.ylim([0.98, 1.02])
                # perc_ticks = [0.0, 0.25, 0.50, 0.75, 1.0]
                plt.hlines(1.0, perc_ticks[0], perc_ticks[-1], linestyle='--', color='black')
                plt.xticks(perc_ticks, ["%d%%"%(100*p) for p in perc_ticks])
                plt.legend()            
                print("%s %s"%(average_agent_str, aggregator), average)

                if save_fig:
                    plt.savefig(os.path.join(post_processing_fig_dir, "distance_performance_errorbars_%s_%s_highlowspeed.png"%(average_agent_str, aggregator)), dpi=300)
                    plt.savefig(os.path.join(post_processing_fig_dir, "distance_performance_errorbars_%s_%s_highlowspeed.svg"%(average_agent_str, aggregator)))
                    plt.close()
                else:
                    plt.show()

            ######################## Cooperative vs. Non-Cooperative
            if not average_agent:
                print(cooperative_pseudoseeds)
                print(noncooperative_pseudoseeds)
                plt.figure()
                top_error = []
                average = []
                bottom_error = []


                all_mixed_p = [p for p in all_p if 0.10<p<0.90]
                for p in all_mixed_p:
                    d_ratios = np.array([dist_baseline[(s,pi)] for (s,pi) in dist_baseline.keys() if pi==p and s in noncooperative_pseudoseeds])
                    t_score = scipy.stats.t.ppf(q=0.975, df = (d_ratios.shape[0] - 1))                    
                    if aggregator.lower() in ["mean", "avg", "average"]:
                        d_ratios_mean = np.mean(d_ratios)
                        d_ratios_std = np.std(d_ratios)
                        d_ratios_stdE = t_score * d_ratios_std / np.sqrt(len(d_ratios))
                    elif aggregator.lower() in ["median", "med"]:
                        d_ratios_mean = np.median(d_ratios)
                        iqr = np.subtract(*np.percentile(d_ratios, [75, 25]))
                        d_ratios_std = iqr
                        d_ratios_stdE = t_score*d_ratios_std / np.sqrt(len(d_ratios))                    
                    else:
                        raise Exception("Not valid aggregator %s"%aggregator)


                    top_error.append(d_ratios_mean + d_ratios_stdE)
                    average.append(d_ratios_mean )
                    bottom_error.append(d_ratios_mean - d_ratios_stdE)

                    plt.plot([p]*len(d_ratios), d_ratios, '.', color='black')

                plt.plot(all_mixed_p, average, '-o', label="Egoistic", color='red')
                plt.fill_between(all_mixed_p, bottom_error, top_error, alpha=0.25, label="95% CI", color='red')

                top_error = []
                average = []
                bottom_error = []
                for p in all_mixed_p:
                    d_ratios = np.array([dist_baseline[(s,pi)] for (s,pi) in dist_baseline.keys() if pi==p and s in cooperative_pseudoseeds])
                    t_score = scipy.stats.t.ppf(q=0.975, df = (d_ratios.shape[0] - 1))                    
                    if aggregator.lower() in ["mean", "avg", "average"]:
                        d_ratios_mean = np.mean(d_ratios)
                        d_ratios_std = np.std(d_ratios)
                        d_ratios_stdE = t_score * d_ratios_std / np.sqrt(len(d_ratios))
                    elif aggregator.lower() in ["median", "med"]:
                        d_ratios_mean = np.median(d_ratios)
                        iqr = np.subtract(*np.percentile(d_ratios, [75, 25]))
                        d_ratios_std = iqr
                        d_ratios_stdE = t_score*d_ratios_std / np.sqrt(len(d_ratios))                    
                    else:
                        raise Exception("Not valid aggregator %s"%aggregator)


                    top_error.append(d_ratios_mean + d_ratios_stdE)
                    average.append(d_ratios_mean )
                    bottom_error.append(d_ratios_mean - d_ratios_stdE)

                    plt.plot([p]*len(d_ratios), d_ratios, '.', color='black')

                plt.plot(all_mixed_p, average, '-o', label="Cooperative", color='blue')
                plt.fill_between(all_mixed_p, bottom_error, top_error, alpha=0.25, label="95% CI", color='blue')


                plt.ylabel("Speed Improvement")    
                plt.xlabel("% Cooperative")
                plt.ylim([0.98, 1.02])
                # perc_ticks = [0.0, 0.25, 0.50, 0.75, 1.0]
                plt.hlines(1.0, perc_ticks[0], perc_ticks[-1], linestyle='--', color='black')
                plt.xticks(perc_ticks, ["%d%%"%(100*p) for p in perc_ticks])
                plt.legend()            
                print("%s %s"%(average_agent_str, aggregator), average)

                if save_fig:
                    plt.savefig(os.path.join(post_processing_fig_dir, "distance_performance_errorbars_%s_%s_coopnoncoop.png"%(average_agent_str, aggregator)), dpi=300)
                    plt.savefig(os.path.join(post_processing_fig_dir, "distance_performance_errorbars_%s_%s_coopnoncoop.svg"%(average_agent_str, aggregator)))
                    plt.close()
                else:
                    plt.show()

            ####################### CONTROLS PLOT #############################33
            plt.figure()
            top_error = []
            average = []
            bottom_error = []

            for p in all_p:
                d_ratios = np.array([controls_baseline[(s,pi)] for (s,pi) in controls_baseline.keys() if pi==p])
                
                if aggregator.lower() in ["mean", "avg", "average"]:
                    d_ratios_mean = np.mean(d_ratios)
                    d_ratios_std = np.std(d_ratios)
                    d_ratios_stdE = d_ratios_std / len(d_ratios)
                elif aggregator.lower() in ["median", "med"]:
                    d_ratios_mean = np.median(d_ratios)
                    iqr = np.subtract(*np.percentile(d_ratios, [75, 25]))
                    d_ratios_std = iqr
                    d_ratios_stdE = d_ratios_std / len(d_ratios)                    
                else:
                    raise Exception("Not valid aggregator %s"%aggregator)

                top_error.append(d_ratios_mean + d_ratios_stdE)
                average.append(d_ratios_mean )
                bottom_error.append(d_ratios_mean - d_ratios_stdE)

                plt.plot([p]*len(d_ratios), d_ratios, '.', color='black')

            plt.plot(all_p, average, '-o', label="%s"%aggregator)
            plt.fill_between(all_p, bottom_error, top_error, alpha=0.25)

            plt.xlabel(r"$P_{cooperative}$")
            plt.ylabel("Controls Performance  (Controls Magnitude / Baseline Controls Magnitude)")    

            if save_fig:
                plt.savefig(os.path.join(post_processing_fig_dir, "controls_errorbars_%s_%s.png"%(average_agent_str,aggregator)), dpi=300)
                plt.savefig(os.path.join(post_processing_fig_dir, "controls_errorbars_%s_%s.svg"%(average_agent_str,aggregator)))

                plt.close()
            else:
                plt.show()

            
            if average_agent:
                seeds_p = dist_baseline.keys()
                for seed in all_seeds:
                    print(seed, sorted([sp[1] for sp in seeds_p if sp[0] == seed]))


    return None

def describe_experiment_params(experiment_params):
    ''' Print out a description of experiment params'''
    fixed_params = {}
    varying_params = {}

    for param_name, param_values in experiment_params.items():
        if isinstance(param_values, list) and len(param_values) > 1:
            varying_params[param_name] = param_values
        else:
            fixed_params[param_name] = param_values

    return fixed_params, varying_params


def load_experiment_results(experiment_dir):
    ''' Load experiment param and results'''
    sim_dirs = glob.glob(os.path.join(experiment_dir,'results/*-*/'))
    print("# Sim Dirs: %d"%len(sim_dirs))

    # experiment_params_path = glob.glob('/home/nbuckman/mpc_results/txe_0719/experiment.json')
    experiment_params_path = os.path.join(experiment_dir, "experiment.json")
    experiment_params = json.load(open(experiment_params_path,'rb'))
    
    all_params = []
    all_trajs = []
    all_vehicles = []
    all_dirs = []
    all_controls = []
    for sim_dir in sim_dirs:
        params_path = glob.glob(sim_dir + '/params.json')
        trajs_path = glob.glob(sim_dir + '/trajectories.npy')
        cntrls_path = glob.glob(sim_dir + '/controls.npy')
        vehicles_path = glob.glob(sim_dir + '/other_vehicles.p')
        
        if len(trajs_path) > 0:
            all_params += [json.load(open(params_path[0],'rb'))]
            all_trajs += [np.load(trajs_path[0])]
            all_vehicles += [pickle.load(open(vehicles_path[0],'rb'))]
            all_dirs += [sim_dir]
            all_controls += [np.load(cntrls_path[0])]
    print("# Simulations: %d"%len(all_dirs))

    return all_params, all_trajs, all_controls, all_vehicles, all_dirs, experiment_params


def animate_by_density(experiment_params, all_dirs, all_params):
    
    for density in experiment_params['car_density']:
        vids_to_concat = []
        for i in range(len(all_dirs)):
            sim_dir = all_dirs[i]
            if all_params[i]['car_density'] == density:
                vid_path = glob.glob(sim_dir + '/vids/*.mp4')
                
                if len(vid_path) > 0:
                    vids_to_concat += [vid_path[0]]
        output_file_name = concat_vids(vids_to_concat, "all_density%d.mp4"%density)
        print(output_file_name)


def animate_by(param_name, experiment_params, all_dirs, all_params):
    for param_value in experiment_params[param_name]:
        vids_to_concat = []
        for i, sim_dir in enumerate(all_dirs):
            if all_params[i][param_name] == param_value:
                vid_path = glob.glob(sim_dir + '/vids/*.mp4')
                if len(vid_path) > 0:
                    vids_to_concat += [vid_path[0]]

        output_file_name = concat_vids(vids_to_concat, "%s_%s.mp4"%(param_name, param_value))
        print(output_file_name)

def animate_by_all(experiment_params, all_dirs, all_params):
    _, varying_params = describe_experiment_params(experiment_params)

    os.makedirs("post_processing", exist_ok=True)
    varying_param_keys = list(varying_params.keys())
    list_of_param_values = [experiment_params[param] for param in varying_param_keys]
    print(varying_param_keys)

    grid_counter = 0
    for param_product in itertools.product(*list_of_param_values):
        vids_to_concat = []
        grid_params = {}
        for exp_i in range(len(all_dirs)):
            exp_dir = all_dirs[exp_i]
            exp_params = all_params[exp_i]
            experiment_matches = True

            for param_ix in range(len(varying_param_keys)):
                param_key = varying_param_keys[param_ix]
                param_value = param_product[param_ix]
                grid_params[param_key] = param_value    
                if float(exp_params[param_key]) != float(param_value):
                    experiment_matches = False
            
            if experiment_matches:
                vid_path = glob.glob(os.path.join(exp_dir, "vids/", "*.mp4"))
                print(vid_path)
                if len(vid_path) > 0:
                    vids_to_concat += [vid_path[0]]
                    

        print(param_product)
        print(vids_to_concat)
        
        if len(vids_to_concat) > 0:
            # grid_params = {k:exp_params[k] for k in varying_param_keys}
            
            grid_name = "%03d"%grid_counter
            json_path = os.path.join("post_processing", '%s.json'%grid_name)
            with open(json_path, 'w') as fp:
                json.dump(grid_params, fp)
            grid_counter += 1
            print(vids_to_concat)
            video_name = os.path.join("post_processing", "%s.mp4"%grid_name)
            print(vids_to_concat)
            output_file_name = concat_vids(vids_to_concat, video_name)
            # print(video_name)



        



def get_collision_stats(all_trajs):
    ''' Compute collision/km'''

    distances_array = np.array([all_trajs[exi][:,0,args.n_mpc_end] - all_trajs[exi][:,0,0] for exi in range(len(all_trajs))])
    total_distance_meters = np.sum(distances_array)
    total_collisions = np.sum([check_collision_experiment(traj) for traj in all_trajs])
    total_experiments = len(all_trajs)
    percentage_collisions_experiments = float(total_collisions) / total_experiments
    collisions_per_km = float(total_collisions) / (total_distance_meters / 1000.0) 
    collisions_per_bkm = collisions_per_km * (1e9)


    return total_collisions, total_experiments, percentage_collisions_experiments, collisions_per_km, collisions_per_bkm


def get_collision_parameters(all_params, all_trajs):
    ''' Return list of parameters corresponding to experiments with collisions'''

    collision_params = [all_params[exp_i] for exp_i in range(len(all_trajs)) if check_collision_experiment(all_trajs[exp_i])]    
    collision_expi = [exp_i for exp_i in range(len(all_trajs)) if check_collision_experiment(all_trajs[exp_i])]    
    return collision_params, collision_expi
    

def summarize_sim_speeds(all_params, all_vehicles, filename="speed_summary_hist.png"):
    ''' Create a histogram of vehicle speeds'''
    fig, ax = plt.subplots(1,1)
    all_max_speeds = [veh.max_v for vehicles in all_vehicles for veh in vehicles]
    if len(all_max_speeds) != 0:
        ax.hist(all_max_speeds)
    ax.set_xlabel('Max Speed')
    ax.set_ylabel('# Vehicles')
    # fig.suptitle("Experiment Settings")

    # fig.tight_layout()
    if filename is not None:
        plt.savefig(filename, bbox_inches="tight", dpi=300)
        plt.close
    else:
        fig.show()

def summarize_sim_params(all_params, all_vehicles, filename="param_summary_hist.png"):
    ''' Create a histogram of all the params'''
    
    fig, axs = plt.subplots(2,2)
    color=iter(plt.cm.rainbow(np.linspace(0,1,4)))
    counter = 0
    
    for param_key in ["seed", "car_density", "p_cooperative"]:
        row = counter%2
        col = counter // 2
        ax = axs[row, col]
        all_p_key = [param[param_key] for param in all_params]
        c = next(color)
        if len(all_p_key) != 0:
            ax.hist(all_p_key, facecolor=c)
        ax.set_xlabel(param_key)
        ax.set_ylabel("# Exp")
        counter+=1
        
    c = next(color)    
    all_max_speeds = [veh.max_v for vehicles in all_vehicles for veh in vehicles]
    if len(all_max_speeds) != 0:
        axs[1,1].hist(all_max_speeds)
    axs[1,1].set_xlabel('Max Speed')
    axs[1,1].set_ylabel('# Vehicles')
    fig.suptitle("Experiment Settings")

    fig.tight_layout()
    if filename is not None:
        plt.savefig(filename)
        plt.close
    else:
        fig.show()



def get_collision_index(traj):
    ''' Check for collision in experiment
        If coollision occurs, the trajectory will be zeros(1,6)
    '''
    collision_array = np.all(traj[:,:,1:] == np.zeros((1, 6, 1)), axis=1) # vehs x time
    collision_array = np.any(collision_array, axis=0) # time
    return np.argmax(collision_array)

def check_collision_experiment(traj):
    ''' Check for collision in experiment
        If coollision occurs, the trajectory will be zeros(1,6)
    '''
    if np.any(np.all(traj[:,:,1:] == np.zeros((1, 6, 1)), axis=1)):
        return True
    return False

def check_out_of_y_bounds(traj, dir):
    ''' Check for very wild driving typical of solver  errors'''
    WORLD_PATH = os.path.join(dir, "world.p")
    world = pickle.load(open(WORLD_PATH, "rb"))    
    
    return np.any(traj[:,1,:] <= world.y_min) or np.any(traj[:,1,:] >= world.y_max)

def get_out_of_y_bounds_params(all_params, all_trajs, all_dirs):

    out_of_y_bounds_params = [all_params[exp_i] for exp_i in range(len(all_trajs)) if check_out_of_y_bounds(all_trajs[exp_i], all_dirs[exp_i])]    
    out_of_y_bounds_expi = [exp_i for exp_i in range(len(all_trajs)) if check_out_of_y_bounds(all_trajs[exp_i], all_dirs[exp_i])]    


    return out_of_y_bounds_params, out_of_y_bounds_expi


def concat_vids(list_vids_to_concat, output_file, dry_run=False, n_max_vids=16, n_columns=2):
    ''' Takes a list of videos to concat and adds them to a single video'''
    
    n_vids = min(len(list_vids_to_concat), n_max_vids)
    n_rows = int(np.ceil(n_vids / n_columns))


    input_files = list_vids_to_concat[:n_vids]
    temp_files = []


    if os.path.exists(output_file):
        os.remove(output_file)

    for input_file in input_files:
        # Crop
        temp_file = input_file[:-4] + "temp.mp4"
        crop_command = "ffmpeg -i %s -filter 'crop=iw*.9:ih*.3' %s"%(input_file, temp_file)
        if not dry_run:
            os.system(crop_command)
        temp_files += [temp_file]


    all_filter_commands = ""
    for vint in range(len(input_files)):
        row = vint//2
        column = vint%2

        if vint==0:
            filter_command = "[%d:v]pad=iw*%d:ih*%d[int%d];"%(vint,n_columns, n_rows, vint)
        else:
            if column == 1:
                overlay_x = "W/2"
            else:
                overlay_x = "0"

            overlay_y = "%d/%d*H"%((row), n_rows)

            filter_command = "[int%d][%d:v]overlay=%s:%s[int%d];"%(vint-1, vint, overlay_x, overlay_y, vint)
        all_filter_commands += filter_command


    concat_command = "ffmpeg "
    for vint in range(len(input_files)):
        concat_command += "-i %s "%temp_files[vint]
    concat_command += "-filter_complex '%s' "%all_filter_commands[:-1]
    concat_command += "-map '[int%d]' -c:v libx264 -crf 23 -preset veryfast %s"%(vint, output_file)


    # concat_command = "ffmpeg -i %s -i %s -i %s -i %s -filter_complex '[0:v]pad=iw*2:ih*3[int];[int][1:v]overlay=W/2:H/3[vid];[vid][2:v]overlay=0:H/3[vid2]' -map '[vid2]' -c:v libx264 -crf 23 -preset veryfast %s"%(
    #                     *temp_files, output_file)
    print(concat_command)
    if not dry_run:
        os.system(concat_command)
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)    
    
    return output_file


def get_max_initial_speed_agents(current_log_dir, num_agents):
    ''' return the highest initial speed agents'''
    vehicle_path = os.path.join(current_log_dir, "other_vehicles.p")
    with open(vehicle_path, 'rb') as fp:
        vehicles = pickle.load(fp)
        neg_speeds = [-veh.max_v for veh in vehicles]
        speeds_idx_sorted = np.argsort(neg_speeds) # this will put largets speeds first 
        agent_indeces = speeds_idx_sorted[:num_agents]
    return agent_indeces

def get_min_initial_speed_agents(current_log_dir, num_agents):
    ''' return lowest initial speed agents'''

    vehicle_path = os.path.join(current_log_dir, "other_vehicles.p")
    with open(vehicle_path, 'rb') as fp:
        vehicles = pickle.load(fp)
        speeds = [veh.max_v for veh in vehicles] # this will put largest speeds first 
        speeds_idx_sorted = np.argsort(speeds) 
        agent_indeces = speeds_idx_sorted[:num_agents]

    return agent_indeces

def get_cooperative_agents(current_log_dir):
    ''' Return the idxs of cooperative agents'''
    vehicle_path = os.path.join(current_log_dir, "other_vehicles.p")
    with open(vehicle_path, 'rb') as fp:
        vehicles = pickle.load(fp)
        svos = [veh.theta_ij[-1] for veh in vehicles]
        agent_indeces = [i for i in range(len(vehicles)) if svos[i] > np.pi/6.0]

    return agent_indeces

def get_noncooperative_agents(current_log_dir):
    ''' Return the idxs of egoistic agents'''
    vehicle_path = os.path.join(current_log_dir, "other_vehicles.p")
    with open(vehicle_path, 'rb') as fp:
        vehicles = pickle.load(fp)
        svos = [veh.theta_ij[-1] for veh in vehicles]
        agent_indeces = [i for i in range(len(vehicles)) if svos[i] < np.pi/6.0]

    return agent_indeces

if __name__ == '__main__':

    # parser = argparse.ArgumentParser()
    # parser.add_argument("experiment_dir", type=str, help="experiment directory to analyze")
    # parser.add_argument("--n-mpc-end", type=int, default=None, help="experiment directory to analyze")

    # parser.add_argument("--analyze-speed", action="store_true", help="Analyze the speed of the cars")
    # parser.add_argument("--analyze-collisions", action="store_true", help="Print out parameters for every simulation with collisions")
    # parser.add_argument("--analyze-crazy-driving", action="store_true", help="Print out parameters for every simulation with solve errors")

    # parser.add_argument("--animate-by-density", action="store_true", help="animate videos and concatenate by SVO")
    # parser.add_argument("--animate-by", type=str, default=None, help="Paramater to organize animation grid")
    # parser.add_argument("--animate-all", action="store_true", help="Animation grid by all the parameters varied")

    # parser.add_argument("--aggregator", type=str, default="both", help="Compute median vs mean")
    # parser.add_argument("--plot-color", type=str, default=None, help="Color for plot")
    # parser.add_argument("--remove-collision-seeds", action="store_true", help="Should we throw out seeds where a collision occurs")
    # parser.add_argument("--p-baseline", type=float, default=0.0, help="What is the baseline performance comparing to. Default: 0.0 proportion cooperative")
    # parser.add_argument("--ignore-out-of-y-bounds", action="store_true", help="Should we measure distance traveled when a vehicle goes very off the road")
    # args = parser.parse_args()
    
    # if args.experiment_dir == ".":
    #     args.experiment_dir = os.getcwd()

    args = argparse.Namespace()
    args.n_mpc_end = None


    args.analyze_speed = True
    args.analyze_collisions = True
    args.analyze_crazy_driving = True

    args.animate_by_density = False
    args.animate_by = None
    args.animate_all = False

    args.aggregator = "both"
    args.remove_collision_seeds = False
    args.p_baseline = 0.00
    args.ignore_out_of_y_bounds = False  

    # Twenty Four Agents
    args.experiment_dir = "/mnt/two_tb_hdd/mpc_results/supercloud/010023_twentyfouragents_combined"
    args.plot_color = "green"
    post_process_experiments(args)

    # # Shared Control
    # args.experiment_dir = "/mnt/two_tb_hdd/mpc_results/supercloud/103122_nctrl2"
    # args.plot_color = "blue"
    # post_process_experiments(args)

    # # Higher Density
    # args.experiment_dir = "/mnt/two_tb_hdd/mpc_results/supercloud/110322_higherdensity"
    # args.plot_color = "red"
    # post_process_experiments(args)
