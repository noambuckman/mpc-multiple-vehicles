import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.offsetbox as offsetbox
import os, psutil
import functools
import numpy as np
import scipy.ndimage as ndimage
import gc
import p_tqdm
from typing import List

from src.traffic_world import TrafficWorld
from src.vehicle import Vehicle


def get_car_color(i: int) -> str:
    ''' 
        Return color from prescribed list based on index 
        Input:  vehicle id
        Output: color name
    '''

    car_colors = ['green', 'blue', 'red', 'purple']
    return car_colors[i % len(car_colors)]


def add_car(x: np.array, vehicle: Vehicle, ax: plt.matplotlib.axis = None, car_name: str = "red", alpha: float = 1.0):
    ''' Adds an image of vehicle to axis 
        We assume images of cars are located in images/* directory

        Inputs:
            x:  Positions of ego vehicle
            vehicle:  Ego vehicle to be plotted
            ax:  Plotting axis
            car_name:  Used to determine color of car to be plotted
            alpha: See through value
        Output:
            axis: plt.matplotlib.axis
    '''

    #TODO:  Add check that red_car.png and ambulance.png exists

    if ax is None:
        # Create new Matplotlib figure
        fig, ax = plt.subplots(figsize=(12, 12))
    else:
        fig = ax.get_figure()

    # Load image of vehicle and recolor if required
    code_path = "/home/nbuckman/mpc-multiple-vehicles/src/utils/plotting/"
    if car_name == "red":
        arr_img = plt.imread(code_path + 'images/red_car.png', format='png')
        car_width_px = 599
        car_height_px = 310
    elif car_name == "purple":
        arr_img = plt.imread(code_path + 'images/red_car.png', format='png')
        arr_img[:, :, 2] = arr_img[:, :, 0]  #Add equal amount blue
        car_width_px = 599
        car_height_px = 310
    elif car_name == "blue":
        arr_img = plt.imread(code_path + 'images/red_car.png', format='png')
        arr_img[:, :, 2] = arr_img[:, :, 0]  # add blue
        arr_img[:, :, 0] = np.where(np.logical_and(0.2 < arr_img[:, :, 0], arr_img[:, :, 0] < .9), 0.0,
                                    arr_img[:, :, 0])  #remove red
        car_width_px = 599
        car_height_px = 310
    elif car_name == "Amb":
        arr_img = plt.imread(code_path + 'images/ambulance.png', format='png')
        car_width_px = 1280
        car_height_px = 640
    else:
        arr_img = plt.imread(code_path + 'images/green_car.png', format='png')
        car_width_px = 599
        car_height_px = 310
    arr_img[:, :, 3] = alpha * arr_img[:, :, 3]

    # Rotate + translate image to position of vehicle
    X, Y, Phi, _, _, _ = x.flatten()
    degree = np.rad2deg(Phi)
    rotated_img = ndimage.rotate(arr_img, degree)

    window_width = ax.get_xlim()[1] - ax.get_xlim()[0]
    window_height = ax.get_ylim()[1] - ax.get_ylim()[0]
    figwidth_in, figheight_in = fig.get_size_inches()

    # Hardcoded rescaling (TODO: automate this)
    dpi = fig.get_dpi()
    if dpi > 100:
        hard_coded_correction = 0.35
    else:
        hard_coded_correction = 0.75

    L = vehicle.L
    if car_name == "Amb":
        zoom_ratio = L / car_width_px * (dpi * figwidth_in) / window_width * hard_coded_correction
    else:
        zoom_ratio = L / car_width_px * (dpi * figwidth_in) / window_width * hard_coded_correction

    rotated_img = np.clip(rotated_img, 0.0, 1.0)
    imagebox = offsetbox.OffsetImage(rotated_img, zoom=zoom_ratio)  #this zoom is to scale L=1
    imagebox.image.axes = ax
    ab = offsetbox.AnnotationBbox(imagebox, (X, Y), frameon=False)
    ax.add_artist(ab)

    return ax


def add_ellipse(x, y, phi, vehicle, ax):
    a, b = vehicle.ax, vehicle.by
    ellipse_patch = patches.Ellipse((x, y), 2 * a, 2 * b, angle=np.rad2deg(phi), fill=False, color='black')
    ax.add_patch(ellipse_patch)

def plot_multiple_cars(k,
                       world: TrafficWorld,
                       vehicle: Vehicle,
                       xamb_plot: np.array,
                       xothers_plot: np.array,
                       folder: str,
                       car_plot_shape="Ellipse",
                       camera_positions=None,
                       car_labels: List[str] = None,
                       car_colors: List[str] = None,
                       xlim=None,
                       vid_track: int = 0,
                       all_other_vehicles: List[Vehicle] = None,
                       frame_width: float = 100.0):
    ''' This only has info from vehicle but not any individual ones
    vehicle: [Deprecated] 
    xamb_plot:  [Deprecated]
    xothers_plot:  This should be the trajectories of ALL vehicles
    folder: output folder for videos/imgs
    car_plot_shape:  Ellipse, Image
    camera_positions: List of camera positions
    car_labels:  Labels for annotating vehicles
    car_colors: List of car colors
    xlim:  Overriding the xlimits of the plot
    vid_track:  ID of vehicle to track with camera (used if camera_positions = None)
    all_other_vehicles: List of all vehicles
    '''
    assert xamb_plot or all_other_vehicles  # we either need an ambulance or the info about other vehicles

    if camera_positions is None:
        if xamb_plot is not None:
            camera_positions = [xamb_plot[0, 0] + t * vehicle.max_v * vehicle.dt for t in range(xamb_plot.shape[1])]
        else:
            camera_positions = [
                xothers_plot[vid_track][0, 0] +
                t * all_other_vehicles[vid_track].max_v * all_other_vehicles[vid_track].dt
                for t in range(xothers_plot[vid_track].shape[1])
            ]


    ymax = world.y_max
    ymin = world.y_min

    center_frame = camera_positions[k]
    axlim_minx, axlim_maxx = center_frame - 20, center_frame + 60,
    axlim_minx, axlim_maxx = center_frame - frame_width/2.0, center_frame + frame_width/2.0,

    if xlim is not None:
        axlim_minx = xlim[0]
        axlim_maxx = xlim[1]
    ## 1080p = 1920×1080,
    # fig, ax = plt.subplots(figsize=(figwidth_in, fig_height), dpi=144)
    # fig, ax = plt.subplots(figsize=(200, 20), dpi=144)
    fig, ax = plt.subplots(figsize=(1920 / 144, 1080 / 144), dpi=144)
    ax.axis('square')
    ax.set_ylim((ymin, ymax))
    ax.set_xlim((axlim_minx, axlim_maxx))

    add_lanes(ax, world)
    add_grass(ax, world)

    if car_plot_shape.lower() not in ["ellipse", "both", "ellipses", "image"]:
        raise Exception("Incorrect car_plot_shape")

    if car_plot_shape.lower() == "ellipse" or car_plot_shape.lower() == "both" or car_plot_shape.lower() == "ellipses":
        # Plot the ambulance as circles
        if xamb_plot is not None:
            x, y, phi = xamb_plot[0, k], xamb_plot[1, k], xamb_plot[2, k]
            a, b = vehicle.ax, vehicle.by
            ellipse_patch = patches.Ellipse((x, y), 2 * a, 2 * b, angle=np.rad2deg(phi), fill=False, color='black')
            ax.add_patch(ellipse_patch)

            if car_labels is not None:
                ax.annotate('R', xy=(xamb_plot[0, k:k + 1], xamb_plot[1, k:k + 1]))

        for i in range(len(xothers_plot)):

            x1_plot = xothers_plot[i]
            vehicle = all_other_vehicles[i]
            if 0.9 * axlim_minx <= x1_plot[0, k] <= 1.1 * axlim_maxx:
                x, y, phi = x1_plot[0, k], x1_plot[1, k], x1_plot[2, k]
                a, b = vehicle.ax, vehicle.by

                if car_colors is None:
                    color = get_car_color(i)
                else:
                    color = car_colors[i]
                ellipse_patch = patches.Ellipse((x, y),
                                                2 * a,
                                                2 * b,
                                                angle=np.rad2deg(phi),
                                                fill=False,
                                                edgecolor=color)
                ax.add_patch(ellipse_patch)
                if car_labels is None:
                    pass
                elif type(car_labels) == list:
                    ax.annotate(str(car_labels[i]), xy=(x, y))
                elif type(car_labels) == bool:
                    ax.annotate(str(all_other_vehicles[i].agent_id), xy=(x, y))

    if car_plot_shape.lower() == "image" or car_plot_shape.lower() == "both":
        for i in range(len(xothers_plot)):
            if 0.9 * axlim_minx <= xothers_plot[i][0, k] <= 1.1 * axlim_maxx:
                if car_colors is None:
                    color = get_car_color(i)
                else:
                    color = car_colors[i]
                if all_other_vehicles is not None:
                    ax = add_car(xothers_plot[i][:, k], all_other_vehicles[i], ax, color, alpha=1.0)
                else:
                    ax = add_car(xothers_plot[i][:, k], vehicle, ax, color, alpha=1.0)

        if xamb_plot is not None:
            ax = add_car(xamb_plot[:, k], vehicle, ax, "Amb")
    fig = plt.gcf()
    ax = plt.gca()
    ax.get_yaxis().set_visible(False)
    # ax.get_xaxis().set_visible(True)
    set_xticks_world(ax, interval = 50.0)

    plt.tight_layout()

    if folder is not None:
        fig.savefig(folder + '{:03d}.png'.format(k))
        plt.close(fig)
        gc.collect()
    else:
        plt.show()
        return fig, ax

def set_xticks_world(ax, interval = 10.0):
    xmin, xmax = ax.get_xlim()

    imin = int(xmin//interval)
    imax = int(xmax//interval)

    visual_buffer = 10.0
    xticks = [interval*i for i in range(imin+1, imax+1) if (xmin + visual_buffer)  <= interval*i <= (xmax - visual_buffer)]
    xlabels = ['%d'%x for x in xticks]

    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)

def plot_single_frame(world: TrafficWorld,
                      vehicle: Vehicle,
                      xamb_plot: np.array,
                      xothers_plot: List[np.array],
                      folder: str,
                      car_plot_shape: str = "Ellipse",
                      camera_speed: float = None,
                      plot_range: List[int] = None,
                      car_ids: List[int] = None,
                      xlims: bool = None):
    '''Plots the progression of all cars in one frame.
        Past poses are plotted with higher transparency (alpha)
    
        Inputs:

            vehicle: Provide a vehicle object so we can correctly scale the images
            xamb_plot: np.array  State of ambulance (or vehicle in center of plot)
            xothers_plot: list of other vehicle states
            car_plot_shape:  Type of shape used to plot the vehicles (ellipse or image or both)
            plot_range:  Subset of time indices to plot
            car_ids:  List of car names to plot
            xlims: We can override the width/xlimits of plot with xlims

    '''
    if camera_speed is None:
        camera_speed = vehicle.max_v

    figwidth_in = 12.0  #Hardcoded

    ymax = world.y_max
    ymin = world.y_min

    # Initialize the plotting frame to first position of the ambulance
    center_frame = xamb_plot[0, 0]
    if xlims is None:
        axlim_minx, axlim_maxx = center_frame - 40, center_frame + 100,
    else:
        axlim_minx, axlim_maxx = xlims[0], xlims[1]
    fig_height = np.ceil(1.1 * figwidth_in * (ymax - ymin) / (axlim_maxx - axlim_minx))
    fig, ax = plt.subplots(figsize=(figwidth_in, fig_height), dpi=144)
    ax.axis('square')
    ax.set_ylim((ymin, ymax))
    ax.set_xlim((axlim_minx, axlim_maxx))

    add_lanes(ax, world)
    add_grass(ax, world)

    if car_plot_shape.lower() not in ["ellipse", "both", "ellipses", "image", "dot"]:
        raise Exception("Incorrect car_plot_shape")

    if plot_range is None:
        plot_range = range(xamb_plot.shape[1])

    for ki in range(len(plot_range)):
        k = plot_range[ki]
        if len(plot_range) == 1:
            alpha_k = 1.0
        else:
            alpha_k = .5 + float(ki / (len(plot_range) - 1)) * (1 - .5)

        if car_plot_shape.lower() in {"ellipse", "both", "ellipses"}:

            # Plot an ellipse centered at position of vehicle
            x, y, phi = xamb_plot[0, k], xamb_plot[1, k], xamb_plot[2, k]
            a, b = vehicle.ax, vehicle.by
            ellipse_patch = patches.Ellipse((x, y),
                                            2 * a,
                                            2 * b,
                                            angle=np.rad2deg(phi),
                                            fill=True,
                                            color='black',
                                            alpha=alpha_k)
            ax.add_patch(ellipse_patch)

            for i in range(len(xothers_plot)):

                # Plot ellipse for other vehicles
                color = get_car_color(car_id)

                x1_plot = xothers_plot[i]
                x, y, phi = x1_plot[0, k], x1_plot[1, k], x1_plot[2, k]
                a, b = vehicle.ax, vehicle.by
                ellipse_patch = patches.Ellipse((x, y),
                                                2 * a,
                                                2 * b,
                                                angle=np.rad2deg(phi),
                                                fill=True,
                                                color=color,
                                                alpha=alpha_k)
                ax.add_patch(ellipse_patch)

                # Annotate cars with ids
                if car_ids is not None:
                    car_id = car_ids[i + 1]  # Plot index starts at 1
                    ax.annotate(str(car_id), (x, y))

        if car_plot_shape.lower() in {"image", "both"}:
            for i in range(len(xothers_plot)):
                x1_plot = xothers_plot[i]
                if car_ids is not None:
                    car_id = car_ids[i + 1]
                else:
                    car_id = i
                color = get_car_color(car_id)

                ax = add_car(x1_plot[:, k], vehicle, ax, color, alpha=alpha_k)

            ax = add_car(xamb_plot[:, k], vehicle, ax, "Amb", alpha=alpha_k)
        if car_plot_shape.lower() == "dot":
            for i in range(len(xothers_plot)):
                x1_plot = xothers_plot[i]
                if car_ids is not None:
                    car_id = car_ids[i + 1]
                else:
                    car_id = i
                color = get_car_color(car_id)

                ax.plot(x1_plot[0, k], x1_plot[1, k], '.', c=color, alpha=alpha_k)
            ax.plot(xamb_plot[0, k], xamb_plot[1, k], '.', c='black', alpha=alpha_k)

    fig = plt.gcf()
    if folder is not None:
        fig.savefig(folder + '{:03d}.png'.format(k))
        plt.close(fig)
        gc.collect()
    else:
        plt.show()


def plot_cars(world: TrafficWorld,
              vehicle: Vehicle,
              xamb_plot,
              xothers_plot,
              folder,
              car_plot_shape="ellipse",
              camera_speed=None,
              car_labels=None,
              car_colors=None,
              n_processors=8,
              vid_track: int = 0,
              all_other_vehicles: List[Vehicle] = None,
              xlim: List[float] = None,
              frame_width: float = 100.0):
    '''
    Note:  The vehicle should not be instantiated in a MPC Optimization or it may mess
            with the parallelization.  You can feed in a dummy vehicle Vehicle(dt) just
            for plotting
    '''
    if xamb_plot is not None:
        N = xamb_plot.shape[1]
    else:
        N = xothers_plot[0].shape[1]
    if psutil.virtual_memory().percent >= 90.0:
        raise Exception("Virtual Memory is too high, exiting to save computer")

    if xamb_plot is not None:
        camera_positions = generate_camera_positions(xamb_plot, vehicle)
    else:
        camera_positions = generate_camera_positions(xothers_plot[vid_track], all_other_vehicles[vid_track])
    if n_processors > 1:
        plot_partial = functools.partial(plot_multiple_cars,
                                         world=world,
                                         vehicle=vehicle,
                                         xamb_plot=xamb_plot,
                                         xothers_plot=xothers_plot,
                                         folder=folder,
                                         car_plot_shape=car_plot_shape,
                                         camera_positions=camera_positions,
                                         car_labels=car_labels,
                                         car_colors=car_colors,
                                         vid_track=vid_track,
                                         all_other_vehicles=all_other_vehicles,
                                         xlim=xlim,
                                         frame_width=frame_width)

        p_tqdm.p_map(plot_partial, range(N), num_cpus=n_processors)
    else:
        for k in range(N):
            plot_multiple_cars(k,
                               world,
                               vehicle,
                               xamb_plot,
                               xothers_plot,
                               folder,
                               car_plot_shape,
                               camera_positions,
                               car_labels,
                               car_colors,
                               vid_track=vid_track,
                               all_other_vehicles=all_other_vehicles,
                               xlim=xlim,
                               frame_width=frame_width)
    return None


def generate_camera_positions(xamb_plot: np.array, amb_veh: Vehicle) -> np.array:
    ''' Generate camera positions for plotting that follows the ego vehicle
        using a simple feedback controller. 

        Input:  
            ambulance state [np.array 2xN]:
            vehicle object: vehicle just so we can get dt TODO: possibly get this from somewhere else

        Output:  x-positions of the camera
    '''
    xc = [xamb_plot[0, 0]]
    xc_dot = [0]
    k_v = .5
    k_d = 0.1
    for k in range(1, xamb_plot.shape[1]):
        prev_xamb_position = xamb_plot[0, k - 1]
        prev_xc = xc[k - 1]

        if prev_xc < prev_xamb_position - 5.0:
            u = k_v * (xamb_plot[4, k - 1] - xc_dot[k - 1]) + k_d * (xamb_plot[0, k - 1] - xc[k - 1])
        elif prev_xc > prev_xamb_position + 5.0:
            u = k_v * (xamb_plot[4, k - 1] - xc_dot[k - 1]) + k_d * (xamb_plot[0, k - 1] - xc[k - 1])
        else:
            u = 0.0
        xc_dot_new = xc_dot[k - 1] + u
        xc_new = xc[k - 1] + xc_dot_new * amb_veh.dt
        xc += [xc_new]
        xc_dot += [xc_dot_new]

    return xc


def add_grass(ax: plt.matplotlib.axis, world: TrafficWorld, n_patches = 8):
    ''' 
        Generate grass in simulation

        Input:  
            axis:  Plotting axis
            world:  Simulation world object
    '''

    # Get boundary of lanes from world
    axlim_minx, axlim_maxx = ax.get_xlim()
    y_bottom_b, _, y_top_b = world.get_bottom_grass_y()
    y_bottom_t, _, y_top_t = world.get_top_grass_y()

    width = axlim_maxx - axlim_minx
    left_x_grass = axlim_minx
    bottom_y_grass = y_bottom_b
    # bottom_grass = patches.Rectangle((left_x_grass, y_bottom_b),
    #                                  width,
    #                                  y_top_b - y_bottom_b,
    #                                  facecolor='g',
    #                                  )
    # ax.add_patch(bottom_grass)

    # top_grass = patches.Rectangle((left_x_grass, y_bottom_t), width, y_top_t - y_bottom_t, facecolor='g', hatch='/')
    # ax.add_patch(top_grass)


    patterns = ["\\", 'x', None]
    colors = ['forestgreen']
    n_patterns = len(patterns)
    n_colors = len(colors)
    
    patch_length = (axlim_maxx - axlim_minx)/float(n_patches)
    
    patch_idx = int(axlim_minx // patch_length)    
    for pi in range(n_patches + 1):

        left_x_grass = patch_idx * patch_length
        if left_x_grass >= axlim_maxx:
            break        
        right_x_grass = (patch_idx + 1) * patch_length

        pattern = patterns[patch_idx % n_patterns]
        color = colors[patch_idx % n_colors]
    
        plot_left_x_grass = left_x_grass
        plot_right_x_grass = right_x_grass
        if right_x_grass >= axlim_maxx:
            plot_right_x_grass = axlim_maxx 
        if left_x_grass <= axlim_minx:
            plot_left_x_grass = axlim_minx

        width = plot_right_x_grass - plot_left_x_grass
            # left_x_grass = axlim_maxx - axlim_minx % (1.5 * width)
        bottom_contrast_grass = patches.Rectangle((plot_left_x_grass, bottom_y_grass),
                                                patch_length,
                                                y_top_b - y_bottom_b,
                                                facecolor=color,
                                                hatch=pattern,
                                                edgecolor='darkgreen',
                                                linewidth=0)
        ax.add_patch(bottom_contrast_grass)


        top_contrast_grass = patches.Rectangle((plot_left_x_grass, y_bottom_t),
                                        patch_length,
                                        y_top_t - y_bottom_t,
                                        facecolor=color,
                                        hatch=pattern,
                                        edgecolor='darkgreen',
                                        linewidth=0)
        ax.add_patch(top_contrast_grass)        

        patch_idx += 1




    # Create top grass
    y_bottom_t, y_center_t, y_top_t = world.get_top_grass_y()
    left_x_grass = axlim_minx
    bottom_y_grass = y_bottom_t
    width = axlim_maxx - axlim_minx

    top_grass = patches.Rectangle((left_x_grass, bottom_y_grass), width, y_top_t - y_bottom_t, facecolor='g', hatch='/')
    # ax.add_patch(top_grass)

    left_x_grass = axlim_maxx - axlim_minx % (1.5 * width)
    top_contrast_grass = patches.Rectangle((left_x_grass, bottom_y_grass),
                                           width / 2.0,
                                           y_top_t - y_bottom_t,
                                           facecolor='g',
                                           hatch='x')

    period = 2*patch_length
    left_x_grass = (axlim_minx // period) * period   
    for pi in range(n_patches + 3):

        right_x_grass = left_x_grass + patch_length
        if right_x_grass < axlim_minx:
            left_x_grass += period
            continue

        plot_left_x_grass = left_x_grass
        plot_right_x_grass = right_x_grass
        if right_x_grass >= axlim_maxx:
            plot_right_x_grass = axlim_maxx 
        if left_x_grass <= axlim_minx:
            plot_left_x_grass = axlim_minx

        width = plot_right_x_grass - plot_left_x_grass
        height = y_top_b - y_bottom_b
            # left_x_grass = axlim_maxx - axlim_minx % (1.5 * width)
        top_contrast_grass = patches.Rectangle((plot_left_x_grass, bottom_y_grass),
                                        width / 2.0,
                                        y_top_t - y_bottom_t,
                                        facecolor='g',
                                        hatch='x',
                                        linewidth=0)
        # ax.add_patch(top_contrast_grass)
        
        left_x_grass += period
        if left_x_grass >= axlim_maxx:
            break    
    
    
    
    
    
    
    # ax.add_patch(top_contrast_grass)


def add_lanes(ax: plt.matplotlib.axis, world: TrafficWorld):
    ''' 
        Add lane markings
        Input:  
            axis:
            world:
    '''
    xmin, xmax = ax.get_xlim()

    for lane_number in range(world.n_lanes_right):
        centerline_y = world.get_lane_centerline_y(lane_number, True)
        topline_y = centerline_y + world.lane_width / 2.0
        if lane_number == (world.n_lanes_right - 1):
            #TODO: Add yellow lines for two way road
            # ax.plot([xmin1, xmax1], [topline_y, topline_y], dashes=[10, 10], color='y')
            continue
        else:
            ax.plot([xmin, xmax], [topline_y, topline_y], dashes=[10, 10], color='0.5')

    for lane_number in range(world.n_lanes_left):
        if lane_number == world.n_lanes_left - 1:
            pass
        else:
            centerline_y = world.get_lane_centerline_y(lane_number, False)
            topline_y = centerline_y + world.lane_width / 2.0
            ax.plot([xmin, xmax], [topline_y, topline_y], dashes=[10, 10], color='0.5')


def animate(folder: str, vid_fname: str, fps: int = 16) -> str:
    ''' Execute command to generate animate .mp4 from .png frames '''
    cmd = 'ffmpeg -r {} -f image2 -i {}/%03d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p {}'.format(
        fps, folder, vid_fname)
    os.system(cmd)

    return vid_fname


def concat_imgs(folder: str) -> str:
    ''' create a single image with multiple images concatenated together '''
    from PIL import Image
    import glob

    im_0 = Image.open(folder + 'imgs/' '{:03d}.png'.format(0))
    buffer = int(0.1 * im_0.height)
    files = glob.glob(folder + 'imgs/*.png')
    n_timesteps = len(files)
    dst = Image.new('RGB', (im_0.width, (im_0.height + buffer) * n_timesteps))

    for k in range(n_timesteps):
        im_k = Image.open(folder + 'imgs/' '{:03d}.png'.format(k))
        dst.paste(im_k, (0, k * (im_0.height + buffer)))
    filename = folder + 'single.png'
    dst.save(filename, "PNG")
    return filename


def plot_initial_positions(log_dir: str,
                           world: TrafficWorld,
                           vehicles: List[Vehicle],
                           initial_positions: List[np.array],
                           number_cars_included: int = 10,
                           xlim = [0, 1000]):
    '''Plot and save initial conditions
    
        log_dir:  Path to log directory for experiment

    
    '''
    vehicles_to_plot = vehicles[:number_cars_included]
    x0_to_plot = initial_positions[:number_cars_included]
    x0_to_plot_reshaped = [x0.reshape(6, 1) for x0 in x0_to_plot]
    
    plot_cars(world,
              None,
              None,
              x0_to_plot_reshaped,
              log_dir,
              n_processors=1,
              all_other_vehicles=vehicles_to_plot,
              car_labels=True,
              xlim=xlim)
