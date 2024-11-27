import sys
import os

# Add the directory of the stride module to sys.path
sys.path.append(os.path.abspath('../stride'))
sys.path.append(os.path.abspath('../mosaic'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

from stride import *
from stride.utils import wavelets

# from utils import analytical_2d


async def main(runtime):
    # transducer coordinates
    geo_scale = 10                   # geometry scaling factor
    trans_rad = 0.35/geo_scale      # transducer array radius, [m]
    N_trans = 10 # np.round(140*42/trans_rad)            # number of transducers
    angle_vec = np.linspace(0,2*np.pi,N_trans+1);   # angle vector, [rad]
    angle_vec = angle_vec[:-1]; 
    # Calculate transducer positions
    trans_x = trans_rad * np.cos(angle_vec)
    trans_y = trans_rad * np.sin(angle_vec)

    # Create the grid
    dx = 0.1e-3    # grid step, [m]
    dy = 0.1e-3    # grid step, [m]
    Nx = np.round(trans_rad*2/dx)       # number of grid points
    Ny = np.round(trans_rad*2/dy);
    shape = (Nx, Ny)
    extra = (50, 50)
    absorbing = (40, 40)
    spacing = (dx, dy)

    space = Space(shape=shape,
                  extra=extra,
                  absorbing=absorbing,
                  spacing=spacing)


    # Create time
    fs = 20e6                   # sampling frequency, [Hz]
    vw = 1500.                  # wave speed, [m/s]
    dt = 1/fs                   # time step, [s]
    Nt = np.round(trans_rad/vw*3.5*fs)  # number of time steps
    t = np.linspace(0, Nt*dt, Nt)       # time vector, [s]
    cfl_num = vw*dt/dx          # CFL number, needs to be smaller than 1
    print(f"CFL number: {cfl_num}")    # Display the value of cfl_num

    time = Time(start=t[0],
                step=dt,
                num=Nt)


    # Create problem
    problem = Problem(name='test2D',
                      space=space, time=time)


    # Create medium
    vp = ScalarField(name='vp', grid=problem.grid)
    vp.fill(1500.)

    rho = ScalarField(name='rho', grid=problem.grid)
    rho.fill(1000.)

    alpha = ScalarField(name='alpha', grid=problem.grid)
    alpha.fill(0.)

    # Define inhomogeneous regions
    # Example: a circular region with different properties
    x_center, y_center = Nx // 2, Ny // 2
    radius = 50  # radius of the inhomogeneous region

    for i in range(Nx):
        for j in range(Ny):
            if (i - x_center) ** 2 + (j - y_center) ** 2 < radius ** 2:
                vp[i, j] = 2000.0  # different wave speed in the region
                rho[i, j] = 1200.0  # different density in the region

    # Assign the medium to the problem
    problem.medium.vp = vp
    problem.medium.rho = rho
    problem.medium.add(alpha)


    # Create a container for transducers
    transducers = Transducers(grid=problem.grid)

    # Generate and add transducers to the container
    for i in range(N_trans):
        transducer = PointTransducer(id=i, position=(trans_x[i], trans_y[i]), grid=problem.grid)
        transducers.add(transducer)

    # Assign the transducers to the problem
    problem.transducers = transducers


    # Add locations to problem geometry
    problem.geometry.locations = transducers

    # Define all transducers as both sources and receivers
    sources = transducers
    receivers = transducers

    # Create shot with all transducers
    shot = Shot(source_id=0,  # First transducer as reference source
                sources=sources,
                receivers=receivers,
                geometry=problem.geometry,
                problem=problem)

    # Add shot to problem acquisitions
    problem.acquisitions.add(shot)

    # Create wavelets for each source
    f_centre = 5e6
    n_cycles = 3

    for i in range(len(sources)):
        shot.wavelets.data[i, :] = wavelets.tone_burst(f_centre, n_cycles,
                                                      time.num, time.step)
    
    # # Define the cutoff frequency and filter order
    # fmax = 5e6 / (fs / 2)  # Normalized cutoff frequency
    # order = 10
    # # Design the Butterworth filter
    # bM, aM = butter(order, fmax, btype='low')
    # # Generate the signal (example)
    # pulse_dur = int(np.round(Nt * 0.01))  # pulse duration, [samples]
    # signal = np.concatenate((np.linspace(-1, 1, pulse_dur), np.zeros(int(Nt - pulse_dur - 1))))
    # # Apply the Butterworth filter
    # signal = filtfilt(bM, aM, signal)
    # # Normalize the filtered signal
    # signal = signal / np.max(np.abs(signal))

    # Create the PDE
    pde = IsoAcousticDevito.remote(space=space, time=time)

    # Set up test cases
    cases = {
        'OT2': {'kernel': 'OT2', 'colour': 'r', 'line_style': '--'},
        'OT2-hicks': {'kernel': 'OT2', 'interpolation_type': 'hicks', 'colour': 'b', 'line_style': '--'},
        'OT2-PML': {'kernel': 'OT2', 'boundary_type': 'complex_frequency_shift_PML_2', 'colour': 'y', 'line_style': '--'},
        'OT4': {'kernel': 'OT4', 'colour': 'r', 'line_style': '-.'},
        'OT4-hicks': {'kernel': 'OT4', 'interpolation_type': 'hicks', 'colour': 'b', 'line_style': '-.'},
        'OT4-rho': {'rho': rho, 'kernel': 'OT4', 'colour': 'r', 'line_style': '-.'},
        'OT4-alpha-0': {'alpha': alpha, 'attenuation_power': 0, 'kernel': 'OT4', 'colour': 'r', 'line_style': '-.'},
        'OT4-alpha-2': {'alpha': alpha, 'attenuation_power': 2, 'kernel': 'OT4', 'colour': 'r', 'line_style': '-.'},
        'OT4-rho-alpha-0': {'rho': rho, 'alpha': alpha, 'attenuation_power': 0, 'kernel': 'OT4', 'colour': 'r',
                            'line_style': '-.'},
    }

    # # Run
    # data_analytic = analytical_2d(space, time, shot, 1500.)
    # data_analytic /= np.max(np.abs(data_analytic))

    # shot.observed.data[:] = data_analytic
    # _, axis = shot.observed.plot(plot=False, colour='k', skip=5)

    # results = {}
    # legends = {}
    # for case, config in cases.items():
    #     runtime.logger.info('\n')
    #     runtime.logger.info('===== Running %s' % case)

    #     shot.observed.deallocate()
    #     sub_problem = problem.sub_problem(shot.id)
    #     shot_wavelets = sub_problem.shot.wavelets

    #     await pde.clear_operators()
    #     traces = await pde(shot_wavelets, vp, problem=sub_problem, diff_source=True, **config).result()

    #     # Check consistency with analytical solution
    #     data_stride = traces.data.copy()
    #     data_stride /= np.max(np.abs(data_stride))
    #     error = np.sqrt(np.sum((data_stride - data_analytic)**2)/data_analytic.shape[0])

    #     # Show results
    #     results[case] = error

    #     shot.observed.data[:] = data_stride
    #     _, axis = shot.observed.plot(plot=False, axis=axis, skip=5,
    #                                  colour=config['colour'], line_style=config['line_style'])
    #     legends[case] = lines.Line2D([0, 1], [1, 0], color=config['colour'], linestyle=config['line_style'])

    # runtime.logger.info('\n')
    # runtime.logger.info('Error results:')
    # for case, error in results.items():
    #     runtime.logger.info('\t* %s : %f' % (case, error))

    plt.legend(legends.values(), legends.keys(), loc='lower right')
    plt.show()


if __name__ == '__main__':
    mosaic.run(main)
