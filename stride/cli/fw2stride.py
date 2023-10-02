import os
from stride.utils.fullwave import *
from stride import Time, Space, Problem, ScalarField
import click
from datetime import datetime


def _printer(name, var=None):
    space = 16
    name_length = len(name)
    if name_length < space:
        offset = (space-name_length)*' '
    else:
        offset = ''
    if var == 'no output':
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), f'       {name}')
    elif var is not None:
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), f'       {name}:', offset, var)
    else:
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), f'       {name}:', offset, 'None')


@click.command()
# Path and project name
@click.option('--path', type=str, default=os.getcwd(), show_default=True,
              help='path to fullwave files')
@click.option('--prjname', type=str,
              help='prefix name for Stride files')
# Acquisition data
@click.option('--ttrname', type=str,
              help='data ttr file name')
@click.option('--ttr0000', type=bool, default=False, show_default=True,
              help='is the data ttr an empty 0000.ttr file')
# Geometry
@click.option('--srcpgyname', type=str,
              help='source pgy file name')
@click.option('--recpgyname', type=str,
              help='receiver pgy file name')
# Source signature
@click.option('--srcttrname', type=str,
              help='source ttr file name')
# Vp model
@click.option('--vpvtrname', type=str,
              help='vp model vtr file name')
@click.option('--vpvalue', type=str,
              help='constant vp value to populate model in m/s (ignored if vp vtr provided)')
# Other
@click.option('--writedata', type=bool, default=True, show_default=True,
              help='whether to extract and write data from data ttr file or not')
@click.option('--srcrecsplit', type=bool, default=True, show_default=True,
              help='whether source and receiver pgy files are different or not')
@click.option('--dx', type=float,
              help='model spacing in meters (same for all dimensions)')
@click.option('--extra', type=int,
              help='how many extra cells to pad model with (same for all dimensions)')
@click.option('--absorb', type=int,
              help='how many absorbing cells to pad model with (same for all dimensions)')
@click.option('--plot', type=bool, default=False, show_default=True,
              help='plot the problem contents')
@click.option('--version', type=int, default=0, show_default=True,
              help='version of the converted problem files')
def go(**kwargs):

    path = kwargs.pop('path')
    _printer('Path', path)

    prjname = kwargs.pop('prjname', None)
    if prjname is None:
        prjname = click.prompt('Please specify a Stride project name', type=str)
    _printer('Project', prjname)

    ttrname = kwargs.pop('ttrname', None)
    _printer('obs.ttr', ttrname)

    ttr0000 = kwargs.pop('ttr0000', False)
    _printer('.ttr0000', ttr0000)

    srcpgyname = kwargs.pop('srcpgyname', None)
    _printer('src.pgy', srcpgyname)

    # Extract model shape from .pgy or prompt user for each dimension
    if srcpgyname is not None:
        srcpgyname = os.path.join(path, srcpgyname)
        src_shape = read_header_pgy(srcpgyname)
    else:
        dim3 = click.prompt('Does the model have 3 dimensions', type=bool, default=True, show_default=True)
        n1 = click.prompt('Please specify the size of the model along the first dimension', type=int)
        n2 = click.prompt('Please specify the size of the model along the second dimension', type=int)
        if dim3:
            n3 = click.prompt('Please specify the size of the model along the third dimension', type=int)
            src_shape = (n1, n2, n3)
        else:
            src_shape = (n1, n2)
    ndim = len(src_shape)
    _printer('Shape', src_shape)

    recpgyname = kwargs.pop('recpgyname', None)
    _printer('rec.pgy', recpgyname)

    # If necessary, check that the receiver .pgy dimensions match those given by the source .pgy
    if recpgyname is not None:
        recpgyname = os.path.join(path, recpgyname)
        rec_shape = read_header_pgy(recpgyname)
        assert src_shape == rec_shape, 'model dimensions mismatch in src and rec pgys'

    srcttrname = kwargs.pop('srcttrname', None)
    _printer('sig.ttr', srcttrname)

    # Extract the num and step if required
    time_grid = False
    if ttrname is not None:
        time_grid = True
        ttrname = os.path.join(path, ttrname)
        _, _, num, _, step = read_header_ttr(ttrname)
    elif srcttrname is not None:
        time_grid = True
        step = click.prompt('Please specify the size of the time step (in seconds)', type=float)
        num = click.prompt('Please specify the number of time samples', type=int)
    _printer('Time step', step)
    _printer('Num. samples', num)

    # Create the temporal grid if required
    if time_grid:
        time = Time(start=0.,
                    step=step,
                    num=num)
    else:
        time = None

    vpvtrname = kwargs.pop('vpvtrname', None)
    _printer('vp.vtr', vpvtrname)

    vpvalue = kwargs.pop('vpvalue', None)
    _printer('Const. vp', vpvalue)

    writedata = kwargs.pop('writedata', True)
    _printer('Write data', writedata)

    srcrecsplit = kwargs.pop('srcrecsplit', True)
    _printer('Split src/rec', srcrecsplit)

    # Extract model spacing
    dx = kwargs.pop('dx', None)
    if dx is None:
        dx = click.prompt('Please specify the model spacing (in meters)', type=float)

    # Extract the number of extra cells to pad model
    extra = kwargs.pop('extra', None)
    if extra is None:
        extra = click.prompt('Please specify the number of extra cells', type=int)

    # Extract the number of absorbing cells to pad model
    absorb = kwargs.pop('absorb', None)
    if absorb is None:
        absorb = click.prompt('Please specify the number of absorbing cells', type=int)

    _printer('Model dx', dx)
    dx = tuple([dx]*ndim)

    _printer('Extra cells', extra)
    extra = tuple([extra]*ndim)

    _printer('Absorb. cells', absorb)
    absorb = tuple([absorb]*ndim)

    # Create the spatial grid
    space = Space(shape=src_shape,
                  extra=extra,
                  absorbing=absorb,
                  spacing=dx)

    # Create problem
    problem = Problem(name=prjname, space=space, time=time)

    # Create vp ScalarField, if required
    if vpvtrname is not None or vpvalue is not None:
        vp = ScalarField(name='vp', grid=problem.grid)
        if vpvtrname is not None:
            _printer('Loading vp model', 'no output')
            vp.data[:] = read_vtr_model3D(vtr_path=os.path.join(path, vpvtrname))
            vp.pad()
        else:
            vp.fill(vpvalue)
        problem.medium.add(vp)

    # Create point transducers
    problem.transducers.default()

    # Load acquisition data
    if srcttrname is not None:
        srcttrname = os.path.join(path, srcttrname)

    # Load geometry, extracting the offset to be added to the receiver ids in the .pgy files
    if srcpgyname is not None:
        _printer('Loading geometry', 'no output')
        offset_id = problem.geometry.from_fullwave(srcpgyname, recpgyname)
    else:
        offset_id = 0
        srcrecsplit = False

    if ttrname is not None:
        _printer('Loading acquisitions', 'no output')
        problem.acquisitions.from_fullwave(acquisition_path=ttrname,
                                source_path=srcttrname,
                                read_traces=writedata,
                                has_traces=ttr0000,
                                src_rcv_split=srcrecsplit,
                                offset_id=offset_id)

    # Save problem files to disk
    version = kwargs.pop('version', 0)
    _printer('Saving to disk', 'no output')
    problem.dump(path=path, project_name=prjname, version=version)

    # Plot if required
    plot = kwargs.pop('plot', False)
    if plot:
        problem.plot()


if __name__ == '__main__':
    go()
