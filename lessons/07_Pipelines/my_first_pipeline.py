
import argparse

import numpy as np

# import lesson_tools

import toast
from toast.mpi import MPI
import toast.pipeline_tools


# Fake focaplane is copied here from lesson_tools to avoid import error

def fake_focalplane(
    samplerate=20,
    epsilon=0,
    net=1,
    fmin=0,
    alpha=1,
    fknee=0.05,
    fwhm=30,
    npix=7,
    fov=3.0
):
    """Create a set of fake detectors.

    This generates 7 pixels (14 dets) in a hexagon layout at the boresight
    and with a made up polarization orientation.

    Args:
        None

    Returns:
        (dict):  dictionary of detectors and their properties.

    """
    from toast.tod import hex_pol_angles_radial, hex_pol_angles_qu, hex_layout

    zaxis = np.array([0, 0, 1.0])
    
    pol_A = hex_pol_angles_qu(npix)
    pol_B = hex_pol_angles_qu(npix, offset=90.0)
    
    dets_A = hex_layout(npix, fov, "", "", pol_A)
    dets_B = hex_layout(npix, fov, "", "", pol_B)
    
    dets = dict()
    for p in range(npix):
        pstr = "{:01d}".format(p)
        for d, layout in zip(["A", "B"], [dets_A, dets_B]):
            props = dict()
            props["quat"] = layout[pstr]["quat"]
            props["epsilon"] = epsilon
            props["rate"] = samplerate
            props["alpha"] = alpha
            props["NET"] = net
            props["fmin"] = fmin
            props["fknee"] = fknee
            props["fwhm_arcmin"] = fwhm
            dname = "{}{}".format(pstr, d)
            dets[dname] = props
    return dets


def parse_arguments(comm):
    """ Declare and parse command line arguments
    
    """
    parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
    
    # pipeline_tools provides arguments to supported operators
    toast.pipeline_tools.add_noise_args(parser)
    toast.pipeline_tools.add_polyfilter_args(parser)
    
    # The pipeline may add its own arguments
    parser.add_argument(
        "--nsample", 
        required=False, 
        default=10000,
        type=np.int,
        help="Length of the simulation",
    )
    
    args = parser.parse_args()
    return args


def create_observations(comm, args):
    """ Create the TOAST data object using `args`
    
    This method will look very different for different pipelines.
    """
    focalplane = fake_focalplane(samplerate=args.sample_rate, fknee=1.0)
    detnames = list(sorted(focalplane.keys()))
    detquats = {x: focalplane[x]["quat"] for x in detnames}
    tod = toast.tod.TODCache(None, detnames, args.nsample, detquats=detquats)
    
    # Write some auxiliary data
    tod.write_times(stamps=np.arange(args.nsample) / args.sample_rate)
    tod.write_common_flags(flags=np.zeros(args.nsample, dtype=np.uint8))
    for d in detnames:
        tod.write_flags(detector=d, flags=np.zeros(args.nsample, dtype=np.uint8))
    
    noise = toast.pipeline_tools.get_analytic_noise(args, comm, focalplane)
    
    observation = {"tod" : tod, "noise" : noise, "name" : "obs-000", "id" : 0}
    
    data = toast.Data(comm)
    data.obs.append(observation)
    
    return data


def main():
    """ The `main` will instantiate and process the data.
    
    """
    mpiworld, procs, rank, comm = toast.pipeline_tools.get_comm()
    args = parse_arguments(comm)
    data = create_observations(comm, args)
    
    mc = 0
    name = "signal"
    tod = data.obs[0]["tod"]
    det = tod.local_dets[0]
    
    # Simulate noise, write out one detector
    toast.pipeline_tools.simulate_noise(args, comm, data, mc, cache_prefix=name)
    tod.local_signal(det, name).tofile("noise.npy")
    
    # Apply polyfilter, write the filtered data
    toast.pipeline_tools.apply_polyfilter(args, comm, data, cache_name=name)
    tod.local_signal(det, name).tofile("filtered.npy")
    

if __name__ == "__main__":
    main()
