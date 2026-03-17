#!/ligo/home/ligo.org/arianna.scarpa/.conda/envs/gwfast/bin/python

import os
import numpy as np
import matplotlib.pyplot as plt
import bilby
from bilby.gw.detector import Interferometer, PowerSpectralDensity
from bilby.gw.conversion import convert_to_lal_binary_black_hole_parameters
from bilby.gw.likelihood.relative import RelativeBinningGravitationalWaveTransient
from bilby.gw.source import lal_binary_black_hole_relative_binning, lal_binary_black_hole
import gwfast.gwfastGlobals as glob

glob.detPath = os.path.join(os.getcwd(), "psds")
fmin = 3.0
sampling_frequency = 2048.0
outdir = "outdir_2_A_BBH_ETD-CE20-CE40"
label = "bibly_check"

bilby.core.utils.setup_logger(outdir=outdir, label=label)

injection_parameters = dict(
    mass_1=33.325256,
    mass_2=18.720100,
    a_1=0.361861,
    a_2=0.079694,
    tilt_1=0.000000,
    tilt_2=3.141593,
    phi_12=0.000000,
    phi_jl=0.0,
    luminosity_distance=792.381276,
    theta_jn=2.463650,
    psi=1.045656,
    phase=1.696272,
    geocent_time=0.0,
    ra=4.459685,
    dec=0.796240,
)

m1_inj = injection_parameters["mass_1"]
m2_inj = injection_parameters["mass_2"]
dL_inj = injection_parameters["luminosity_distance"]
t_inj = injection_parameters["geocent_time"]

Mc_inj= ((m1_inj *m2_inj) **(3.0 / 5.0)) / ((m1_inj +m2_inj) ** (1.0 / 5.0))
q_inj =min(m1_inj, m2_inj) / max(m1_inj,m2_inj)

print(f"Chirp mass: {Mc_inj:.2f} Msun")
print(f"Mass ratio: {q_inj:.2f}")

def compute_duration(m1, m2, f_low, sampling_frequency, margin=1.1, pad_seconds=4.0):
    from bilby.gw.utils import calculate_time_to_merger
    t_signal = calculate_time_to_merger(frequency=f_low, mass_1=m1, mass_2=m2)
    raw = t_signal * margin + pad_seconds
    duration = 2 ** int(np.ceil(np.log2(raw)))
    return duration

duration = compute_duration(m1_inj, m2_inj, fmin, sampling_frequency)

waveform_arguments = dict(waveform_approximant="IMRPhenomXPHM", reference_frequency=50.0,minimum_frequency=fmin)

waveform_generator_inj = bilby.gw.WaveformGenerator(duration=duration,
                                                 sampling_frequency=sampling_frequency,
                                                 frequency_domain_source_model=lal_binary_black_hole,
                                                 parameter_conversion=convert_to_lal_binary_black_hole_parameters,
                                                 waveform_arguments=waveform_arguments)

waveform_generator_rb = bilby.gw.WaveformGenerator(duration=duration,
                                                   sampling_frequency=sampling_frequency,
                                                    frequency_domain_source_model=lal_binary_black_hole_relative_binning,
                                                    parameter_conversion=convert_to_lal_binary_black_hole_parameters,
                                                  waveform_arguments=waveform_arguments)

CE2NM = Interferometer(name="CE2NM",
                       power_spectral_density=PowerSpectralDensity(asd_file=os.path.join(glob.detPath, "ce_strain/cosmic_explorer_20km.txt")),
                       minimum_frequency=fmin, maximum_frequency=2048.0, length=20.0,
                       latitude=33.160, longitude=-106.480, elevation=0.0,
                       xarm_azimuth=-105.0, yarm_azimuth=-105.0 + 90.0)

CE1Id = Interferometer(name="CE1Id",
                       power_spectral_density=PowerSpectralDensity(asd_file=os.path.join(glob.detPath, "ce_strain/cosmic_explorer.txt")),
                       minimum_frequency=fmin, maximum_frequency=2048.0, length=40.0,
                       latitude=43.827, longitude=-112.825, elevation=0.0,
                       xarm_azimuth=-45.0, yarm_azimuth=-45.0 + 90.0)

LIGOI = Interferometer(name="LIGOI",
                       power_spectral_density=PowerSpectralDensity(asd_file=os.path.join(glob.detPath, "ligo_india/Asharp_strain.txt") ),
                       minimum_frequency=fmin, maximum_frequency=2048.0, length=4.0,
                       latitude=19.613, longitude=77.031, elevation=0.0,
                       xarm_azimuth=287.384, yarm_azimuth=287.384 + 90.0)

ET_list = bilby.gw.detector.InterferometerList(["ET"])
ETD = ET_list[0]
ETD.minimum_frequency = fmin
ETD.maximum_frequency = 2048.0
ETD.power_spectral_density = PowerSpectralDensity(asd_file=os.path.join(glob.detPath, "ET_designs_comparison_paper/HFLF_cryo/ETLength10km.txt"))

ifos = bilby.gw.detector.InterferometerList([LIGOI, CE1Id, ETD])


ifos.set_strain_data_from_zero_noise(sampling_frequency=sampling_frequency,  duration=duration, start_time=injection_parameters["geocent_time"]- 2)


ifos.inject_signal(waveform_generator=waveform_generator_inj,parameters=injection_parameters,)

priors = bilby.gw.prior.BBHPriorDict()

priors["chirp_mass"] = bilby.prior.Uniform(name="chirp_mass", minimum=Mc_inj - 0.1, maximum=Mc_inj + 0.1)
priors["mass_ratio"] = bilby.prior.Uniform(name="mass_ratio", minimum=q_inj - 0.1, maximum=min(1.0, q_inj + 0.1))
priors["luminosity_distance"] = bilby.prior.Uniform(name="luminosity_distance", minimum=dL_inj - (dL_inj * 0.1), maximum=dL_inj + (dL_inj * 0.1))
priors["geocent_time"] = bilby.prior.Uniform(name="geocent_time", minimum=t_inj - 0.002, maximum=t_inj + 0.002)
priors["ra"] = bilby.prior.Uniform(name="ra", minimum=injection_parameters["ra"] - 0.1, maximum=injection_parameters["ra"] + 0.1)
priors["dec"] = bilby.prior.Uniform(name="dec", minimum=injection_parameters["dec"] - 0.1, maximum=injection_parameters["dec"] + 0.1)
priors["theta_jn"] = bilby.prior.Uniform(name="theta_jn", minimum=injection_parameters["theta_jn"] - 0.2, maximum=injection_parameters["theta_jn"] + 0.2)
priors["psi"] = bilby.prior.Uniform(name="psi", minimum=injection_parameters["psi"] - 0.2, maximum=injection_parameters["psi"] + 0.2)

for key in ["a_1", "a_2", "tilt_1", "tilt_2", "phi_12", "phi_jl", "phase"]:
    priors[key] = bilby.prior.DeltaFunction(name=key, peak=injection_parameters[key])

priors.validate_prior(duration, fmin)

fiducial_parameters = injection_parameters.copy()
fiducial_parameters.pop("mass_1")
fiducial_parameters.pop("mass_2")
fiducial_parameters["chirp_mass"] = Mc_inj
fiducial_parameters["mass_ratio"] = q_inj

parameter_bounds = dict(
    chirp_mass=(priors["chirp_mass"].minimum, priors["chirp_mass"].maximum),
    mass_ratio=(priors["mass_ratio"].minimum, priors["mass_ratio"].maximum),
    luminosity_distance=(priors["luminosity_distance"].minimum, priors["luminosity_distance"].maximum),
    geocent_time=(priors["geocent_time"].minimum, priors["geocent_time"].maximum),
    theta_jn=(priors["theta_jn"].minimum, priors["theta_jn"].maximum),
    psi=(priors["psi"].minimum, priors["psi"].maximum),
    ra=(0.0, 2.0 * np.pi),
    dec=(-0.5 * np.pi, 0.5 * np.pi))

likelihood = RelativeBinningGravitationalWaveTransient(interferometers=ifos,
                                                    waveform_generator=waveform_generator_rb,
                                                    fiducial_parameters=fiducial_parameters,
                                                    parameter_bounds=parameter_bounds,
                                                    update_fiducial_parameters=False,
                                                    distance_marginalization=False,   
                                                    time_marginalization=False,       
                                                    phase_marginalization=False,      
                                                    priors=priors,
                                                    jitter_time=False)


result = bilby.run_sampler(
    likelihood=likelihood, 
    priors=priors, 
    sampler='dynesty',
    injection_parameters=injection_parameters,
    outdir=outdir, 
    label=label,
    #nlive=1000, 
    #sample='rwalk', 
    #nact=5, 
    sample='slice',  
    nlive=2000,      
    nact=10,
    walks=50, 
    dlogz=0.1)

print("Done!")
result.injection_parameters["chirp_mass"] = Mc_inj
result.injection_parameters["mass_ratio"] = q_inj
result.plot_corner()
