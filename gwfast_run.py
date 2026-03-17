#!/ligo/home/ligo.org/arianna.scarpa/.conda/envs/gwfast/bin/python
import os
import sys
sys.path.insert(0, os.getcwd())
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

plt.rcParams["figure.dpi"] = 150

import copy
import numpy as np
from astropy.cosmology import Planck18

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd())))
sys.path.append(SCRIPT_DIR)

import gwfast.gwfastGlobals as glob
glob.detPath = os.path.join(os.getcwd(), "psds")
from gwfast.waveforms import IMRPhenomD
from gwfast.signal import GWSignal
from gwfast.network import DetNet
from fisherTools import CovMatr, compute_localization_region, check_covariance, fixParams
import gwfast.waveforms
from contextlib import contextmanager

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

if len(sys.argv) > 2:
    job_id = int(sys.argv[1])
    total_jobs = int(sys.argv[2])
else:
    job_id = 0
    total_jobs = 1

configurations = {
    # A
    "A_BBH_ETD-CE20-CE40": ["ETSea", "CE2NM", "CE1Id"],
    "A_BBH_ETD-CE20-LI": ["ETSea", "CE2NM", "LIGOI"],
    "A_BBH_ETD-CE40-LI": ["ETSea", "CE1Id", "LIGOI"],

    # B
    "B_BBH_ET2L-CE20": ["ETSL", "ETMRL45d", "CE2NM"],
    "B_BBH_ET2L-CE40": ["ETSL", "ETMRL45d", "CE1Id"],
    "B_BBH_ET2L-LI": ["ETSL", "ETMRL45d", "LIGOI"],

    # C
    "C_BBH_ET1L-CE20-LI": ["ETSL", "CE2NM", "LIGOI"],
    "C_BBH_ET1L-CE40-CE20": ["ETSL", "CE1Id", "CE2NM"],
    "C_BBH_ET1L-CE40-LI": ["ETSL", "CE1Id", "LIGOI"],
}

RUN_SET = "ALL"

if RUN_SET == "ALL":
    selected_configurations = configurations
elif RUN_SET in ["A", "B", "C"]:
    selected_configurations = {k: v for k, v in configurations.items() if k.startswith(RUN_SET + "_")}
elif isinstance(RUN_SET, list):
    selected_configurations = {k: configurations[k] for k in RUN_SET}
else:
    raise ValueError("RUN_SET non valido")

output_dir = "/ligo/home/ligo.org/arianna.scarpa/fisher_scripts/results"
os.makedirs(output_dir, exist_ok=True)

with h5py.File("BBH_injections.h5", "r") as f:
    total_N_events = len(f["chirp_mass"])
    per_job = total_N_events // total_jobs
    
    start_idx = job_id * per_job
    if job_id == total_jobs - 1:
        end_idx = total_N_events
    else:
        end_idx = (job_id + 1) * per_job
    
    BBH_inj = {key: f[key][start_idx:end_idx] for key in f.keys() if isinstance(f[key], h5py.Dataset)}

N = len(BBH_inj["chirp_mass"])
print(f"JOB {job_id}/{total_jobs}: Elaborazione di {N} eventi (da {start_idx} a {end_idx})")

BBH_injections = {
    "Mc": BBH_inj["chirp_mass"],
    "eta": BBH_inj["symmetric_mass_ratio"],
    "chi1z": BBH_inj["spin_1z"],
    "chi2z": BBH_inj["spin_2z"],
    "dL": BBH_inj["luminosity_distance"]/ 1e3,
    "theta": np.pi/2 - BBH_inj["dec"],
    "phi": BBH_inj["ra"],
    "iota": BBH_inj["theta_jn"],
    "psi": BBH_inj["psi"],
    "Phicoal": BBH_inj["phase"],
    "z": BBH_inj["redshift"],
}

BBH_injections["tcoal"] = np.zeros(N)

Pars = gwfast.waveforms.LAL_WF('IMRPhenomXPHM').ParNums
idL = Pars["dL"]
iiota = Pars["iota"]
iMc  = Pars["Mc"]
ieta = Pars["eta"]

SNRth = 10
CHUNK = 10 

results = {}

for config_name, det_list in selected_configurations.items():

    print("\nRunning:", config_name)

    alldetectors = copy.deepcopy(glob.detectors)
    detectors = {det: alldetectors[det] for det in det_list}

    for d in detectors.keys():
        if d == "ETSea":
            detectors[d]["psd_path"] = os.path.join(glob.detPath, "ET_designs_comparison_paper/HFLF_cryo/ETLength10km.txt")
        if d in ["ETSL", "ETMRL45d"]:
            detectors[d]["psd_path"] = os.path.join(glob.detPath, "ET_designs_comparison_paper/HFLF_cryo/ETLength15km.txt")
        if d == "CE1Id":
            detectors[d]["psd_path"] = os.path.join(glob.detPath, "ce_strain/cosmic_explorer.txt")
        if d == "CE2NM":
            detectors[d]["psd_path"] = os.path.join(glob.detPath, "ce_strain/cosmic_explorer_20km.txt")
        if d == "LIGOI":
            detectors[d]["psd_path"] = os.path.join(glob.detPath, "ligo_india/Asharp_strain.txt")

    Signals = {}
    for d in detectors.keys():
        Signals[d] = GWSignal(
            gwfast.waveforms.LAL_WF('IMRPhenomXPHM'),
            psd_path=detectors[d]['psd_path'],
            detector_shape=detectors[d]['shape'],
            det_lat=detectors[d]['lat'],
            det_long=detectors[d]['long'],
            det_xax=detectors[d]['xax'],
            verbose=False,
            useEarthMotion=False,
            fmin=3.,
            IntTablePath=None
        )

    myNet = DetNet(Signals)

    SNR = myNet.SNR(BBH_injections)
    mask = SNR > SNRth

    BBH_injections_survived = {k: v[mask] for k, v in BBH_injections.items()}
    SNR_survived = SNR[mask]

    N_surv = len(BBH_injections_survived["theta"])

    skyArea = np.full(N_surv, np.nan)
    status = np.zeros(N_surv, dtype=int)
    rel_dL = np.full(N_surv, np.nan)
    diota = np.full(N_surv, np.nan)
    rel_Mc = np.full(N_surv, np.nan)
    deta = np.full(N_surv, np.nan)

    for start in tqdm(range(0, N_surv, CHUNK), desc=f"Job {job_id} - {config_name}"):
        end = min(start + CHUNK, N_surv)
        ev_chunk = {k: v[start:end] for k, v in BBH_injections_survived.items()}

        try:
            with suppress_stdout():
                F = myNet.FisherMatr(ev_chunk)

            Cov, inv_err = CovMatr(F)

            sky_chunk = compute_localization_region(Cov, Pars, ev_chunk["theta"])
            skyArea[start:end] = sky_chunk

            sigma_dL = np.sqrt(Cov[idL, idL, :])
            sigma_iota = np.sqrt(Cov[iiota, iiota, :])
            rel_dL[start:end] = sigma_dL / ev_chunk["dL"]
            diota[start:end] = sigma_iota

            sigma_Mc  = np.sqrt(Cov[iMc, iMc, :])
            sigma_eta = np.sqrt(Cov[ieta, ieta, :])
            rel_Mc[start:end] = sigma_Mc / ev_chunk["Mc"]
            deta[start:end]   = sigma_eta

        except Exception:
            status[start:end] = 1
            continue

    filepath = os.path.join(output_dir, f"{config_name}_job{job_id}.h5")

    local_indices = np.arange(len(SNR))
    global_indices = start_idx + local_indices
    global_indices_surv = global_indices[mask]

    with h5py.File(filepath, "w") as f_out:
        f_out.create_dataset("global_idx", data=global_indices_surv)
        f_out.create_dataset("SNR", data=SNR_survived)
        f_out.create_dataset("skyArea", data=skyArea)
        f_out.create_dataset("rel_dL", data=rel_dL)
        f_out.create_dataset("rel_Mc", data=rel_Mc)
        f_out.create_dataset("deta", data=deta)
        f_out.create_dataset("diota", data=diota)
        f_out.attrs["SNRth"] = SNRth
        f_out.attrs["start_idx"] = start_idx
        f_out.attrs["end_idx"] = end_idx
