# ETcentric_forecast
This repository contains the codes to forecast different detector networks using gwfast.

We firstly compute the `detection efficiency and rate`, then perform the gwfast runs (`gwfast_run.py`) with the following .sub:

```bash
universe        = vanilla
executable      = /ligo/home/ligo.org/arianna.scarpa/fisher_scripts/cluster_gwfast_run.py
error           = /ligo/home/ligo.org/arianna.scarpa/fisher_scripts/output/log_$(Process).err
output          = /ligo/home/ligo.org/arianna.scarpa/fisher_scripts/output/log_$(Process).out
log             = /ligo/home/ligo.org/arianna.scarpa/fisher_scripts/output/log.log
arguments       = $(Process) 100
getenv          = True
request_memory  = 12288
request_cpus    = 1
notification    = never
queue 100
```

Then, we analyze the gwfast runs with `post_fisher_analysis.ipynb` and finally we check the results with `bilby_runs_best_events.py`
