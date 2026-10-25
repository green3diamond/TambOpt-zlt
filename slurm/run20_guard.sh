# Sourced by the run 20 jobs (Malata): set the run 20 env, then refuse to start unless modules/constants.py
# resolves to it. Run 20 = the Malata mesh (data/malata.h5, group malata), the Julia tau_wholesky_n750k.h5
# decays, run 15's generated corpus filtered to >= 30 deg from the C8 field axis (scripts/20_filter_corpus.py),
# Malata-sized layout strategies; otherwise run 19's settings. Defaults in constants.py stay run 19's.
export TAMBOPT_RUN_LOCATION=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/zdimitrov/detector_optimization_v6/20_malata_showerframe
export TAMBOPT_GEOMETRY_PATH=$TAMBOPT_RUN_LOCATION/inputs/malata.h5 TAMBOPT_GEOMETRY_GROUP=malata
export TAMBOPT_TAU_PATH=$TAMBOPT_RUN_LOCATION/inputs/tau_wholesky_n750k.h5 TAMBOPT_SITE=malata
python - <<'PY' || exit 1
import json, os, sys; sys.path.insert(0, ".")
import modules.constants as C
print(f"RUN_LOCATION={C.RUN_LOCATION}\ngeometry={C.GEOMETRY_PATH_RESOLVED} [{C.GEOMETRY_GROUP}]\n"
      f"primaries={C.TAU_WHOLESKY_PATH}\nsite={C.SITE}\nkernel frame={C.KERNEL_FRAME} t_interp={C.KERNEL_T_INTERP} "
      f"transverse={C.KERNEL_TRANSVERSE} sigma={C.SIGMA_SPATIAL} k={C.KERNEL_INTERP_K} "
      f"width cap={C.KERNEL_INTERP_MAX} m dataset_fraction={C.DATASET_FRACTION}")
filt = C.DUAL_SHOWER_CACHE_PATH + ".filter.json"
ok = (C.RUN_LOCATION.rstrip("/").endswith("20_malata_showerframe") and C.GEOMETRY_GROUP == "malata"
      and C.SITE == "malata" and C.TAU_WHOLESKY_PATH.endswith("tau_wholesky_n750k.h5")
      and C.KERNEL_FRAME == "shower" and C.KERNEL_T_INTERP and C.KERNEL_TRANSVERSE == "ring"
      and C.KERNEL_INTERP_K == 8 and C.SIGMA_SPATIAL == 10.0 and C.KERNEL_INTERP_MAX == 50.0
      and C.DATASET_FRACTION >= 1.0
      and os.path.exists(filt) and json.load(open(filt))["min_angle_b"] == 30.0)
sys.exit(0 if ok else "constants do not match run 20 (or the filtered corpus is not finished)")
PY
R=$TAMBOPT_RUN_LOCATION
