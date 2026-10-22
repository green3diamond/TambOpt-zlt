# Sourced by the run 19 jobs: refuse to start unless modules/constants.py is set up for run 19.
python - <<'PY' || exit 1
import sys; sys.path.insert(0, ".")
import modules.constants as C
print(f"RUN_LOCATION={C.RUN_LOCATION}\ngeometry={C.GEOMETRY_PATH_RESOLVED} [{C.GEOMETRY_GROUP}]\n"
      f"primaries={C.TAU_WHOLESKY_PATH}\nkernel frame={C.KERNEL_FRAME} t_interp={C.KERNEL_T_INTERP} "
      f"transverse={C.KERNEL_TRANSVERSE} sigma={C.SIGMA_SPATIAL} k={C.KERNEL_INTERP_K} "
      f"width cap={C.KERNEL_INTERP_MAX} m dataset_fraction={C.DATASET_FRACTION}")
ok = (C.RUN_LOCATION.rstrip("/").endswith("19_plane45S_12km_showerframe_400k") and C.GEOMETRY_GROUP == "plane45s12"
      and C.KERNEL_FRAME == "shower" and C.KERNEL_T_INTERP and C.KERNEL_TRANSVERSE == "ring"
      and C.KERNEL_INTERP_K == 8 and C.SIGMA_SPATIAL == 10.0 and C.KERNEL_INTERP_MAX == 50.0
      and C.DATASET_FRACTION >= 1.0 and C.NUM_SHOWERS == 421_200
      and getattr(C, "RING_RADIUS_SCALE", 1.0) == 1.0)                      # run 19 labels use rings at |t1|
sys.exit(0 if ok else "constants do not match run 19")
PY
R=$(python -c 'import sys; sys.path.insert(0, "."); import modules.constants as C; print(C.RUN_LOCATION)')
