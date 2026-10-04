# Sourced by the run 18 jobs: refuse to start unless modules/constants.py is set up for run 18.
python - <<'PY' || exit 1
import sys; sys.path.insert(0, ".")
import modules.constants as C
print(f"RUN_LOCATION={C.RUN_LOCATION}\ngeometry={C.GEOMETRY_PATH_RESOLVED} [{C.GEOMETRY_GROUP}]\n"
      f"primaries={C.TAU_WHOLESKY_PATH}\nkernel frame={C.KERNEL_FRAME} t_interp={C.KERNEL_T_INTERP} "
      f"circular={C.KERNEL_CIRCULAR} sigma={C.SIGMA_SPATIAL} k={C.KERNEL_INTERP_K} "
      f"width cap={C.KERNEL_INTERP_MAX} m dataset_fraction={C.DATASET_FRACTION}")
ok = (C.RUN_LOCATION.rstrip("/").endswith("18_plane45S_6km_showerframe_200k") and C.GEOMETRY_GROUP == "plane45s"
      and C.KERNEL_FRAME == "shower" and C.KERNEL_T_INTERP and C.KERNEL_CIRCULAR
      and C.KERNEL_INTERP_K == 8 and C.SIGMA_SPATIAL == 10.0 and C.KERNEL_INTERP_MAX == 50.0
      and C.DATASET_FRACTION >= 1.0 and C.NUM_SHOWERS == 210_600)
sys.exit(0 if ok else "constants do not match run 18")
PY
R=$(python -c 'import sys; sys.path.insert(0, "."); import modules.constants as C; print(C.RUN_LOCATION)')
