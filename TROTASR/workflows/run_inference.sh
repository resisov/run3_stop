#!/bin/bash
# Numerical runtime only: no external analysis Python path or ROOT writer.
set -euo pipefail
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TF_NUM_INTEROP_THREADS=1 TF_NUM_INTRAOP_THREADS=1 NUMBA_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=2
set +u
cd /hep2-scratch/twkim/envs/combine_20260914/CMSSW_14_1_0_pre4/src
source /cvmfs/cms.cern.ch/cmsset_default.sh
eval "$(scramv1 runtime -sh)"
set -u
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/cvmfs/sft.cern.ch/lcg/releases/blas/0.3.20.openblas-c07f1/x86_64-el9-gcc13-opt/lib:/cvmfs/sft.cern.ch/lcg/releases/hdf5/1.12.2-4c7fc/x86_64-el9-gcc13-opt/lib"
export PYTHONPATH="/hep2-scratch/twkim/envs/trotasr_inference_py39:$ROOTSYS/lib:/cvmfs/sft.cern.ch/lcg/views/LCG_104/x86_64-el9-gcc13-opt/lib/python3.9/site-packages"
export ROOTENV_NO_HOME=1
cd /hep2-scratch/twkim/postprocessing
exec python3 -u "$@"
