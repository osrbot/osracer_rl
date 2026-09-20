#!/usr/bin/env bash
set -euo pipefail
isaac_dir=${NEORACER_ISAAC_DIR:-/home/osrbot/rlgpu_ws/isaac-sim-standalone-6.0.1-linux-x86_64}
nccl_library="$isaac_dir/extsDeprecated/omni.isaac.ml_archive/pip_prebundle/nvidia/nccl/lib/libnccl.so.2"
test -f "$nccl_library"
export LD_PRELOAD="$nccl_library${LD_PRELOAD:+:$LD_PRELOAD}"
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json
exec bash "$isaac_dir/python.sh" "$@"
