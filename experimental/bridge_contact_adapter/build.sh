#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
isaac_dir=${OSRACER_ISAAC_DIR:-/home/osrbot/rlgpu_ws/isaac-sim-standalone-6.0.1-linux-x86_64}
sdk=output/racing/bridge_adapter_sdk
if [[ ! -f "$sdk/omni_include/omni/physx/IPhysx.h" ]]; then python3 experimental/bridge_contact_adapter/fetch_sdk.py; fi
usd_lib="$isaac_dir/extscache/omni.usd.libs-1.0.3+f9bf0dda.lx64.r.cp312/bin"
g++ -std=c++17 -O2 -DNDEBUG -fPIC -shared \
 -I/usr/include/eigen3 -I"$sdk/physx_include" -I"$sdk/usd_include" \
 -I"$isaac_dir/kit/dev/include" -I"$sdk/omni_include/omni/physx" \
 experimental/bridge_contact_adapter/adapter.cpp experimental/bridge_contact_adapter/contact_kernel.cpp experimental/bridge_contact_adapter/geometry_dispatch.cpp experimental/bridge_contact_adapter/mesh_surface.cpp \
 -L"$usd_lib" -L"$isaac_dir/kit" -lusd_tf -lusd_sdf -lcarb -lfcl -lccd -loctomap -loctomath \
 -Wl,-rpath,"$usd_lib" -Wl,-rpath,"$isaac_dir/kit" \
 -o "$sdk/libbridge_contact_adapter.so"

python3 - <<'PYBUILD'
import hashlib,json
from pathlib import Path
root=Path('experimental/bridge_contact_adapter');binary=Path('output/racing/bridge_adapter_sdk/libbridge_contact_adapter.so')
files=[p for p in root.rglob('*') if p.is_file() and p.suffix in ('.cpp','.hpp','.sh','.json','.usda')]
manifest={'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'compiled_source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'sdk_sources':json.loads(Path('output/racing/bridge_adapter_sdk/sdk_sources.json').read_text()),'compiler_flags':'g++ -std=c++17 -O2 -DNDEBUG -fPIC -shared','sdk_tag':'110.1-omni-and-physx-5.9.0','usd_abi':'0.25.11'}
binary.with_suffix('.build.json').write_text(json.dumps(manifest,indent=2)+'\n')
PYBUILD
