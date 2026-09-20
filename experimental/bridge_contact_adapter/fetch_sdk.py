#!/usr/bin/env python3
"""Fetch pinned public headers only; never clone or replace installed libraries."""
import hashlib,json,tarfile,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
DEST=ROOT/'output/racing/bridge_adapter_sdk'
SOURCES={
 'physx':('https://codeload.github.com/NVIDIA-Omniverse/PhysX/tar.gz/refs/tags/110.1-omni-and-physx-5.9.0','c924aa80e73562730d6b3335a31843fe1acbe2af9f5a56563e0953ecbe3ff77d'),
 'usd':('https://codeload.github.com/PixarAnimationStudios/OpenUSD/tar.gz/refs/tags/v25.11','c37c633b5037a4552f61574670ecca8836229b78326bd62622f3422671188667')}
def fetch():
 DEST.mkdir(parents=True,exist_ok=True)
 for name,(url,digest) in SOURCES.items():
  archive=DEST/(name+'.tar.gz')
  if not archive.exists():
   with urllib.request.urlopen(url,timeout=120) as response:archive.write_bytes(response.read())
  if hashlib.sha256(archive.read_bytes()).hexdigest()!=digest:raise RuntimeError(f'Checksum mismatch: {archive}')
  with tarfile.open(archive) as tar:
   for member in tar.getmembers():
    if not member.isfile():continue
    rel=None
    if name=='physx' and '/physx/include/' in member.name:rel=Path('physx_include')/member.name.split('/physx/include/',1)[1]
    elif name=='physx' and '/omni/ovruntime/include/omni/physx/' in member.name:rel=Path('omni_include')/member.name.split('/include/',1)[1]
    elif name=='usd' and '/pxr/' in member.name and member.name.endswith(('.h','.h.in')):rel=Path('usd_include/pxr')/member.name.split('/pxr/',1)[1]
    elif member.name.count('/')==1 and member.name.rsplit('/',1)[1].lower().startswith('license'):rel=Path('licenses')/(name+'_'+member.name.rsplit('/',1)[1])
    if rel is None:continue
    target=(DEST/rel).resolve()
    if DEST.resolve() not in target.parents:raise RuntimeError('Unsafe archive path')
    target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(tar.extractfile(member).read())
 template=DEST/'usd_include/pxr/pxr.h.in';text=template.read_text()
 values={'PXR_MAJOR_VERSION':'0','PXR_MINOR_VERSION':'25','PXR_PATCH_VERSION':'11','PXR_VERSION':'2511','PXR_USE_NAMESPACES':'1','PXR_EXTERNAL_NAMESPACE':'pxr','PXR_INTERNAL_NAMESPACE':'pxrInternal_v0_25_11','PXR_PYTHON_SUPPORT_ENABLED':'1','PXR_PREFER_SAFETY_OVER_SPEED':'0'}
 for key,value in values.items():text=text.replace('@'+key+'@',value)
 template.with_suffix('').write_text(text)
 (DEST/'sdk_sources.json').write_text(json.dumps(SOURCES,indent=2)+'\n')
if __name__=='__main__':fetch()
