#!/usr/bin/env python3
"""Run independent Isaac tracks in one Kit process, preserving normal artifacts.

Use tools/runtime/run_isaac.sh. A new physics stage, World and vehicles are built for
each track; only application startup and renderer compilation are shared.
"""
import argparse
import json
from pathlib import Path
import sys
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tracks',nargs='+',required=True)
    parser.add_argument('--record-tracks',nargs='*',default=['bahrain'])
    parser.add_argument('--batch-log',type=Path,help='Override the run-scoped batch log path')
    parser.add_argument('--stop-file',type=Path,
                        help='Finish the current track, then stop if this file exists; completed artifacts remain available')
    args,forward=parser.parse_known_args()
    if '--engine' in forward or '--track' in forward or '--record' in forward:
        parser.error('Engine is isaac; use --tracks and --record-tracks for track and video selection')
    from racing.runtime import run
    from racing.runtime.artifacts import RunLayout
    from racing.simulators.isaac import environment as isaac_backend
    def forwarded(name,default=None):
        return forward[forward.index(name)+1] if name in forward else default
    layout=RunLayout.open(forwarded('--run-id',forwarded('--tag','isaac-batch')),forwarded('--runs-root'))
    batch_log=args.batch_log or layout.logs/'isaac_batch.json'
    native_class=isaac_backend.RaceIsaacEnv
    app=native_class.create_app()
    class SharedApplicationEnv(native_class):
        def __init__(self,*positional,**kwargs):
            super().__init__(*positional,simulation_app=app,**kwargs)
    isaac_backend.RaceIsaacEnv=SharedApplicationEnv
    rows=[]
    try:
        for track in args.tracks:
            if args.stop_file is not None and args.stop_file.exists():
                print('RACING_BATCH_PAUSED',json.dumps({'next_track':track,'stop_file':str(args.stop_file)}),flush=True)
                break
            sys.argv=['run_racing.py','--engine','isaac','--track',track,*forward]
            if track in args.record_tracks:sys.argv+=['--record']
            try:
                run.main()
                rows.append({'track':track,'execution':'finished'})
            except Exception:
                rows.append({'track':track,'execution':'failed','error':traceback.format_exc()})
                # A partially constructed world may not have reached env.close.
                from isaacsim.core.api import World
                World.clear_instance()
            run.save(batch_log,rows)
            print('RACING_BATCH',json.dumps(rows[-1]),flush=True)
    finally:
        isaac_backend.RaceIsaacEnv=native_class
        app.close()


if __name__=='__main__':main()
