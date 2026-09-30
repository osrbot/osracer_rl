"""Load a run checkpoint, record one native evaluation, and optionally play it."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

from ..simulators import available_simulators
from .artifacts import RunLayout
from .configuration import apply_play_overrides, task_track
from .run import main as run_main


def resolve_checkpoint(run_id: str, runs_root=None, checkpoint=None) -> tuple[RunLayout,Path]:
    layout=RunLayout.existing(run_id,runs_root)
    if checkpoint:path=Path(checkpoint).expanduser().resolve()
    else:
        pt=layout.checkpoints/'policy.pt'
        path=pt if pt.is_file() else layout.checkpoints/'policy.json'
    if not path.is_file():
        raise FileNotFoundError(f'checkpoint not found: {path}')
    return layout,path


def training_defaults(layout: RunLayout) -> dict:
    """Load simulator/task settings saved by training, as ASAP evaluation does."""
    configs=sorted(layout.config.glob('training_*.json'))
    if not configs:return {}
    return json.loads(configs[-1].read_text())


def main(argv=None) -> None:
    parser=argparse.ArgumentParser(description=__doc__,epilog=(
        'ASAP-style example: osracer-play experiment_name=demo'))
    parser.add_argument('overrides',nargs='*',metavar='KEY=VALUE',
                        help='Composable settings such as +simulator=isaac')
    parser.add_argument('--experiment-name','--run-id',dest='run_id',
                        help='Training run containing checkpoints/policy.pt or policy.json')
    parser.add_argument('--runs-root',help=argparse.SUPPRESS)
    parser.add_argument('--checkpoint',help='Override the checkpoint inside the run')
    parser.add_argument('--simulator','--engine',dest='engine',choices=available_simulators())
    parser.add_argument('--task','--track',dest='track',type=task_track)
    parser.add_argument('--seconds',type=float,help=argparse.SUPPRESS)
    parser.add_argument('--seed',type=int,default=0,help=argparse.SUPPRESS)
    parser.add_argument('--opponent-speed',type=float,help=argparse.SUPPRESS)
    parser.add_argument('--opponent-gap',type=float,help=argparse.SUPPRESS)
    parser.add_argument('--observation-profile',choices=['sim','real'],help=argparse.SUPPRESS)
    parser.add_argument('--no-open',action='store_true',help='Generate the video without opening a player')
    parser.add_argument('--player',default='ffplay',help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    try:apply_play_overrides(args,args.overrides,available_simulators())
    except (TypeError,ValueError) as exc:parser.error(str(exc))
    if not args.run_id:parser.error('provide experiment_name=<run> or --experiment-name <run>')
    try:layout,checkpoint=resolve_checkpoint(args.run_id,args.runs_root,args.checkpoint)
    except (ValueError,FileNotFoundError) as exc:parser.error(str(exc))
    saved=training_defaults(layout)
    args.engine=args.engine or saved.get('engine','mujoco')
    args.track=args.track or saved.get('track','bahrain')
    args.seconds=args.seconds if args.seconds is not None else float(saved.get('seconds',120.))
    args.opponent_speed=(args.opponent_speed if args.opponent_speed is not None
                         else float(saved.get('opponent_speed',2.8)))
    args.opponent_gap=(args.opponent_gap if args.opponent_gap is not None
                       else float(saved.get('opponent_gap',3.)))
    args.observation_profile=args.observation_profile or saved.get('observation_profile','sim')
    if args.seconds<=0:parser.error('--seconds must be positive')
    result=run_main(['--engine',args.engine,'--track',args.track,'--seconds',str(args.seconds),
        '--seeds',str(args.seed),'--checkpoint',str(checkpoint),'--record','--tag','playback',
        '--run-id',layout.run_id,'--runs-root',str(layout.root.parent),
        '--opponent-speed',str(args.opponent_speed),'--opponent-gap',str(args.opponent_gap),
        '--observation-profile',args.observation_profile])
    video=Path(result['video'])
    print(f'PLAYBACK_VIDEO {video}',flush=True)
    if args.no_open or not os.environ.get('DISPLAY'):
        return
    player=shutil.which(args.player)
    if player is None:
        parser.error(f'video player not found: {args.player}; use --no-open to keep the recording only')
    subprocess.run([player,'-autoexit','-loglevel','error',str(video)],check=False)


if __name__ == '__main__':
    main()
