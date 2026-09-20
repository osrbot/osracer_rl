#!/usr/bin/env python3
"""Source-engine validation of the isolated sensor-only closed-loop candidate."""
import argparse
import hashlib
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--track',default='bahrain')
    parser.add_argument('--ratio',type=float,default=2.5)
    parser.add_argument('--pulse',type=float,default=.6)
    parser.add_argument('--steer-bias',type=float,default=.1)
    parser.add_argument('--target-slip-deg',type=float,default=25.)
    parser.add_argument('--seconds',type=float,default=120)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--record',action='store_true')
    parser.add_argument('--safety',action='store_true')
    parser.add_argument('--tag',default='v6d_isaac')
    args=parser.parse_args()
    from racing import run
    from racing.policy_closed_loop import ClosedLoopRacingPolicy
    from racing.policy import RacingPolicy as BasePolicy
    from racing.tracks import Track
    from racing.isaac_env import RaceIsaacEnv
    class Candidate(ClosedLoopRacingPolicy):
        def __init__(self,parameters=None):
            super().__init__(parameters,max_boost_seconds=args.pulse,steering_bias=args.steer_bias,
                             target_slip_deg=args.target_slip_deg)
            self.feedback_rows=[]
            self.supervisor=None
            if args.safety:
                from racing.safety import LocalSafetySupervisor
                self.supervisor=LocalSafetySupervisor()
        def action(self,observation):
            wheels,steers=super().action(observation)
            if self.supervisor:
                wheels,steers=self.supervisor.filter(observation,wheels,steers)
                if self.supervisor.diagnostics.get('must_brake'):
                    self.abort_drift('wall_supervisor')
                self.feedback['safety'] = dict(self.supervisor.diagnostics)
            self.feedback_rows.append(dict(self.feedback))
            return wheels,steers
    actors=[]
    def factory(parameters=None):
        # The opponent remains the same fixed baseline used in qualification.
        actor=Candidate(parameters) if not actors else BasePolicy(parameters)
        actors.append(actor)
        return actor
    run.RacingPolicy=factory
    parameters=Candidate.initial.copy();parameters[-2]=args.ratio
    out=ROOT/'output/racing/experimental'/args.tag;out.mkdir(parents=True,exist_ok=True)
    sources={p.name:p.read_bytes() for p in (ROOT/'racing').glob('*.py')}
    for name,payload in sources.items():(out/name).write_bytes(payload)
    (out/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    track=Track(args.track);(out/'track.json').write_bytes(track.path.read_bytes())
    env=RaceIsaacEnv(track,render=args.record)
    try:
        video=run.Video(out/'native.mp4','isaac',track) if args.record else None
        result,rows=run.rollout(env,parameters,seed=args.seed,seconds=args.seconds,video=video)
        if video:result['video']=video.close()
        result.update(engine='isaac',experimental=True,qualified_for_leaderboard=False,
            policy_version=Candidate.version,parameters=parameters.tolist(),max_boost_seconds=args.pulse,
            steering_bias=args.steer_bias,target_slip_deg=args.target_slip_deg,safety_supervisor=args.safety,
            track_sha256=hashlib.sha256(track.path.read_bytes()).hexdigest(),
            runtime_sha256={name:hashlib.sha256(data).hexdigest() for name,data in sources.items()})
        run.save(out/'trace.json',rows);run.save(out/'feedback.json',actors[0].feedback_rows)
        run.save(out/'result.json',result)
        print({k:result.get(k) for k in ['valid_lap','failure','duration_s','effective_overtakes',
            'max_rear_slip_deg','max_continuous_drift_duration_s']},flush=True)
    except Exception:
        import traceback
        run.save(out/'error.json',{'error':traceback.format_exc()});raise
    finally:env.close()


if __name__=='__main__':main()
