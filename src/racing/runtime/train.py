"""One-command entry point for PPO training with a retained CEM baseline."""
from __future__ import annotations

import sys

from .configuration import composed_values, load_training_profile


def main() -> None:
    arguments=sys.argv[1:]
    tokens=[token for token in arguments if '=' in token and not token.startswith('--')]
    values=composed_values(tokens)
    profile=load_training_profile(values.get('train','default'))
    algorithm=values.get('algorithm',profile.algorithm)
    if algorithm=='ppo':
        from .ppo import main as ppo_main
        ppo_main(arguments)
    elif algorithm=='cem':
        from .run import main as run_main
        run_main(['--train',*arguments])
    else:
        raise SystemExit(f'unknown training algorithm: {algorithm!r}; choose ppo or cem')


if __name__ == '__main__':
    main()
