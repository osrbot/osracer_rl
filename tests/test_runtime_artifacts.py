import json

import pytest

from racing.runtime.artifacts import RunLayout
from racing.runtime.play import resolve_checkpoint
from racing.runtime.tracking import MetricLogger


def test_run_layout_and_playback_checkpoint_contract(tmp_path):
    layout=RunLayout.open('cem-demo',tmp_path)
    checkpoint=layout.checkpoints/'policy.json'
    checkpoint.write_text('{"parameters": []}')
    resolved_layout,resolved_checkpoint=resolve_checkpoint('cem-demo',tmp_path)
    assert resolved_layout.root==layout.root
    assert resolved_checkpoint==checkpoint
    assert layout.tensorboard==tmp_path/'cem-demo/metrics/tensorboard'
    assert json.loads(layout.manifest.read_text())['run_id']=='cem-demo'


def test_run_id_cannot_escape_runs_root(tmp_path):
    with pytest.raises(ValueError):
        RunLayout.open('../outside',tmp_path)


def test_jsonl_metrics_are_flushed_without_tensorboard(tmp_path):
    with MetricLogger(tmp_path,enabled=False) as logger:
        logger.log(3,'candidate',{'score':12.5})
        row=json.loads((tmp_path/'metrics.jsonl').read_text())
    assert row=={'step':3,'scope':'candidate','values':{'score':12.5}}
