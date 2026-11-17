"""Read-only, CPU-only audit of the 2026-09-30 interrupted time-study run."""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BUNDLE = Path('D:/pa-csi-dg-revision-runtime/studies/r_time_pooling_v1_20260929')
RUN_ID = 'E2_G_NATIVE_MERGE_502'
OUT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(BUNDLE / 'code'))

import numpy as np
import torch
from experiments.revision_validation.experiment_design import build_experiment_model
from experiments.revision_validation.extended_engine import verify_extension
from experiments.revision_validation.evaluate_study import metrics_for
from experiments.revision_validation.source_data import epoch_schedule, schedule_hash
from experiments.revision_validation.study_engine import equal_state
from experiments.revision_validation.study_io import read, sha256, value_hash, verify_completed, write, now


def finite(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(v) for v in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def main():
    torch.set_num_threads(4)
    assert not torch.cuda.is_initialized()
    protocol, bundle_hash = verify_extension(BUNDLE)
    config = next(c for c in protocol['jobs'] if c['job_id'] == RUN_ID)
    run = BUNDLE / 'runs' / RUN_ID
    failed = read(run / 'status.json')
    failure_hash = sha256(run / 'status.json')
    queue_failure = read(BUNDLE / 'queue_failure.json')
    assert failed['status'] == 'failed' and 'PermissionError: [WinError 5]' in failed['error']
    assert 'status.json.tmp' in failed['error'] and 'os.replace(temporary, path)' in failed['error']
    assert 'extended_engine.py\", line 201' in failed['error']
    assert read(run / 'config.json') == config
    assert read(run / 'runtime.json') == config['expected_runtime']
    assert sha256(config['source_manifest_path']) == config['source_manifest_sha256']
    source = read(config['source_manifest_path'])
    progress = read(run / 'progress.json')
    checkpoint = run / progress['checkpoint']
    assert checkpoint.parent == run and checkpoint.is_file()
    assert sha256(checkpoint) == progress['sha256']
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert state['config_value_sha256'] == value_hash(config)
    history = read(run / 'history.json')
    assert state['history'] == history
    epoch, step = progress['completed_epochs'], state['step']
    assert len(history) == epoch == 136
    assert step == failed['optimizer_updates_attempted'] == epoch * config['updates_per_epoch'] == 40800
    assert finite(state), 'Nonfinite checkpoint tensor or scalar'
    assert finite(history), 'Nonfinite JSON history'
    assert [h['epoch'] for h in history] == list(range(1, epoch + 1))
    assert all(h['optimizer_steps'] == h['epoch'] * config['updates_per_epoch'] for h in history)
    for index, row in enumerate(history):
        assert row['schedule_sha256'] == schedule_hash(epoch_schedule(source['train'], config['seed'], index, steps=config['updates_per_epoch']))
        assert row['learning_rate_after_epoch'] == config['learning_rate'] * 0.9 ** (row['optimizer_steps'] / 10000)
        assert row['parameters_with_gradient_at_first_batch'] == 1779454
        assert row['regularizer_weight_at_epoch_end'] == 0
        assert row['mean_primary_regularizer_total'][1] == 0
    selected = max(range(epoch), key=lambda i: history[i]['source_validation']['macro_f1'])
    assert state['best_epoch'] == selected + 1
    assert state['best'] == history[selected]['source_validation']['macro_f1']
    predictions = state['best_predictions']
    assert len(predictions) == len(source['val']) == 1200
    for row, pred in zip(source['val'], predictions):
        assert row['label'] == pred['true']
        assert all(row[k] == pred[k] for k in ('sample_id','subject_id','acquisition_id'))
    assert metrics_for(predictions) == history[selected]['source_validation']
    model = build_experiment_model(config)
    model.load_state_dict(state['model'], strict=True)
    assert equal_state(model.state_dict(), state['model'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    optimizer.load_state_dict(state['optimizer'])
    assert equal_state(optimizer.state_dict(), state['optimizer'])
    model.load_state_dict(state['best_state'], strict=True)
    assert equal_state(model.state_dict(), state['best_state'])
    assert sum(p.numel() for p in model.parameters()) == config['total_parameters'] == 1878142
    group = state['optimizer']['param_groups']
    assert len(group) == 1 and group[0]['lr'] == history[-1]['learning_rate_after_epoch']
    assert group[0]['betas'] == (0.9, 0.999) and group[0]['eps'] == 1e-8 and group[0]['weight_decay'] == 0
    named = list(model.named_parameters())
    expected_active = {i for i,(name,p) in enumerate(named) if not name.startswith('projection_head.')}
    assert set(state['optimizer']['state']) == expected_active
    adam_steps = {int(v['step']) for v in state['optimizer']['state'].values()}
    assert adam_steps == {step}
    assert sum(named[i][1].numel() for i in expected_active) == 1779454
    for i, value in state['optimizer']['state'].items():
        assert value['exp_avg'].shape == value['exp_avg_sq'].shape == named[i][1].shape
        assert bool((value['exp_avg_sq'] >= 0).all())
    rng = state['rng']
    cpu_generator = torch.Generator(device='cpu')
    cpu_generator.set_state(rng['torch'])
    assert torch.equal(cpu_generator.get_state(), rng['torch'])
    py_rng = random.Random(); py_rng.setstate(rng['python'])
    assert py_rng.getstate() == rng['python']
    np_state = rng['numpy']
    array_rng = np.random.RandomState()
    array_rng.set_state((np_state[0], np.asarray(np_state[1],dtype=np.uint32),np_state[2],np_state[3],np_state[4]))
    restored = array_rng.get_state()
    assert restored[0] == np_state[0] and np.array_equal(restored[1],np_state[1]) and tuple(restored[2:]) == tuple(np_state[2:])
    assert len(rng['cuda']) == 1 and all(v.device.type=='cpu' and v.dtype==torch.uint8 and v.numel()>0 for v in rng['cuda'])
    previous_path = run / 'resume_epoch_0135.pt'
    previous = torch.load(previous_path,map_location='cpu',weights_only=True)
    assert previous['config_value_sha256'] == state['config_value_sha256']
    assert previous['history'] == history[:-1] and previous['step'] == 40500
    assert finite(previous)
    existing_epochs = sorted(int(p.stem.rsplit('_',1)[1]) for p in run.glob('resume_epoch_*.pt'))
    assert existing_epochs[-1] == epoch
    events = [json.loads(line) for line in (run/'events.jsonl').read_text(encoding='utf-8').splitlines()]
    committed_events = [e['epoch'] for e in events if e['event']=='epoch_committed']
    assert committed_events == list(range(1,136))
    assert events[-1]['event']=='failed' and events[-1]['optimizer_updates_attempted']==40800
    numerical_error_strings = ('Nonfinite objective','Missing/nonfinite gradient','Nonfinite evaluation logits','FloatingPointError')
    assert not any(marker in failed['error']+queue_failure['error'] for marker in numerical_error_strings)
    assert source['outer_target']=='E2' and set(source['source_environments'])=={'E1','E3'}
    assert all(r['environment'] in ('E1','E3') for part in ('train','val') for r in source[part])
    profile = read(run/'resource_profile.json')
    assert profile['config_value_sha256']==value_hash(config) and not profile['target_data_loaded']
    assert all(s.split(':')[0] in ('E1','E3') for s in profile['source_sample_ids'])
    assert failed['target_data_loaded'] is False and config['target_data_loaded'] is False
    assert not (BUNDLE/'target_gate.json').exists() and not (BUNDLE/'evaluation').exists()
    complete = []
    for c in protocol['jobs']:
        directory = BUNDLE/'runs'/c['job_id']
        if (directory/'status.json').exists() and read(directory/'status.json')['status']=='completed':
            status=verify_completed(directory,c)
            assert read(directory/'config.json')==c
            h=read(directory/'history.json')
            p=read(directory/'selected_source_predictions.json')
            assert metrics_for(p)==h[status['best_epoch']-1]['source_validation']
            assert status['optimizer_updates']==60000 and status['target_data_loaded'] is False
            complete.append({'job_id':c['job_id'],'epochs':status['epochs'],'optimizer_updates':status['optimizer_updates'],
                             'best_epoch':status['best_epoch'],'status_sha256':sha256(directory/'status.json'),
                             'all_saved_artifact_hashes_valid':True,'source_metrics_reconstructed':True,'do_not_retrain':True})
    assert len(complete)==4
    assert [c['job_id'] for c in complete]==[c['job_id'] for c in protocol['jobs'][:4]]
    assert sha256(run/'status.json')==failure_hash, 'Run changed during read-only audit'
    assert not torch.cuda.is_initialized()
    report={'passed':True,'checked_at':now(),'schema':'interrupted-run-cpu-audit/v1','bundle':str(BUNDLE),
            'bundle_sha256':bundle_hash,'failed_job_id':RUN_ID,'failure_kind':'Windows status JSON atomic-replace access denial',
            'failure_status_sha256':failure_hash,'queue_failure_sha256':sha256(BUNDLE/'queue_failure.json'),
            'failure_time_utc':failed['updated_at'],'maximum_verified_recoverable_epoch':epoch,
            'resume_from_next_epoch':epoch+1,'completed_optimizer_updates':step,'remaining_optimizer_updates':60000-step,
            'checkpoint':str(checkpoint),'checkpoint_sha256':progress['sha256'],
            'checkpoint_file_bytes':checkpoint.stat().st_size,
            'hashes':{name:sha256(run/name) for name in ('config.json','runtime.json','progress.json','history.json','events.jsonl','resource_profile.json')},
            'checks':{'frozen_bundle_intact':True,'configuration_exact':True,'runtime_matches_frozen_config':True,
                      'progress_checkpoint_hash_valid':True,'checkpoint_history_json_exact':True,
                      'all_136_schedules_reconstructed':True,'learning_rates_and_steps_exact':True,
                      'all_model_best_state_optimizer_tensors_finite':True,'model_strict_cpu_load_exact':True,
                      'optimizer_cpu_load_exact':True,'all_active_Adam_steps_40800':True,'Adam_active_parameter_count':1779454,
                      'earliest_source_best_epoch_correct':True,'source_best_predictions_metrics_reconstruct':True,
                      'prior_135_checkpoint_is_exact_history_prefix':True,'no_later_checkpoint_exists':True,
                      'CPU_Python_NumPy_rng_restored_locally':True,'CUDA_rng_present_and_well_formed':True,
                      'CUDA_rng_actual_restore_tested':False,'numerical_failure_found':False,
                      'source_only_artifacts_consistent':True,'no_target_gate_or_predictions':True,
                      'audit_gpu_initialized':False,'audit_optimizer_steps':0,'audit_signal_arrays_opened':False},
            'rng_summary':{'cpu_state_bytes':rng['torch'].numel(),'cuda_state_bytes':[v.numel() for v in rng['cuda']],
                           'python_state_version':rng['python'][0],'numpy_generator':np_state[0]},
            'best_source_epoch':state['best_epoch'],'best_source_macro_f1':state['best'],
            'last_epoch_source_macro_f1':history[-1]['source_validation']['macro_f1'],
            'elapsed_training_seconds_in_checkpoint':state['elapsed_seconds'],
            'event_log_last_successful_epoch':135,'available_checkpoint_epochs':existing_epochs,
            'verified_completed_runs':complete,
            'recovery_recommendation':'Archive the original failed status and queue failure, then resume the same config/runtime from the progress-bound epoch136 checkpoint. Use bounded Windows atomic-write retries outside the frozen scientific code. Keep the four completed jobs and all seeds unchanged.',
            'limitations':['CPU-only checkpoint inspection cannot prove exact future CUDA replay; the original nondeterminism limit remains.',
                           'No new source or target forward pass was performed. This validates stored states, identities, metrics, and code/configuration bindings.',
                           'WinError5 proves an access-denied write; the particular competing handle or OS cause has not been identified.'],
            'audit_script_sha256':sha256(Path(__file__))}
    write(OUT/'checkpoint_audit.json',report)
    lines=['# 2026-09-30：中断 checkpoint 独立核验','',
           '**结论：可从已完整保存的第 136 epoch 恢复，下一轮为 137；无需回退至 135。四个已完成运行保持不变。**','',
           f'- 中断运行：`{RUN_ID}`。',
           '- 故障时间：2026-09-30 02:54:31 日本时间（2026-09-29 17:54:31 UTC）。',
           '- 故障：`status.json.tmp → status.json` 的 `os.replace` 返回 Windows WinError 5。无法仅凭该错误确定具体占用者。',
           '- checkpoint、progress 和 history 已先于 status 写入成功；事件日志停在 135 不代表 136 的训练未保存。',
           f'- 已保存 40,800 次更新；剩余 19,200 次。checkpoint SHA256：`{progress["sha256"]}`。','',
           '## 验证内容','',
           '- 冻结 bundle、配置与保存的 runtime 一致；4 个 completed 的全部附件哈希、60,000 步预算及最早源验证最优选模验证通过。',
           '- 第 136 轮模型、最佳模型、Adam 参数/状态、历史均有限；CPU 严格加载并逐位核对通过。全部有效 Adam 参数的 step 为 40,800。',
           '- 重算全部 136 轮采样哈希，与保存记录一致；学习率、步数、历史长度、135 轮 checkpoint 前缀一致。',
           '- 1,200 条最佳源验证预测身份正确，macro-F1 可从预测重新构造，最优 epoch 符合最早最大值规则。',
           '- CPU/Python/NumPy RNG 状态可恢复；CUDA RNG 字节状态存在且结构有效。本检查没有初始化 GPU，未测试实际 CUDA 状态恢复。',
           '- 未发现数值错误；源环境仅 E1/E3，失败状态/配置/预检一致声明未读取目标数据，尚无目标 gate 或预测。',
           '- 本检查未打开信号数组、未运行模型前向、未更新优化器、未修改旧 bundle 或状态。','',
           '## 恢复边界','',
           '先归档原始失败状态和队列错误，再允许相同配置从 progress 绑定的 136 轮恢复。Windows 原子写入重试应仅针对明确的临时共享/访问错误并设置上限；保留冻结科学代码、全部环境/种子和已完成运行。',
           'GPU 逐位复现不在这次 CPU 检查的证明范围内；保持既有 CUDA 非严格确定性说明。','',
           '完整逐项证据见同目录 `checkpoint_audit.json`。','']
    (OUT/'checkpoint_audit.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'passed':True,'recoverable_epoch':epoch,'next_epoch':epoch+1,'verified_completed_runs':len(complete),
                      'gpu_initialized':torch.cuda.is_initialized(),'best_source_epoch':state['best_epoch'],'report':str(OUT/'checkpoint_audit.json')}),flush=True)


if __name__=='__main__':main()
