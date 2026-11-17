"""CPU-only recovery-wrapper fixtures: never starts a real runner or model."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
RUNTIME=Path('D:/pa-csi-dg-revision-runtime')
SOURCE=ROOT/'analysis/resume_revision_io_20260930.py'
BASE_SOURCE=ROOT/'analysis/coordinate_revision_experiments_20260929.py'
OUT=RUNTIME/'qa/revision_recovery_20260930.json'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True)+'\n',encoding='utf-8')


def main():
    before=sha(SOURCE)
    recovery=load(SOURCE,'recovery_under_test')
    base=load(BASE_SOURCE,'completion_verifier_under_test')
    checks={}
    fixture=Path(tempfile.mkdtemp(prefix='revision_recovery_fixture_',dir=str(RUNTIME/'qa')))
    bundle=fixture/'study';run=bundle/'runs/J';run.mkdir(parents=True)
    code=bundle/'code';code.mkdir()
    module=code/'entry.py';module.write_text("raise RuntimeError('This fixture module must never be executed')\n",encoding='utf-8')
    protocol={'entry_module':'entry','jobs':[{'job_id':'J','epochs':200}]}
    dump(bundle/'protocol.json',protocol)
    dump(bundle/'bundle_manifest.json',{'files_sha256':{'protocol.json':sha(bundle/'protocol.json'),'code/entry.py':sha(module)}})
    directory=fixture/'coordinator';directory.mkdir()
    driver=directory/'driver.py';driver.write_text('# fixed infrastructure fixture\n',encoding='utf-8')
    plan={'code_sha256':{'driver.py':sha(driver)},'studies':[{'bundle':str(bundle),'bundle_sha256':sha(bundle/'bundle_manifest.json')}]}
    dump(directory/'plan.json',plan)
    events_path=run/'events.jsonl'
    active=[{'event':'failed','optimizer_updates_attempted':40800},
            {'event':'resume_state_audit','model_exact':True,'optimizer_exact':True,'rng_exact':True},
            {'event':'resumed','completed_epochs':136},
            {'event':'epoch_committed','epoch':137},
            {'event':'completed','optimizer_updates':60000}]
    expected=[('failed',0),('failed',0),('running',136),('running',137),('completed',200)]
    for count,(status,epoch) in enumerate(expected,1):
        events_path.write_text(''.join(json.dumps(e)+'\n' for e in active[:count]),encoding='utf-8')
        with patch.object(recovery,'read',side_effect=AssertionError('observe must never open replaced JSON')):
            result=recovery.observe(plan,[protocol])[0]
        if status=='completed':
            assert result['completed']==1 and result['active']==[]
        else:
            assert result['active'][0]['status']==status and result['active'][0]['epoch']==epoch
    checks['failed_resumed_epoch_completed_sequence']=True
    events_path.write_text('clipped-first-line\n'+json.dumps(active[2])+'\n'+json.dumps(active[3])+'\n{"event":"epoch_',encoding='utf-8')
    result=recovery.observe(plan,[protocol])[0]
    assert result['active'][0]['status']=='running' and result['active'][0]['epoch']==137
    checks['clipped_first_and_partial_last_jsonl_ignored']=True
    events_path.unlink()
    result=recovery.observe(plan,[protocol])[0]
    assert result['completed']==0 and result['active']==[]
    checks['missing_event_log_stays_pending']=True
    # Worker is executed only against this dummy bundle. The scientific entry
    # module is intercepted; no torch import, GPU call or subprocess is possible.
    calls=[];installs=[]
    shim=types.ModuleType('windows_atomic_retry_20260930')
    def install(roots,path,timeout_seconds):
        installs.append({'roots':[str(p) for p in roots],'path':str(path),'timeout':timeout_seconds})
    shim.install_replace_retry=install
    def intercept(name,run_name):
        calls.append({'module':name,'run_name':run_name,'cwd':str(Path.cwd()),
                      'argv':list(sys.argv),'sys_path_zero':sys.path[0],
                      'cublas':os.environ.get('CUBLAS_WORKSPACE_CONFIG')})
        assert name=='entry' and run_name=='__main__'
        assert Path.cwd()==code and Path(sys.path[0])==code
        assert sys.argv==['entry','run','--bundle',str(bundle)]
        assert os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
    original_cwd=Path.cwd();original_path=sys.path[:];original_argv=sys.argv[:]
    original_cublas=os.environ.get('CUBLAS_WORKSPACE_CONFIG')
    try:
        with patch.dict(sys.modules,{'windows_atomic_retry_20260930':shim}), \
             patch.object(recovery,'load_base',return_value=base), \
             patch.object(recovery.runpy,'run_module',side_effect=intercept), \
             patch.object(recovery.subprocess,'Popen',side_effect=AssertionError('No subprocess permitted in this fixture')):
            recovery.worker(directory,0)
            assert len(calls)==1 and installs==[{'roots':[str(bundle)],'path':str(directory/'0_atomic_retry.jsonl'),'timeout':30}]
            checks['worker_frozen_cwd_sys_path_argv_and_retry_scope']=True
            # A valid infrastructure plan with a modified scientific file must
            # fail before installing retries or dispatching its entry point.
            original_science=module.read_bytes();module.write_text('# tampered fixture\n',encoding='utf-8')
            try:
                recovery.worker(directory,0)
            except RuntimeError as error:
                assert 'Frozen study file differs' in str(error)
            else:raise AssertionError('Worker accepted altered frozen science')
            module.write_bytes(original_science)
            assert len(calls)==1 and len(installs)==1
            checks['worker_rejects_frozen_scientific_code_tampering']=True
            driver.write_text('# changed infrastructure\n',encoding='utf-8')
            try:
                recovery.worker(directory,0)
            except RuntimeError as error:
                assert 'Recovery infrastructure code changed' in str(error)
            else:raise AssertionError('Worker accepted altered recovery infrastructure')
            assert len(calls)==1 and len(installs)==1
            checks['worker_rejects_infrastructure_code_tampering']=True
    finally:
        os.chdir(original_cwd);sys.path[:]=original_path;sys.argv[:]=original_argv
        if original_cublas is None:os.environ.pop('CUBLAS_WORKSPACE_CONFIG',None)
        else:os.environ['CUBLAS_WORKSPACE_CONFIG']=original_cublas
    checks['observe_never_reads_mutable_study_JSON']=True
    checks['no_real_training_dispatch_or_subprocess']=True
    after=sha(SOURCE)
    assert before==after,'Recovery wrapper changed during QA; rerun against final code'
    report={'passed':all(checks.values()),'checks':checks,
            'recovery_module_sha256':after,
            'recovery_wrapper_sha256_at_start':before,'recovery_wrapper_sha256_at_end':after,
            'completion_verifier_sha256':sha(BASE_SOURCE),'test_script_sha256':sha(Path(__file__)),
            'fixture_directory':str(fixture),'worker_dispatches_intercepted':calls,
            'models_constructed':False,'gpu_initialized':False,'real_training_or_target_inference':False,
            'real_study_state_modified':False}
    OUT.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({'passed':report['passed'],'check_count':len(checks),'wrapper_sha256':after,'report':str(OUT)}),flush=True)


if __name__=='__main__':main()
