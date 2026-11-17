"""Execute the sealed within-link unwrap control serially on one GPU, preserving attempts."""
from __future__ import annotations
import argparse, json, os, sys, uuid
from pathlib import Path
from common import BUNDLE, HERE, JOBS, OUTPUT, PARAMETERS, base_module, now, read, require, sha, verify_bundle, write

def backend():
    base = base_module('runner')
    base.OUTPUT_ROOT = OUTPUT
    base.EXPERIMENT_ROOT = HERE
    base.SCHEMA_VERSION = 'h-phase-link-unwrap-runner/v1'
    base.COMPLETION_VERSION = 'h-phase-link-unwrap-completion/v1'
    base.SEAL_VERSION = 'h-phase-link-unwrap-seal/v1'
    original = base.validate_run
    def validate(job, attempt, python):
        health = original(job, attempt, python)
        config = read(attempt / 'job_config.json')
        c = config['h_controls'];run = attempt / 'payload/run'
        audit = read(run / 'control_alignment_audit.json')
        initialization = read(run / 'initialization_audit.json')
        require(audit['status'] == initialization['status'] == 'passed', 'Control alignment failed')
        require(health['parameter_count'] == initialization['parameter_count'] == PARAMETERS, 'Actual capacity drift')
        require(initialization['initial_state_sha256'] == c['initial_state_sha256'], 'Actual initialization drift')
        require(initialization['common_state_sha256'] == c['common_state_sha256'], 'Shared initialization drift')
        require(health['sampler_schedule_sha256'] == c['expected_sampler_sha256'], 'Actual sampling drift')
        split = read(run / 'split_manifest.json')
        require(split['grouped_source_manifest_sha256'] == c['source_manifest_sha256'], 'Actual source membership drift')
        for k in ('sample_id','subject_id','acquisition_id'):
            require(not set(split['group_identity']['train'][k]) & set(split['group_identity']['val'][k]), 'Actual source identity overlap')
        metrics = read(run / 'metrics.json')
        require(min(row['train']['supcon_frac_anchors_with_positive'] for row in metrics['history']) == 1., 'Positive anchor coverage drift')
        health.update(initial_state_sha256=initialization['initial_state_sha256'],
            common_state_sha256=initialization['common_state_sha256'],
            control_alignment_audit_sha256=sha(run/'control_alignment_audit.json'))
        return health
    base.validate_run = validate
    return base

def main():
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-run',action='store_true');mode.add_argument('--execute',action='store_true')
    args=p.parse_args()
    manifest,registry=verify_bundle()
    base=backend();python=Path(sys.executable)
    jobs=[base.Job(j['job_id'],j['target'],j['seed'],j['arm'],BUNDLE/j['config'],
        sha(BUNDLE/j['config']),base.payload_sha256(read(BUNDLE/j['config']))) for j in registry['jobs']]
    if args.dry_run:
        print(json.dumps({'status':'dry_run_passed','jobs':len(jobs),'first_job':jobs[0].job_id,
            'new_environment_experiment':False,'plan':[{'job_id':j.job_id,'action':base.scan(j,python).action} for j in jobs]}))
        return
    approval=read(OUTPUT/'qa/approval.json')
    require(approval['status']=='passed' and approval['bundle_manifest_sha256']==sha(BUNDLE/'bundle_manifest.json'), 'Preflight not approved for this bundle')
    import psutil
    for process in psutil.process_iter(['pid','name','cmdline']):
        try:
            command=' '.join(process.info['cmdline'] or [])
            if process.pid!=os.getpid() and 'python' in (process.info['name'] or '').lower() and '--config' in command:
                raise RuntimeError('An existing training worker is active: '+str(process.pid))
        except (psutil.NoSuchProcess,psutil.AccessDenied): pass
    invocation_id=uuid.uuid4().hex
    with base.RunnerLock(invocation_id):
        invocation=OUTPUT/'runner_invocations'/invocation_id;invocation.mkdir(parents=True)
        write(invocation/'started.json',{'at':now(),'pid':os.getpid(),'process_create_time':psutil.Process().create_time(),
            'status':'started','bundle_manifest_sha256':sha(BUNDLE/'bundle_manifest.json'),'jobs':JOBS},exclusive=True)
        completed=[]
        try:
            for index,job in enumerate(jobs):
                verify_bundle()
                state=base.scan(job,python)
                write(OUTPUT/'queue/status.json',{'status':'running','at':now(),'pid':os.getpid(),
                    'process_create_time':psutil.Process().create_time(),'current_job':job.job_id,
                    'current_attempt':state.next_attempt,'completed_jobs':len(completed),'total_jobs':JOBS,
                    'position':index+1,'manifest_sha256':sha(BUNDLE/'bundle_manifest.json')})
                if state.action=='run':
                    attempt,health=base.execute_job(job,state.next_attempt,python)
                else:
                    attempt=state.successful_attempt
                    health=next(p['health'] for p in state.prior if p['status']=='success' and Path(p['path'])==attempt)
                completed.append({'job_id':job.job_id,'attempt':str(attempt),'health':health,
                    'completion_sha256':sha(attempt/'attempt_completed.json')})
                write(OUTPUT/'queue/completed_jobs.json',{'status':'complete' if len(completed)==JOBS else 'partial',
                    'at':now(),'completed':completed,'total_jobs':JOBS})
                from analyze import analyze
                analyze(completed)
                if (OUTPUT/'queue/stop_after_current.json').exists():
                    write(OUTPUT/'queue/status.json',{'status':'stopped_after_completed_job','at':now(),
                        'completed_jobs':len(completed),'total_jobs':JOBS})
                    return
            write(invocation/'completed.json',{'status':'complete','at':now(),'jobs':JOBS},exclusive=True)
            write(OUTPUT/'queue/status.json',{'status':'complete','at':now(),'completed_jobs':JOBS,'total_jobs':JOBS})
        except BaseException as error:
            write(OUTPUT/'queue/status.json',{'status':'failed','at':now(),'current_job':job.job_id,
                'completed_jobs':len(completed),'total_jobs':JOBS,'error':repr(error)})
            raise

if __name__=='__main__':main()
