"""Prepare versioned operational wrappers without changing frozen H code."""
from pathlib import Path
import ast
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new, 1)


def save(name, text):
    ast.parse(text)
    path = ROOT / 'analysis' / name
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(text)
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    wrapper = (ROOT / 'analysis/execute_h_grouped_l1248_recovery.py').read_text(encoding='utf-8')
    start = wrapper.index('    gpu = subprocess.run(')
    end = wrapper.index('    base = frozen_run.backend()', start)
    wrapper = wrapper[:start] + '''    from h_gpu_availability_v2 import verify_available
    guard = Path(__file__).with_name("h_gpu_availability_v2.py")
    common.require(common.sha(guard) == owner["gpu_guard_sha256"], "GPU guard changed")
    common.require(common.sha(Path(__file__)) == owner["execution_wrapper_sha256"], "Execution wrapper changed")
    gpu_report = verify_available()
    common.write_new(h_root / "queue/gpu_execution_probe_v2.json", gpu_report)
    common.require(gpu_report["available"], "Unknown GPU client or insufficient idle resources")

''' + wrapper[end:]
    wrapper = wrapper.replace('This wrapper replaces only the interrupted prerequisite-process binding.',
                              'This v2 wrapper replaces prerequisite binding and the WDDM desktop-client GPU gate.')
    saved = [save('execute_h_grouped_l1248_recovery_v2.py', wrapper)]

    queue = (ROOT / 'analysis/recover_h_grouped_l1248_queue.py').read_text(encoding='utf-8')
    queue = replace_once(queue, 'parser.add_argument("--inputs-pid", type=int, required=True)',
                         'parser.add_argument("--inputs-pid", type=int, default=0)')
    queue = replace_once(queue, '''    birth, command = process_identity(
        args.inputs_pid, ("analysis/continue_inputs_after_controls.py", str(inputs_output)))''', '''    # Inputs are sealed and their owners have exited; no identity-holder process.
    require(args.inputs_pid == 0, "Use completed-input binding, not a live prerequisite")
    completed_inputs = verify_input_completion(inputs_output, input_bundle, input_hash)
    verified_path = Path(__file__).resolve().parents[1] / "docs/revision/evidence/multienv_inputs_completion_verified_20260917.json"
    verified = read(verified_path)
    require(verified["completed_source_runs"] == verified["completed_target_results"] == 63,
            "Input verification is incomplete")
    require(verified["coordinator_status_sha256"] == completed_inputs["input_coordinator_status_sha256"],
            "Completed input provenance differs")
    require(verified["evaluation_summary_sha256"] == completed_inputs["input_evaluation_summary_sha256"],
            "Completed input evaluation differs")
    birth, command = None, []''')
    queue = replace_once(queue, '''    write_new(queue / "owner.json", identity)''', '''    guard = Path(__file__).with_name("h_gpu_availability_v2.py")
    wrapper = Path(__file__).with_name("execute_h_grouped_l1248_recovery_v2.py")
    identity.update(completed_input_verification_sha256=sha256(verified_path),
                    gpu_guard_sha256=sha256(guard), execution_wrapper_sha256=sha256(wrapper),
                    recovery_reason="WDDM reports desktop clients as compute processes; known desktop identity and idle-resource gate")
    write_new(queue / "owner.json", identity)''')
    queue = replace_once(queue, '''        while same_process(args.inputs_pid, birth):
            time.sleep(30)
''', '')
    queue = replace_once(queue, '''        while not gpu_idle():
            time.sleep(30)''', '''        from h_gpu_availability_v2 import verify_available
        while True:
            gpu_report = verify_available()
            if gpu_report["available"]:
                write_new(queue / "gpu_readiness_v2.json", gpu_report)
                break
            emit(events, "waiting_for_verified_gpu_resources", blockers=[x["blockers"] for x in gpu_report["observations"]])
            time.sleep(30)''')
    queue = queue.replace('with_name("execute_h_grouped_l1248_recovery.py")',
                          'with_name("execute_h_grouped_l1248_recovery_v2.py")')
    queue = queue.replace('runner.recovery20260911.log', 'runner.recovery20260917_v5.log')
    queue = queue.replace('"queue_recovered_after_host_interruption", upstream_pid=args.inputs_pid',
                          '"queue_recovered_after_wddm_gate_block", upstream_pid=None')
    saved.append(save('recover_h_grouped_l1248_queue_v2.py', queue))

    final = (ROOT / 'analysis/continue_multienv_final_evidence.py').read_text(encoding='utf-8')
    final = final.replace('parser.add_argument("--inputs-pid", type=int, required=True)',
                          'parser.add_argument("--inputs-pid", type=int, default=0)')
    final = final.replace('parser.add_argument("--source-pid", type=int, required=True)',
                          'parser.add_argument("--source-pid", type=int, default=0)')
    final = replace_once(final, '    args = parser.parse_args()', '''    parser.add_argument("--compatibility-marker", required=True)
    args = parser.parse_args()
    if args.compatibility_marker != "analysis/continue_multienv_final_evidence.py":
        raise RuntimeError("Wrong downstream routing marker")''')
    start = final.index('    inputs_birth, inputs_command = process_identity(')
    end = final.index('    output.mkdir(parents=True)', start)
    final = final[:start] + '''    if args.inputs_pid or args.source_pid:
        raise RuntimeError("Inputs/source must be bound through completed records")
    inputs_status = read(inputs_output / "status.json")
    source_completion_path = source_output / "completion.json"
    source_completion = read(source_completion_path)
    verified_input = read(ROOT / "docs/revision/evidence/multienv_inputs_completion_verified_20260917.json")
    if inputs_status.get("status") != "completed" or verified_input["coordinator_status_sha256"] != sha256(inputs_output / "status.json"):
        raise RuntimeError("Completed input binding differs")
    if sha256(source_completion_path) != "481767f5c2b05137c83e1ee61ec0528af2b6ef81bbb8bbaa5b13aa5fc54a383b":
        raise RuntimeError("Sealed source completion differs")
    if source_completion.get("completed_count") != 117 or source_completion.get("target_data_loaded"):
        raise RuntimeError("Completed source scope differs")
    inputs_birth = source_birth = None
    inputs_command = source_command = []
    h_birth, h_command = process_identity(args.h_pid, ("recover_h_grouped_l1248_queue_v2.py", str(h_root)))
    h_queue_recovery = True
''' + final[end:]
    final = replace_once(final, '''        while (live(args.inputs_pid, inputs_birth) or live(args.source_pid, source_birth)
               or live(args.h_pid, h_birth)):''', '''        while live(args.h_pid, h_birth):''')
    final = replace_once(final, '''        "code_sha256": code_hashes}''', '''        "code_sha256": code_hashes,
        "completed_upstreams_bound_without_identity_holders": True,
        "input_completion_sha256": sha256(inputs_output / "status.json"),
        "source_completion_sha256": sha256(source_completion_path)}''')
    saved.append(save('continue_multienv_final_evidence_v2.py', final))
    print(json.dumps(saved, indent=2))


if __name__ == '__main__':
    main()
