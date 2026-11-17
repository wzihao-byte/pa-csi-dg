"""Quantify native-clock/resampling changes without training or scoring a model."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.revision_validation.study_io import bound_path, now, read, sha256, write


def comparison(reference, changed):
    a, b = reference.reshape(len(reference),-1).astype(np.float64), changed.reshape(len(changed),-1).astype(np.float64)
    difference = b-a
    energy = (a*a).sum(1)
    centered_a, centered_b = a-a.mean(1,keepdims=True), b-b.mean(1,keepdims=True)
    denominator = np.sqrt((centered_a**2).sum(1)*(centered_b**2).sum(1))
    correlation = np.divide((centered_a*centered_b).sum(1), denominator, out=np.full(len(a),np.nan), where=denominator>1e-12)
    rmse = np.sqrt((difference*difference).mean(1))
    relative = np.sqrt(np.divide((difference*difference).sum(1),energy,out=np.full(len(a),np.nan),where=energy>1e-12))
    return [{"float32_elements_exact": bool(np.array_equal(reference[i],changed[i])),
             "max_abs_error": float(np.abs(difference[i]).max()), "rmse": float(rmse[i]),
             "relative_l2_error": float(relative[i]) if np.isfinite(relative[i]) else None,
             "pearson_correlation": float(correlation[i]) if np.isfinite(correlation[i]) else None,
             "elements_abs_error_gt_1e_minus5": int(np.count_nonzero(np.abs(difference[i])>1e-5)),
             "reference_near_zero_energy": bool(energy[i]<=1e-12)} for i in range(len(a))]


def aggregate(rows):
    result = {"rows":len(rows), "exact_rows":sum(r["float32_elements_exact"] for r in rows),
              "rows_with_abs_error_gt_1e_minus5":sum(r["elements_abs_error_gt_1e_minus5"]>0 for r in rows)}
    for metric in ("max_abs_error","rmse","relative_l2_error","pearson_correlation"):
        values = [r[metric] for r in rows if r[metric] is not None]
        result[metric] = {"valid_rows":len(values), "min":float(min(values)), "median":float(np.median(values)),
                          "p95":float(np.quantile(values,.95)), "max":float(max(values))} if values else None
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-compiled",type=Path,required=True)
    parser.add_argument("--native-prepared",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    native,prepared,output=args.native_compiled.resolve(),args.native_prepared.resolve(),args.output.resolve()
    manifest=read(native/'dataset_manifest.json')
    if not manifest.get('complete') or manifest['rows']!=9000:
        raise RuntimeError('Complete native input compilation is required')
    for relative,digest in manifest['artifacts_sha256'].items():
        if sha256(bound_path(native,relative))!=digest:
            raise RuntimeError('Compiled native artifact changed')
    parent_summary=read(prepared/'summary.json')
    if sha256(prepared/'summary.json')!=read(native/'compile_config.json')['preparation_summary_sha256']:
        raise RuntimeError('Native preparation and compiled inputs do not match')
    if output.exists():
        raise FileExistsError('Preserve previous reconstruction audit')
    output.mkdir(parents=True)
    results,clock=[] ,[]
    pairs=(('historical_vs_native_merge','historical','native_time_legacy_merge'),
           ('historical_vs_native_linear','historical','native_time_linear'),
           ('native_merge_vs_native_linear','native_time_legacy_merge','native_time_linear'))
    source_hashes={}
    for environment in ('E1','E2','E3'):
        old=read(ROOT/f'experiments/revision_validation/verified_protocol/deferred_target_{environment}.json')
        spec=old['files'][environment]
        original_path=spec['data_files']['amp']['path']
        if sha256(original_path)!=spec['actual_file_sha256']['amp']:
            raise RuntimeError('Historical amplitude bytes differ')
        source_hashes[environment]=spec['actual_file_sha256']['amp']
        arrays={'historical':np.load(original_path,mmap_mode='r',allow_pickle=False)}
        for variant in ('native_time_legacy_merge','native_time_linear'):
            arrays[variant]=np.load(native/variant/f'{environment}_amp.npy',mmap_mode='r',allow_pickle=False)
        try:
            for start in range(0,3000,32):
                end=min(start+32,3000)
                batches={name:np.asarray(array[start:end],dtype=np.float32) for name,array in arrays.items()}
                for name,a,b in pairs:
                    values=comparison(batches[a],batches[b])
                    for index,value in enumerate(values,start):
                        identity=old['test'][index]
                        results.append({'comparison':name,'environment':environment,'index':index,'label':identity['label'],
                            'sample_id':identity['sample_id'],'subject_id':identity['subject_id'],**value})
            for identity in old['test']:
                name=identity['raw_path'];p=prepared/'records'/(Path(name).stem+'.json')
                if sha256(p)!=parent_summary['record_sha256'][name]:
                    raise RuntimeError('Native clock provenance record changed')
                row=read(p)
                clock.append({'environment':environment,'label':identity['label'],'sample_id':identity['sample_id'],
                    'duration_seconds':row['duration_seconds'],'packet_count':row['packet_count'],
                    'max_gap_seconds':row['max_gap_seconds'],'duplicate_timestamp_steps':row['duplicate_timestamp_steps'],
                    'uint32_wraps':row['uint32_wraps'],
                    'effective_packets_per_second':(row['packet_count']-1)/row['duration_seconds']})
        finally:
            for array in arrays.values():array._mmap.close()
        print(f'Reconstruction sensitivity measured: {environment}',flush=True)
    for name,rows in (('per_record_differences.csv',results),('native_clock_statistics.csv',clock)):
        with (output/name).open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    statistics={name:{'overall':aggregate([r for r in results if r['comparison']==name]),
        'by_environment':{env:aggregate([r for r in results if r['comparison']==name and r['environment']==env]) for env in ('E1','E2','E3')},
        'by_class':{str(label):aggregate([r for r in results if r['comparison']==name and r['label']==label]) for label in range(6)}} for name,_,_ in pairs}
    timing={env:{key:{'min':float(min(r[key] for r in clock if r['environment']==env)),
                      'median':float(np.median([r[key] for r in clock if r['environment']==env])),
                      'max':float(max(r[key] for r in clock if r['environment']==env))}
               for key in ('duration_seconds','packet_count','max_gap_seconds','effective_packets_per_second')}
            for env in ('E1','E2','E3')}
    report={'completed_at':now(),'rows':9000,'model_training_or_scoring':False,'all_comparisons_use_float32_model_inputs':True,
        'calibration_or_fixed_sampling_frequency_claimed':False,'statistics':statistics,'native_clock':timing,
        'native_dataset_manifest_sha256':sha256(native/'dataset_manifest.json'),'native_preparation_summary_sha256':sha256(prepared/'summary.json'),
        'historical_amplitude_sha256':source_hashes,'code_sha256':sha256(__file__),
        'artifacts_sha256':{p.name:sha256(p) for p in output.glob('*.csv')}}
    write(output/'summary.json',report)
    print({name:value['overall'] for name,value in statistics.items()},flush=True)


if __name__=='__main__':main()
