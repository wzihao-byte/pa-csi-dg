"""Reuse frozen native AP inputs for the full H/H-S gated-model controls.

No CSI is reconstructed, no model is imported, and no training/inference runs.
The saved R input manifests establish array identity only: their source split
memberships MUST NOT replace the original H or H-S memberships.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np

REPO_ROOT=Path(__file__).resolve().parents[2]
DEFAULT_RUNTIME=Path("D:/pa-csi-dg-revision-runtime")
HISTSTYLE="native_time_wrapped_linear_feature_unwrap_v1"
DIFFERENTIAL="native_time_differential_phase"
VARIANTS=(HISTSTYLE,DIFFERENTIAL)
VARIANT_ALIASES={"native_historical":HISTSTYLE,"native_differential":DIFFERENTIAL}
ENVS=("E1","E2","E3")
PINS={
 "native_amp_arrays_v1/dataset_manifest.json":"2d1ae9d9f570fcc0e137297daaa9dc3aaf504558c9628ef101ef75238a252f73",
 "modality_arrays_v1/dataset_manifest.json":"fed82c5eff308ce864d192abd0a50b129bac031f81c659274941badc932842c0",
 "native_histstyle_phase_v1/dataset_manifest.json":"812e3394f06f21abf04d3ef2366c92ab2eeca7afb2478419da60671820da6c67",
 "native_amp_v1/summary.json":"320c40c808550555a0b579cf3a998dbb35e707c792411f4ce7eaf1941022e8b7",
 "native_phase_v2/summary.json":"7391a7e4f0491c32fbdde821cd48853bee111732f0abc68432a2417a1f182186",
}
def require(ok:bool,message:str)->None:
 if not ok:raise RuntimeError(message)
def read(path:Path)->Any:return json.loads(path.read_text(encoding="utf-8-sig"))
def value_hash(value:Any)->str:
 return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
def sha256(path:Path)->str:
 h=hashlib.sha256()
 with path.open("rb") as f:
  for data in iter(lambda:f.read(8*1024*1024),b""):h.update(data)
 return h.hexdigest()
def ordered_identity(rows:list[dict])->list[tuple]:
 return [tuple(r[k] for k in ("sample_id","environment","index","label","subject_id","acquisition_id")) for r in rows]
def check_manifest_pair(left:dict,right:dict,env:str)->None:
 """Reject reordered/relabelled/native-grid-mismatched AP conditions."""
 require(ordered_identity(left["test"])==ordered_identity(right["test"]),"phase variants differ in ordered sample/label/subject/acquisition identity")
 require(left["test"]==right["test"],"phase variants differ in DAT/MAT row provenance")
 a,b=left["files"][env],right["files"][env]
 for kind in ("amp","label"):
  require(a["data_files"][kind]==b["data_files"][kind],"native phase conditions do not share identical "+kind+" files")
  require(a["actual_file_sha256"][kind]==b["actual_file_sha256"][kind],"native phase conditions differ in "+kind+" digest")

def check_historical_receipts(historical:dict,original_files:dict,env:str)->None:
 """Tie MAT row provenance to the same frozen original H amp/label receipts."""
 entry=historical["files"][env]
 for kind in ("amp","label"):
  saved=entry["data_files"][kind];original=original_files[kind]
  require(Path(saved["path"]).resolve()==Path(original["path"]).resolve(),"historical provenance path differs from original H "+kind)
  for key in ("sha256","shape","dtype"):require(saved[key]==original[key],"historical provenance receipt differs from original H "+kind)
  require(entry["actual_file_sha256"][kind]==original["sha256"],"historical provenance actual digest differs from original H "+kind)

def phase_input_patch(base_config:dict,variant:str,audit:dict)->dict:
 """Return a clone changing only AP input bindings and the required unwrap flag.

Source environment order, target routing, splits, model/loss/optimizer/budget
are retained. Preprocessed phase MUST NOT be unwrapped a second time.
A resolved H config's source_file_records/target file_records are updated when
present. Job names/output locations/protocol provenance are owned by the caller.
"""
 variant=VARIANT_ALIASES.get(variant,variant)
 require(variant in VARIANTS,"unknown phase variant")
 require(audit.get("status")=="passed" and audit.get("rows")==9000,"require completed input-alignment audit")
 bindings=audit["variants"][variant]["environments"]
 cfg=deepcopy(base_config)
 require(cfg["data"].get("format")=="env_npy_triplets","require original H input format")
 require(cfg["data"].get("unwrap_phase") is True,"phase patch must start from the historical H loader")
 sources=list(cfg["data"]["env_files"])
 for env in sources:
  require(env in ENVS,"unknown source environment")
  cfg["data"]["env_files"][env]={kind:bindings[env]["file_records"][kind]["path"] for kind in ("amp","phase","label")}
 cfg["data"]["unwrap_phase"]=False
 if "source_file_records" in cfg:
  require(set(cfg["source_file_records"])==set(sources),"source file-record environments differ")
  cfg["source_file_records"]={env:deepcopy(bindings[env]["file_records"]) for env in sources}
 if "target_data" in cfg:
  target=cfg["target_data"]["target_env"]
  require(target==cfg["target_env"] and target not in sources,"target/source routing mismatch")
  data=cfg["target_data"]["data"]
  require(list(data["env_files"])==[target] and data.get("unwrap_phase") is True,"require original deferred target loader")
  data["env_files"][target]={kind:bindings[target]["file_records"][kind]["path"] for kind in ("amp","phase","label")}
  data["unwrap_phase"]=False
  if "file_records" in cfg["target_data"]:cfg["target_data"]["file_records"]=deepcopy(bindings[target]["file_records"])
 allowed={"data","source_file_records","target_data"}
 require(all(cfg[k]==base_config[k] for k in base_config if k not in allowed),"non-input scientific configuration changed")
 return cfg

def audit_phase_alignment(runtime:Path=DEFAULT_RUNTIME,repo_root:Path=REPO_ROOT)->dict:
 """Validate completed manifests, array bytes, MAT/DAT identities and grid links.

All actual files read are recorded with SHA256. Historical H signal arrays are
read-only references: their frozen metadata is cross-checked, while their labels
are rehashed and compared. No raw DAT/MAT or historical signal is reconstructed.
"""
 runtime=Path(runtime);repo_root=Path(repo_root);data=runtime/"data"
 hashes={};verified_arrays={};array_cache={};label_cache={}
 def bind(path:Path,expected:str|None=None)->str:
  path=Path(path);key=str(path.resolve())
  digest=hashes.get(key)
  if digest is None:digest=sha256(path);hashes[key]=digest
  if expected is not None:require(digest==expected,"frozen artifact SHA256 differs: "+key)
  return digest
 manifests={}
 for relative,expected in PINS.items():
  path=data/relative;bind(path,expected);manifests[relative]=read(path)
 for folder in ("native_amp_arrays_v1","modality_arrays_v1","native_histstyle_phase_v1"):
  m=manifests[folder+"/dataset_manifest.json"]
  require(m.get("complete") and m.get("rows")==9000 and m.get("original_partition_order_preserved",m.get("historical_partition_sample_order_preserved")),"incomplete or unaligned compilation")
  bind(data/folder/"compile_config.json",m["artifacts_sha256"]["compile_config.json"])
 require(bind(repo_root/"docs/revision/evidence/modality_dataset_manifest.json")==PINS["modality_arrays_v1/dataset_manifest.json"],"repository modality manifest differs from frozen runtime manifest")
 amp_root=data/"native_amp_arrays_v1"
 diff_root=data/"modality_arrays_v1"
 hist_root=data/"native_histstyle_phase_v1"
 hist_compile=read(hist_root/"compile_config.json");diff_compile=read(diff_root/"compile_config.json");amp_compile=read(amp_root/"compile_config.json")
 require(hist_compile["native_dataset_manifest_sha256"]==PINS["native_amp_arrays_v1/dataset_manifest.json"],"historical-style native parent differs")
 require(diff_compile["native_dataset_manifest_sha256"]==PINS["native_amp_arrays_v1/dataset_manifest.json"],"differential native parent differs")
 require(amp_compile["preparation_summary_sha256"]==PINS["native_amp_v1/summary.json"],"native amplitude parent differs")
 require(diff_compile["phase_summary_sha256"]==PINS["native_phase_v2/summary.json"],"differential preparation parent differs")
 qa_path=hist_root/"identity_and_grid_qa.json";bind(qa_path,manifests["native_histstyle_phase_v1/dataset_manifest.json"]["artifacts_sha256"]["identity_and_grid_qa.json"])
 qa=read(qa_path)
 require(qa.get("passed") and qa["rows"]==9000 and qa["same_native_C_elapsed_endpoint_grid_definition"],"native historical-style grid evidence is incomplete")
 amp_summary=manifests["native_amp_v1/summary.json"];phase_summary=manifests["native_phase_v2/summary.json"]
 require(amp_summary["complete_9000"] and not amp_summary["failures"] and phase_summary["complete_9000"] and not phase_summary["failures"],"native preparation incomplete")
 variants={v:{"loader_unwrap_phase":False,"phase_unwrap_axis_preserved_but_disabled":-1,"environments":{}} for v in VARIANTS}
 env_checks={};prep_records=0;historical_config=repo_root/"outputs_tpp_level8_ablation/l1248_v1/runs/E1/seed_42/H11/attempt_001/payload/run/resolved_config.json"
 bind(historical_config);h=read(historical_config)
 historical_files={**h["source_file_records"],h["target_env"]:h["target_data"]["file_records"]}
 def array_check(item:dict,expected:str)->dict:
  path=Path(item["path"]);key=str(path.resolve())
  require(item["sha256"]==expected,"metadata array digest mismatch")
  if key in verified_arrays:return verified_arrays[key]
  bind(path,expected);values=np.load(path,mmap_mode="r",allow_pickle=False)
  try:
   require(list(values.shape)==item["shape"] and str(values.dtype)==item["dtype"],"array header differs from frozen metadata")
   for start in range(0,len(values),32):require(np.isfinite(values[start:start+32]).all(),"nonfinite values in "+key)
   if values.ndim==1:label_cache[key]=np.asarray(values).copy()
   else:require(values.shape==(3000,850,90) and values.dtype==np.float32,"native AP signal shape/dtype differs")
   result={"path":str(path),"sha256":expected,"shape":list(values.shape),"dtype":str(values.dtype),"all_finite":True}
  finally:
   if hasattr(values,"_mmap"):values._mmap.close()
  verified_arrays[key]=result;return result
 for env in ENVS:
  phase_meta={}
  for variant,folder in ((HISTSTYLE,hist_root),(DIFFERENTIAL,diff_root)):
   p=folder/variant/"manifests"/f"deferred_target_{env}.json"
   m=manifests[folder.name+"/dataset_manifest.json"];bind(p,m["artifacts_sha256"][p.relative_to(folder).as_posix()]);phase_meta[variant]=read(p)
  check_manifest_pair(phase_meta[HISTSTYLE],phase_meta[DIFFERENTIAL],env)
  native_path=amp_root/"native_time_linear/manifests"/f"deferred_target_{env}.json"
  bind(native_path,manifests["native_amp_arrays_v1/dataset_manifest.json"]["artifacts_sha256"][native_path.relative_to(amp_root).as_posix()])
  native=read(native_path);require(native["test"]==phase_meta[HISTSTYLE]["test"],"native amplitude row order differs")
  historical_path=diff_root/"historical_feature_axis_unwrap/manifests"/f"deferred_target_{env}.json"
  bind(historical_path,manifests["modality_arrays_v1/dataset_manifest.json"]["artifacts_sha256"][historical_path.relative_to(diff_root).as_posix()])
  historical=read(historical_path)
  check_historical_receipts(historical,historical_files[env],env)
  require(ordered_identity(historical["test"])==ordered_identity(native["test"]),"historical H row identity/order differs")
  original_labels=Path(historical_files[env]["label"]["path"]);bind(original_labels,historical_files[env]["label"]["sha256"])
  labels=np.load(original_labels,allow_pickle=False);require(labels.shape==(3000,),"historical label shape differs")
  rows=native["test"];require(len(rows)==3000 and [r["index"] for r in rows]==list(range(3000)),"native array indices are reordered or incomplete")
  require([int(x) for x in labels]==[r["label"] for r in rows],"historical labels differ from native row labels")
  records_path=hist_root/"environment_records"/f"{env}.json"
  bind(records_path,manifests["native_histstyle_phase_v1/dataset_manifest.json"]["artifacts_sha256"][records_path.relative_to(hist_root).as_posix()])
  phase_records=read(records_path)["records"]
  require(ordered_identity(phase_records)==ordered_identity(rows),"historical-style phase record order differs")
  for old,row,record in zip(historical["test"],rows,phase_records):
   require(row["identity_source_mat_path"]==old["raw_path"] and row["identity_source_mat_sha256"]==old["raw_sha256"],"native row does not map to historical MAT identity")
   stem=Path(row["raw_path"]).stem
   ap=data/"native_amp_v1/records"/(stem+".json");pp=data/"native_phase_v2/records"/(stem+".json")
   ah=bind(ap,amp_summary["record_sha256"][row["identity_source_mat_path"]]);bind(pp,phase_summary["record_sha256"][row["identity_source_mat_path"]])
   ar,pr=read(ap),read(pp)
   require(ar["verified"] and ar["all_complex_csi_and_packet_ids_equal"],"native amplitude DAT/MAT identity evidence fails")
   for r in (ar,pr):
    require(r["environment"]==env and r["index"]==row["index"] and r["mat_path"]==row["identity_source_mat_path"] and r["dat_path"]==row["raw_path"] and env+":"+r["subject_id"]==row["subject_id"] and r["acquisition_id"]==row["acquisition_id"],"native preparation record identity differs")
   require(ar["mat_sha256"]==row["identity_source_mat_sha256"] and ar["dat_sha256"]==row["raw_sha256"]==pr["raw_sha256"],"native parent signal digest differs")
   require(pr["parent_record_sha256"]==ah==record["native_preparation_record_sha256"],"phase native-parent binding differs")
   require(record["duration_seconds"]==ar["duration_seconds"] and record["grid_points"]==850,"phase/amplitude duration or grid length differs")
   grid=np.linspace(0.0,float(round(ar["duration_seconds"]*1e6)),850)
   require(hashlib.sha256(grid.astype("<f8").tobytes()).hexdigest()==record["grid_float64_sha256"],"native endpoint grid hash differs")
   prep_records+=2
  for variant in VARIANTS:
   meta=phase_meta[variant];entry=meta["files"][env]
   file_records={kind:array_check(entry["data_files"][kind],entry["actual_file_sha256"][kind]) for kind in ("amp","phase","label")}
   native_labels=label_cache[str(Path(file_records["label"]["path"]).resolve())]
   require(np.array_equal(labels,native_labels),"native and original H label arrays differ")
   variants[variant]["environments"][env]={"file_records":file_records,"ordered_identity_sha256":value_hash(ordered_identity(rows)),"target_manifest_path":str((hist_root if variant==HISTSTYLE else diff_root)/variant/"manifests"/f"deferred_target_{env}.json")}
  env_checks[env]={"rows":3000,"historical_MAT_to_native_DAT_identity_verified":True,"historical_amp_label_receipts_match_original_H":True,"same_native_amplitude_labels_and_endpoint_grid":True,"source_manifest_memberships_reused_for_H":False}
  print("Phase alignment verified:",env,flush=True)
 for variant in VARIANTS:
  variants[variant]["env_files"]={env:{kind:variants[variant]["environments"][env]["file_records"][kind]["path"] for kind in ("amp","phase","label")} for env in ENVS}
  variants[variant]["file_records"]={env:variants[variant]["environments"][env]["file_records"] for env in ENVS}
 return {"schema":"h-main-phase-input-alignment/v1","variant_aliases":dict(VARIANT_ALIASES),"status":"passed","created_at_utc":datetime.now(timezone.utc).isoformat(),"rows":9000,"variants":variants,"environments":env_checks,"source_artifacts_sha256":hashes,"verified_native_arrays":verified_arrays,"native_preparation_records_rehashed":prep_records,"new_native_arrays_written":False,"raw_DAT_or_MAT_decoding_performed":False,"historical_signal_arrays_newly_rehashed":False,"model_training_or_target_inference_performed":False,"source_split_policy":"Retain each exact original H/H-S source membership and sampler; compiled R source manifests supply row provenance only.","historical_reference_loader":"original float64 phase -> np.unwrap(axis=-1) -> float32","native_loader":"already-preprocessed float32 phase; unwrap_phase=False","phase_comparison":"Native historical-style AP versus native differential AP at identical amplitude/grid/labels. Historical H/H-S reference versus native arms changes the complete time/phase input chain.","planned_new_runs":72,"completed_historical_TPP_references_reused":36,"cohort_exclusions":[],"physical_phase_calibration_claimed":False}

def main()->None:
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--runtime",type=Path,default=DEFAULT_RUNTIME);p.add_argument("--output",type=Path,default=Path(__file__).with_name("phase_alignment_audit.json"));a=p.parse_args()
 require(not a.output.exists(),"preserve existing phase audit; choose a new audit output path")
 audit=audit_phase_alignment(a.runtime);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(audit,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
 print(json.dumps({k:audit[k] for k in ("status","rows","native_preparation_records_rehashed","planned_new_runs","completed_historical_TPP_references_reused")},indent=2))
if __name__=="__main__":main()

