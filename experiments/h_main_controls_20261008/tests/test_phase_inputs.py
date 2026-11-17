"""Meaningful negative checks for input alignment and H configuration preservation."""
from copy import deepcopy
import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from phase_inputs import HISTSTYLE, DIFFERENTIAL, check_manifest_pair, check_historical_receipts, phase_input_patch

def records(env,variant):
 return {kind:dict(path=f"/frozen/{variant}/{env}_{kind}.npy",sha256="a"*64,shape=[3000] if kind=="label" else [3000,850,90],dtype="int32" if kind=="label" else "float32",all_finite=True) for kind in ("amp","phase","label")}

def audit_fixture():
 return dict(status="passed",rows=9000,variants={v:{"environments":{e:{"file_records":records(e,v)} for e in ("E1","E2","E3")}} for v in (HISTSTYLE,DIFFERENTIAL)})

def config_fixture():
 return {"target_env":"E1","seed_list":[42],"model":{"name":"full_H_gated_tpp","tpp_levels":[1,2,4,8],"antenna_layout":{"mode":"tx","num_rx":1,"num_tx":3,"num_subcarriers":30}},
 "training":{"batch_size":32,"epochs":200,"updates_per_epoch":150,"selection_metric":"accuracy"},"losses":{"lambda_supcon":.05,"supcon_positive_mode":"global_class_instance_views"},"split":{"train_hash":"fixed_H","val_hash":"fixed_H"},
 "data":{"format":"env_npy_triplets","unwrap_phase":True,"phase_unwrap_axis":-1,"env_files":{"E2":{"amp":"oldA","phase":"oldP","label":"oldL"},"E3":{"amp":"oldA","phase":"oldP","label":"oldL"}}},
 "source_file_records":{"E2":{},"E3":{}},"target_data":{"target_env":"E1","data":{"format":"env_npy_triplets","unwrap_phase":True,"phase_unwrap_axis":-1,"env_files":{"E1":{"amp":"oldA","phase":"oldP","label":"oldL"}}},"file_records":{}}}

def manifest_fixture():
 rows=[dict(sample_id=f"E1:{i:04d}",environment="E1",index=i,label=i%2,subject_id="E1:S01",acquisition_id=f"A{i}",raw_path=f"DAT/{i}.dat",raw_sha256="b"*64,identity_source_mat_path=f"MAT/{i}.mat",identity_source_mat_sha256="c"*64) for i in range(3)]
 return {"test":rows,"files":{"E1":{"data_files":records("E1","native_linear"),"actual_file_sha256":{k:"a"*64 for k in ("amp","phase","label")}}}}

class PhaseInputTests(unittest.TestCase):
 def test_native_patch_preserves_full_H_model_training_loss_and_split(self):
  original=config_fixture();before=deepcopy(original);new=phase_input_patch(original,HISTSTYLE,audit_fixture())
  self.assertEqual(original,before)
  for key in ("model","training","losses","split","seed_list","target_env"):self.assertEqual(new[key],original[key])
  self.assertFalse(new["data"]["unwrap_phase"]);self.assertFalse(new["target_data"]["data"]["unwrap_phase"])
  self.assertEqual(list(new["data"]["env_files"]),["E2","E3"])
  self.assertEqual(list(new["target_data"]["data"]["env_files"]),["E1"])
  self.assertIn(HISTSTYLE,new["data"]["env_files"]["E2"]["phase"])
 def test_logical_aliases_bind_to_actual_frozen_variants(self):
  for alias,actual in (("native_historical",HISTSTYLE),("native_differential",DIFFERENTIAL)):
   self.assertEqual(phase_input_patch(config_fixture(),alias,audit_fixture()),phase_input_patch(config_fixture(),actual,audit_fixture()))
 def test_double_patch_rejected_instead_of_double_unwrap_or_silent_drift(self):
  new=phase_input_patch(config_fixture(),DIFFERENTIAL,audit_fixture())
  with self.assertRaisesRegex(RuntimeError,"historical H loader"):phase_input_patch(new,HISTSTYLE,audit_fixture())
 def test_unknown_or_failed_input_audit_rejected(self):
  with self.assertRaisesRegex(RuntimeError,"unknown phase"):phase_input_patch(config_fixture(),"guessed_phase",audit_fixture())
  failed=audit_fixture();failed["status"]="failed"
  with self.assertRaisesRegex(RuntimeError,"completed"):phase_input_patch(config_fixture(),HISTSTYLE,failed)
 def test_reordered_phase_samples_rejected(self):
  left=manifest_fixture();right=deepcopy(left);right["test"][0],right["test"][1]=right["test"][1],right["test"][0]
  with self.assertRaisesRegex(RuntimeError,"ordered"):check_manifest_pair(left,right,"E1")
 def test_changed_label_or_MAT_mapping_rejected(self):
  left=manifest_fixture();right=deepcopy(left);right["test"][1]["label"]=5
  with self.assertRaisesRegex(RuntimeError,"ordered"):check_manifest_pair(left,right,"E1")
  right=deepcopy(left);right["test"][1]["identity_source_mat_sha256"]="d"*64
  with self.assertRaisesRegex(RuntimeError,"provenance"):check_manifest_pair(left,right,"E1")
 def test_amplitude_grid_file_mismatch_rejected(self):
  left=manifest_fixture();right=deepcopy(left);right["files"]["E1"]["data_files"]["amp"]["path"]="/other/grid.npy"
  with self.assertRaisesRegex(RuntimeError,"identical amp"):check_manifest_pair(left,right,"E1")
 def test_unchanged_phase_pairs_accepted(self):
  left=manifest_fixture();check_manifest_pair(left,deepcopy(left),"E1")
 def test_provenance_manifest_must_bind_original_H_amp_label_receipts(self):
  m=manifest_fixture();original=deepcopy(m["files"]["E1"]["data_files"])
  check_historical_receipts(m,original,"E1")
  original["amp"]["sha256"]="b"*64
  with self.assertRaisesRegex(RuntimeError,"original H amp"):check_historical_receipts(m,original,"E1")
 def test_target_source_overlap_rejected(self):
  cfg=config_fixture();cfg["data"]["env_files"]["E1"]=cfg["data"]["env_files"].pop("E3");cfg["source_file_records"]={"E1":{},"E2":{}}
  with self.assertRaisesRegex(RuntimeError,"target/source"):phase_input_patch(cfg,HISTSTYLE,audit_fixture())
if __name__=="__main__":unittest.main()

