import io
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from experiments.wimans_20261014.acquisition import acquire_one


class AcquisitionRecoveryTests(unittest.TestCase):
    row = {"sample_id": "act_fixture_1", "raw_relative_path": "wifi_csi/mat/act_fixture_1.mat"}
    payload = b"MATLAB 5.0 MAT-file" + b"fixture" * 20

    def response(self, payload):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as z:
            z.writestr("act_fixture_1.mat", payload)
        class Response:
            content = stream.getvalue()
            def raise_for_status(self):
                pass
        return Response()

    def test_interrupted_raw_rename_recovery_keeps_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw" / self.row["raw_relative_path"]
            raw.parent.mkdir(parents=True)
            raw.write_bytes(self.payload)
            with patch("experiments.wimans_20261014.acquisition.requests.get", return_value=self.response(self.payload)):
                receipt = acquire_one(root, self.row)
            self.assertIn("redownload", receipt["origin"])
            self.assertEqual(raw.read_bytes(), self.payload)
            # Accepted receipt is verified locally and makes no network call.
            with patch("experiments.wimans_20261014.acquisition.requests.get", side_effect=AssertionError("network")):
                self.assertEqual(acquire_one(root, self.row), receipt)

    def test_unreceipted_mismatch_does_not_replace_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw" / self.row["raw_relative_path"]
            raw.parent.mkdir(parents=True)
            raw.write_bytes(self.payload)
            with patch("experiments.wimans_20261014.acquisition.requests.get", return_value=self.response(self.payload + b"changed")):
                with self.assertRaisesRegex(ValueError, "preserve and investigate"):
                    acquire_one(root, self.row)
            self.assertEqual(raw.read_bytes(), self.payload)
            self.assertFalse((root / "acquisition/records/act_fixture_1.json").exists())

    def test_accepted_file_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("experiments.wimans_20261014.acquisition.requests.get", return_value=self.response(self.payload)):
                acquire_one(root, self.row)
            (root / "raw" / self.row["raw_relative_path"]).write_bytes(self.payload + b"changed")
            with self.assertRaisesRegex(ValueError, "Previously accepted raw bytes changed"):
                acquire_one(root, self.row)


if __name__ == "__main__":
    unittest.main()
