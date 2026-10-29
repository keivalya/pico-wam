import tempfile
from pathlib import Path
import unittest
from scripts.evidence_protocol import write_json,digest
from followup.s4_imitation import validate_result,read_checked,load

class S4ImitationTests(unittest.TestCase):
    def test_pairing_and_raw_aggregates(self):
        r=dict(episode_seeds=[4,5],stage='test',max_coverage=[.2,.8],mean_max_coverage=.5,success=[False,True],success_rate=.5)
        validate_result(r,[4,5])
        with self.assertRaises(ValueError): validate_result(r,[5,4])
        with self.assertRaises(ValueError): validate_result(dict(r,mean_max_coverage=.6),[4,5])
        with self.assertRaises(ValueError): validate_result(dict(r,note='truncated'),[4,5])
        with self.assertRaises(ValueError): validate_result(dict(r,max_coverage=[float('nan'),.8]),[4,5])

    def test_sealed_manifest_and_source_changes(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);src=p/'checkpoint';src.write_bytes(b'frozen')
            f=p/'manifest.json';write_json(f,{'files':{str(src):digest(src)}})
            f.with_suffix('.sha256').write_text(digest(f))
            load(p)
            src.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'Source changed'):load(p)
            f.write_text(f.read_text()+' ')
            with self.assertRaisesRegex(ValueError,'Changed result'):read_checked(f)

if __name__=='__main__':unittest.main()
