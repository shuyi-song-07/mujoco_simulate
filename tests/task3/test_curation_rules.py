"""Legacy strict raw-contiguous ACT window contract, independent of curation."""
import unittest
from dataset.task3.curation_rules import build_windows


def rows(count=8):
    return [{"source_episode":"s/0","session":"s","episode_index":0,
        "control_epoch":1,"raw_frame_index":i,"source_control_tick":i*2,
        "original_time_s":i/30.,"keep":True,"segment_id":0,
        "terminal":{"actual_done":False}} for i in range(count)]


class BoundedWindowTests(unittest.TestCase):
    def test_padding_is_not_real_future_or_terminal(self):
        r=rows(2); w=build_windows(r,length=5,observation_horizon=3)
        self.assertEqual(w[0]["source_indices"],[0,1,1,1,1])
        self.assertEqual(w[0]["loss_mask"],[True,True,False,False,False])
        self.assertEqual(w[0]["observation_is_pad"],[True,True,False])
        self.assertFalse(any(w[0]["actual_terminal_mask"]))
        self.assertTrue(w[0]["truncated"])
        r[1]["terminal"]["actual_done"]=True
        w=build_windows(r,length=5)
        self.assertEqual(w[0]["actual_terminal_mask"],[False,True,False,False,False])
        self.assertFalse(w[0]["truncated"])

    def test_filter_epoch_source_and_raw_time_changes_are_hard_window_limits(self):
        for kind in ("filter","epoch","source","time","raw","fragment"):
            with self.subTest(kind=kind):
                r=rows()
                if kind=="filter": r[3]["keep"]=False
                if kind=="epoch": r[4]["control_epoch"]=2
                if kind=="source": r[4]["source_episode"]="other"
                if kind=="time": r[4]["original_time_s"]+=1
                if kind=="raw": r[4]["raw_frame_index"]+=5
                if kind=="fragment": r[4]["segment_id"]=2
                w=build_windows(r,length=8)
                self.assertFalse(any(2 in x["source_indices"] and 5 in x["source_indices"] for x in w))

    def test_unpadded_windows_require_complete_length(self):
        self.assertEqual(build_windows(rows(2),length=5,pad=False),[])
        self.assertEqual(len(build_windows(rows(8),length=5,pad=False)),4)

    def test_invalid_horizons_rejected(self):
        for value in (0,-1,True,1.5):
            with self.assertRaises(ValueError): build_windows(rows(),length=value)

if __name__=="__main__": unittest.main()
