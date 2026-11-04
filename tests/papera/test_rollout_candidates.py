"""Only immutable retained candidates may extend the old best/last API."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from rollout.policy import candidate_record


class RolloutCandidateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)
        (self.run / 'checkpoints').mkdir()
        self.config = {'steps': 60000, 'candidate_save_every': 5000}
        self.row = {'checkpoint': 'step_045000.pt', 'step': 45000, 'saved_step': 45000,
                    'validation_step': 45000, 'validation_source': 'metrics.jsonl:validation',
                    'validation': {'step': 45000, 'event': 'validation', 'prior_action_l1': .2}}
        self.index = {'version': 'papera-act-checkpoint-candidates-v1',
                      'run_signature': 'signature', 'candidates': [self.row]}
        self.write()

    def write(self):
        (self.run / 'checkpoints/candidates.json').write_text(json.dumps(self.index), encoding='utf-8')

    def test_valid_candidate_and_unchanged_legacy_without_index(self):
        self.assertEqual(candidate_record(self.run, 'step_045000', 'signature', self.config), self.row)
        (self.run / 'checkpoints/candidates.json').unlink()
        for legacy in ('best', 'last'):
            self.assertIsNone(candidate_record(self.run, legacy, 'signature', {}))

    def test_rejects_arbitrary_path_unknown_step_and_bad_cadence(self):
        for name in ('../last', 'step_045000.pt', '/tmp/checkpoint', 'step_45000', 'step_000000',
                     'step_045001', 'step_065000', 'step_050000'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                candidate_record(self.run, name, 'signature', self.config)

    def test_rejects_wrong_signature_duplicate_and_step_evidence(self):
        original = copy.deepcopy(self.index)
        mutations = [lambda x: x.update(run_signature='other'),
                     lambda x: x['candidates'].append(copy.deepcopy(self.row)),
                     lambda x: x['candidates'][0].update(saved_step=40000),
                     lambda x: x['candidates'][0].update(validation_step=40000),
                     lambda x: x['candidates'][0]['validation'].update(step=40000)]
        for mutate in mutations:
            self.index = copy.deepcopy(original)
            mutate(self.index)
            self.write()
            with self.assertRaises(ValueError):
                candidate_record(self.run, 'step_045000', 'signature', self.config)

    def test_unscheduled_validation_is_not_invented(self):
        self.row.update(validation=None, validation_step=None, validation_source=None)
        self.write()
        self.assertIsNone(candidate_record(self.run, 'step_045000', 'signature', self.config)['validation'])
        self.row['validation_step'] = 45000
        self.write()
        with self.assertRaises(ValueError):
            candidate_record(self.run, 'step_045000', 'signature', self.config)


if __name__ == '__main__':
    unittest.main()
