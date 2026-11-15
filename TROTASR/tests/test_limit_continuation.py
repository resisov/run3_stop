"""Numerical policies used by the canonical finite recovery stage."""
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from TROTASR.utils.io import read_json, write_json
from TROTASR.utils.paths import ROOT
from TROTASR.workflows.combine_limits import (
    accepted_fit_record, historical_success_commands, reconcile_interrupted_attempts, replacement_card)
from TROTASR.workflows.limit_recovery import choose_range, command_for, target_nll


def block(rows):
    return 'Will search for NLL crossing by bisection\n'+''.join(
        'At r = %s: delta(nll) = %s\n'%(r,n) for r,n in rows)


class NumericalContinuationTests(unittest.TestCase):
    def test_interrupted_attempt_uses_real_receipt_or_remains_consumed_unknown(self):
        with TemporaryDirectory(dir=ROOT) as directory:
            work = Path(directory)
            state = work/'workflow.json'
            attempts = []
            for name in ('interrupted', 'saved'):
                here = work/name
                here.mkdir()
                attempts.append(dict(name=name, command=['combine', name], exit_code=None,
                    log=str((here/'combine.log').relative_to(ROOT)),
                    output=str((here/'limit.root').relative_to(ROOT))))
            write_json(work/'saved/validation.json', dict(attempts[1], exit_code=0, passed=True))
            reconcile_interrupted_attempts(dict(attempts=attempts, pid=123), state)
            saved = read_json(state)
            self.assertIsNone(saved['pid'])
            self.assertIsNone(saved['attempts'][0]['exit_code'])
            self.assertFalse(saved['attempts'][0]['passed'])
            self.assertTrue(saved['attempts'][0]['interruption'])
            self.assertEqual(saved['attempts'][1]['exit_code'], 0)

    def test_accepted_fit_with_missing_ancillary_files(self):
        with TemporaryDirectory(dir=ROOT) as directory:
            work = Path(directory)
            root = work/'limit.root'
            root.write_bytes(b'accepted product')
            point = dict(model='T2tt', mass='mStop1200_mLSP500', model_revision='era_merged')
            attempt = dict(exit_code=0, passed=True, output=str(root.relative_to(ROOT)),
                           log=str((work/'combine.log').relative_to(ROOT)),
                           command=['combine', 'recorded'])
            record = dict(point, status='complete', accepted_attempt=attempt,
                          validation=dict(quantiles={'.025': 1, '.16': 2, '.5': 3, '.84': 4, '.975': 5}))
            self.assertEqual(accepted_fit_record(point, record, str(work/'workflow.json'))['output'], str(root))
            root.unlink()
            with self.assertRaises(FileNotFoundError):
                accepted_fit_record(point, record, str(work/'workflow.json'))

    def test_malformed_optional_numerical_recipe_is_inapplicable(self):
        with TemporaryDirectory(dir=ROOT) as directory:
            output = Path(directory)
            point = dict(model='T2tt', mass='mStop1200_mLSP500')
            write_json(output/'successful_recipes.json',
                       {'points': {'T2tt/mStop1200_mLSP500': {'command': ['combine'], 'name': 'old'}}})
            commands, status = historical_success_commands(point, output/'workspace.root', output)
            self.assertEqual(commands, [])
            self.assertEqual(status['status'], 'inapplicable')

    def test_replacement_point_cannot_fall_back_to_old_cards_or_receipts(self):
        with TemporaryDirectory(dir=ROOT) as directory:
            products = Path(directory)/'new_signal'
            card = products/'grid/datacards/card.txt'
            card.parent.mkdir(parents=True)
            card.write_text('test card')
            point = dict(card=str(card.relative_to(ROOT)), card_sha256='legacy', fit_ready=True)
            write_json(card.with_suffix('.manifest.json'), dict(signal_input_revision='new_uuid',
                card_sha256=point['card_sha256'], sr_data_blinded=True))
            self.assertEqual(replacement_card(point, products, 'new_uuid'), card)
            with self.assertRaises(ValueError):
                replacement_card(point, products.parent, 'new_uuid')
            card.unlink()
            with self.assertRaises(FileNotFoundError):
                replacement_card(point, products, 'new_uuid')

    def test_exhausted_current_scan_expands_seed_not_limit(self):
        seed,evidence=choose_range([('refined',block([(20,.06786),(30,.14213),(39.84375,.23368)]),40)],40)
        self.assertGreater(seed,300)
        self.assertLess(seed,1000)
        self.assertEqual(len(evidence),1)
        self.assertEqual(evidence[0]['sampled_r'],39.84375)

    def test_adequate_range_does_not_expand_for_asimov_warning(self):
        text='Best fit of asimov dataset is at r = 0.027\n'+block([(10,6.66),(5,2.08),(4.76,1.918)])
        self.assertEqual(choose_range([('initial',text,20)],40),(40,[]))

    def test_partial_sequence_uses_only_measured_median(self):
        median=block([(20,.06786),(30,.14213),(39.84375,.23368)])
        for count in (2,3,4):
            text=median+block([(20,1e-15),(30,2e-15),(39,3e-15)])*(count-1)
            seed,evidence=choose_range([('refined',text,40)],40)
            self.assertEqual(seed,430)
            self.assertEqual(len(evidence),1)
            self.assertEqual(evidence[0]['quantile'],.5)
            self.assertEqual(evidence[0]['search_scope'],'median_only_from_incomplete_sequence')

    def test_partial_without_valid_median_does_not_use_later_blocks(self):
        text=block([])+block([(20,.06786),(30,.14213),(39.84375,.23368)])
        self.assertEqual(choose_range([('refined',text,40)],40),(40,[]))

    def test_extra_searches_are_not_assigned_quantiles(self):
        text=block([(20,.06786),(30,.14213),(39.84375,.23368)])*6
        self.assertEqual(choose_range([('refined',text,40)],40),(40,[]))

    def test_no_scan_is_not_invented_range_evidence(self):
        self.assertEqual(choose_range([('refined','failed NaN',40)],40),(40,[]))

    def test_no_boundary_samples_no_extrapolation(self):
        self.assertEqual(choose_range([('a',block([(1,.01),(2,.02),(3,.03)]),40)],40),(40,[]))

    def test_zero_nll_no_division_by_zero(self):
        self.assertEqual(choose_range([('a',block([(20,0),(30,0),(39,0)]),40)],40),(40,[]))

    def test_measured_crossing_prevents_range_failure_claim(self):
        self.assertEqual(choose_range([('a',block([(20,.1),(30,3),(39,.5)]),40)],40),(40,[]))

    def test_finite_cap_and_invalid_range(self):
        with self.assertRaises(ValueError):choose_range([('a',block([(20,1e-12),(30,2e-12),(39,3e-12)]),40)],40)
        with self.assertRaises(ValueError):choose_range([],float('nan'))
        with self.assertRaises(ValueError):choose_range([],0)

    def test_target_matches_median_convention(self):
        self.assertAlmostEqual(target_nll(.5),1.920729410347062,places=12)
