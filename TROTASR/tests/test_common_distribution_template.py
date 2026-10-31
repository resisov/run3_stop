"""One CR drawing path, with unchanged measured/prefit numerical inputs."""
from copy import deepcopy
from pathlib import Path
from unittest import TestCase, mock
import numpy as np
from TROTASR.tests.test_prediction_plots import raw_fixture, model_records
from TROTASR.tests.test_nominal_plot_adapters import physical_fixture
from TROTASR.workflows.prediction_plot_inputs import combine, flat_blocks
from TROTASR.workflows.plotting_gnn_controls import (
    prepare, distribution_blocks, dy_distribution_blocks, POOLED)
from TROTASR.workflows import plotting_nominal_distributions as common
from TROTASR.workflows import plotting_gnn_controls as gnn
from TROTASR.workflows import plotting_card_predictions as combined
from TROTASR.workflows.renderers import gnn_control as historical


class CommonDistributionTemplateTests(TestCase):
    def test_identical_function_object_and_render_options(self):
        self.assertIs(gnn.draw_distribution, common.draw_distribution)
        self.assertIs(combined.draw_distribution, common.draw_distribution)
        with common.nominal_renderer(2024):
            physical=common.make_records(common.physical_boundary(physical_fixture()),
                                         ['lowdm'],['LLCR'])[0][-1]
        record=combine([model_records()['2024/LLCR/lowdm']])
        for blocks,rotation in (([physical],0.0),(distribution_blocks(record,'LLCR'),90.0)):
            with mock.patch.object(common.renderer,'draw_flat_blocks',return_value={}) as draw:
                common.draw_distribution(blocks,Path('unused'),'x',109.82)
                self.assertEqual(draw.call_args.kwargs,dict(xlabel='x',reference_style=True,
                    show_yields=True,luminosity_fb=109.82,uncertainty_label_override=None,
                    x_tick_rotation=rotation))
                self.assertTrue(all(b['reference_style'] for b in draw.call_args.args[0]))

    def test_all_card_numbers_and_nb_projection_preserved(self):
        records=model_records()
        for region in ('LLCR','QCDCR','GCR'):
            record=combine([records['2024/'+region+'/lowdm']])
            before=deepcopy(record)
            old=flat_blocks(record,'lowdm',region,pool_llcr_display=False)
            new=distribution_blocks(record,region)
            self.assertEqual(sum(b['nbin'] for b in new),10)
            for a,b in zip(old,new):
                for field in ('background','background_unc','background_stat_unc','data','data_unc'):
                    np.testing.assert_array_equal(a[field],b[field])
                for group in a['groups']:
                    np.testing.assert_array_equal(a['groups'][group],b['groups'][group])
                self.assertFalse(b['unit_area'])
                self.assertEqual(b['xlabels'],['[0, 0.709)','[0.709, 0.867)',
                    '[0.867, 0.928)','[0.928, 0.961)','[0.961, 1]'])
                for key in ('figure_width','main_panel_ymax_factor','annotation','label_fontsize'):
                    self.assertNotIn(key,b)
            np.testing.assert_array_equal(record['background'],before['background'])

    def test_auxiliary_dy_matches_old_numbers_not_old_drawing_code(self):
        payload=prepare(*raw_fixture(),True)
        with mock.patch.object(historical,'CATEGORIES',POOLED):
            old=historical.flattened(payload,'DYCR')
        with common.nominal_renderer(2024):
            blocks=dy_distribution_blocks(payload)
        for field,prior in [('background','background'),('data','data')]:
            np.testing.assert_allclose(np.concatenate([b[field] for b in blocks]),old[prior],rtol=1e-14)
        np.testing.assert_allclose(np.concatenate([b['background_unc']**2 for b in blocks]),
                                   old['background2'],rtol=1e-14)
        for group in common.renderer.GROUP_ORDER:
            np.testing.assert_allclose(np.concatenate([b['groups'][group] for b in blocks]),
                                       old['groups'].get(group,np.zeros(10)),rtol=1e-14)
