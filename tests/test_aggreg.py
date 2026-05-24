import copy
import numpy as np
import pandas as pd
import pytest
import dask.dataframe as dd

from src.aggreg import (
    merge_3dicts,
    merge_2dicts,
    create_count_table,
    build_clinical_df,
    build_anndata,
)


# ---------------------------------------------------------------------------
# merge_3dicts
# ---------------------------------------------------------------------------

class TestMerge3Dicts:
    def test_basic_merge_returns_all_three_samples(
        self, three_sample_rna_dict, three_sample_cnv_dict, three_sample_meth_dict, alignment_df_3
    ):
        result = merge_3dicts(
            three_sample_rna_dict, three_sample_cnv_dict, three_sample_meth_dict,
            alignment_df_3, 'Tm'
        )
        assert len(result) == 3
        assert set(result.keys()) == {'Tm1', 'Tm2', 'Tm3'}

    def test_merged_entry_has_required_omic_keys(
        self, three_sample_rna_dict, three_sample_cnv_dict, three_sample_meth_dict, alignment_df_3
    ):
        result = merge_3dicts(
            three_sample_rna_dict, three_sample_cnv_dict, three_sample_meth_dict,
            alignment_df_3, 'Tm'
        )
        for entry in result.values():
            assert set(entry.keys()) == {'cd', 'rna', 'gene', 'meth'}

    def test_cd_contains_correct_case_id(
        self, three_sample_rna_dict, three_sample_cnv_dict, three_sample_meth_dict, alignment_df_3
    ):
        result = merge_3dicts(
            three_sample_rna_dict, three_sample_cnv_dict, three_sample_meth_dict,
            alignment_df_3, 'Tm'
        )
        case_ids = {result[k]['cd']['Case ID'].iloc[0] for k in result}
        assert case_ids == {'C1', 'C2', 'C3'}

    def test_missing_case_in_one_dict_is_skipped(
        self, three_sample_rna_dict, three_sample_cnv_dict, three_sample_meth_dict,
        alignment_df_3, make_clinical, make_rna_df
    ):
        # Add a 4th entry to alignment_df that has no match in any omic dict
        extra_row = pd.DataFrame({'Case ID': ['C99'], 'File ID': ['FX'],
                                   'GFile ID': ['GFX'], 'MFile ID': ['MFX']})
        alignment_extended = pd.concat([alignment_df_3, extra_row], ignore_index=True)
        result = merge_3dicts(
            three_sample_rna_dict, three_sample_cnv_dict, three_sample_meth_dict,
            alignment_extended, 'Tm'
        )
        assert len(result) == 3

    def test_missing_case_in_cnv_dict_is_skipped(
        self, three_sample_rna_dict, three_sample_meth_dict, alignment_df_3, make_clinical, make_gene_df
    ):
        # cnv_dict only has C1 and C2 — C3 should be skipped
        partial_cnv = {
            'GTW1': {'cd': make_clinical('C1', 'GF1', gfile_id='GF1'), 'gene': make_gene_df(0)},
            'GTW2': {'cd': make_clinical('C2', 'GF2', gfile_id='GF2'), 'gene': make_gene_df(1)},
        }
        result = merge_3dicts(
            three_sample_rna_dict, partial_cnv, three_sample_meth_dict,
            alignment_df_3, 'Tm'
        )
        assert len(result) == 2

    def test_omic_dataframes_are_the_same_objects(
        self, three_sample_rna_dict, three_sample_cnv_dict, three_sample_meth_dict, alignment_df_3
    ):
        result = merge_3dicts(
            three_sample_rna_dict, three_sample_cnv_dict, three_sample_meth_dict,
            alignment_df_3, 'Tm'
        )
        # Verify rna DataFrame identity — not a copy
        assert result['Tm1']['rna'] is three_sample_rna_dict['Tumor1']['rna']


# ---------------------------------------------------------------------------
# merge_2dicts
# ---------------------------------------------------------------------------

class TestMerge2Dicts:
    def test_basic_merge_returns_all_samples(
        self, three_sample_rna_dict, three_sample_mirna_dict, alignment_df_2
    ):
        result = merge_2dicts(
            three_sample_rna_dict, three_sample_mirna_dict, 'rna', 'mirna',
            alignment_df_2, 'Tw'
        )
        assert len(result) == 3
        assert set(result.keys()) == {'Tw1', 'Tw2', 'Tw3'}

    def test_both_omic_cat_keys_present(
        self, three_sample_rna_dict, three_sample_mirna_dict, alignment_df_2
    ):
        result = merge_2dicts(
            three_sample_rna_dict, three_sample_mirna_dict, 'rna', 'mirna',
            alignment_df_2, 'Tw'
        )
        for entry in result.values():
            assert 'rna' in entry
            assert 'mirna' in entry
            assert 'cd' in entry

    def test_custom_omic_cat_names(
        self, three_sample_rna_dict, three_sample_mirna_dict, alignment_df_2
    ):
        # Build dicts whose inner omic key matches the custom names we will pass
        d1 = {k: {'cd': v['cd'], 'omic_a': v['rna']} for k, v in three_sample_rna_dict.items()}
        d2 = {k: {'cd': v['cd'], 'omic_b': v['mirna']} for k, v in three_sample_mirna_dict.items()}
        result = merge_2dicts(d1, d2, 'omic_a', 'omic_b', alignment_df_2, 'Tw')
        for entry in result.values():
            assert 'omic_a' in entry
            assert 'omic_b' in entry

    def test_non_rna_first_omic(
        self, three_sample_cnv_dict, three_sample_meth_dict, alignment_df_3
    ):
        # Merge CNV + methylation (no RNA involved)
        alignment_2 = alignment_df_3[['Case ID', 'GFile ID', 'MFile ID']].copy()
        result = merge_2dicts(
            three_sample_cnv_dict, three_sample_meth_dict, 'gene', 'meth',
            alignment_2, 'Tw'
        )
        assert len(result) == 3
        for entry in result.values():
            assert 'gene' in entry
            assert 'meth' in entry

    def test_missing_case_skipped(
        self, three_sample_rna_dict, three_sample_mirna_dict, alignment_df_2,
        make_clinical, make_mirna_df
    ):
        partial_mirna = {
            'MiR1': three_sample_mirna_dict['MiR1'],
            'MiR2': three_sample_mirna_dict['MiR2'],
        }
        result = merge_2dicts(
            three_sample_rna_dict, partial_mirna, 'rna', 'mirna',
            alignment_df_2, 'Tw'
        )
        assert len(result) == 2

    def test_join_key_sample_id(
        self, make_clinical, make_rna_df, make_mirna_df
    ):
        rna = {
            'T1': {'cd': make_clinical('C1', 'F1', sample_id='S1'), 'rna': make_rna_df(0)},
        }
        mirna = {
            'M1': {'cd': make_clinical('C1', 'MiRF1', sample_id='S1'), 'mirna': make_mirna_df(0)},
        }
        alignment = pd.DataFrame({'Sample ID': ['S1'], 'File ID': ['F1'], 'MiRFile ID': ['MiRF1']})
        result = merge_2dicts(rna, mirna, 'rna', 'mirna', alignment, 'Tw', join_key='Sample ID')
        assert len(result) == 1
        assert result['Tw1']['cd']['Sample ID'].iloc[0] == 'S1'


# ---------------------------------------------------------------------------
# create_count_table
# ---------------------------------------------------------------------------

class TestCreateCountTable:
    @pytest.fixture
    def dict_cases(self, make_rna_df):
        return {
            'Tumor1': {'rna': make_rna_df(0)},
            'Tumor2': {'rna': make_rna_df(1)},
        }

    @pytest.fixture
    def dict_controls(self, make_rna_df):
        return {
            'Normal1': {'rna': make_rna_df(2)},
        }

    def test_output_shape(self, dict_cases, dict_controls):
        result = create_count_table(dict_cases, dict_controls, 'rna', 'unstranded', 'gene_id')
        # columns: gene_id + 2 cases + 1 control = 4 columns; 5 features
        assert result.shape == (5, 4)

    def test_column_names_match_dict_keys(self, dict_cases, dict_controls):
        result = create_count_table(dict_cases, dict_controls, 'rna', 'unstranded', 'gene_id')
        assert list(result.columns) == ['gene_id', 'Tumor1', 'Tumor2', 'Normal1']

    def test_label_column_values(self, dict_cases, dict_controls):
        result = create_count_table(dict_cases, dict_controls, 'rna', 'unstranded', 'gene_id')
        from tests.conftest import FEATURES
        assert list(result['gene_id']) == FEATURES

    def test_non_contiguous_keys_work(self, make_rna_df):
        # Keys that cannot be reconstructed with a simple counter
        d1 = {'Alpha': {'rna': make_rna_df(0)}, 'Gamma': {'rna': make_rna_df(1)}}
        d2 = {'Delta': {'rna': make_rna_df(2)}}
        result = create_count_table(d1, d2, 'rna', 'unstranded', 'gene_id')
        assert 'Alpha' in result.columns
        assert 'Gamma' in result.columns
        assert 'Delta' in result.columns

    def test_values_match_source(self, dict_cases, dict_controls):
        result = create_count_table(dict_cases, dict_controls, 'rna', 'unstranded', 'gene_id')
        expected = dict_cases['Tumor1']['rna']['unstranded'].values
        assert np.allclose(result['Tumor1'].values, expected)


# ---------------------------------------------------------------------------
# build_clinical_df
# ---------------------------------------------------------------------------

class TestBuildClinicalDf:
    @pytest.fixture
    def small_merged(self, make_clinical, make_rna_df):
        return {
            'Tm1': {'cd': make_clinical('C1', 'F1'), 'rna': make_rna_df(0)},
            'Tm2': {'cd': make_clinical('C2', 'F2'), 'rna': make_rna_df(1)},
        }

    def test_returns_one_row_per_sample(self, small_merged):
        from src.aggreg import build_clinical_df
        result = build_clinical_df(small_merged)
        assert len(result) == 2

    def test_key_column_added(self, small_merged):
        from src.aggreg import build_clinical_df
        result = build_clinical_df(small_merged)
        assert 'key' in result.columns
        assert set(result['key']) == {'Tm1', 'Tm2'}

    def test_does_not_mutate_source_dfs(self, small_merged):
        from src.aggreg import build_clinical_df
        original_cols_tm1 = list(small_merged['Tm1']['cd'].columns)
        build_clinical_df(small_merged)
        assert list(small_merged['Tm1']['cd'].columns) == original_cols_tm1
        assert 'key' not in small_merged['Tm1']['cd'].columns

    def test_calling_twice_gives_same_result(self, small_merged):
        from src.aggreg import build_clinical_df
        r1 = build_clinical_df(small_merged)
        r2 = build_clinical_df(small_merged)
        pd.testing.assert_frame_equal(r1, r2)


# ---------------------------------------------------------------------------
# build_anndata
# ---------------------------------------------------------------------------

class TestBuildAnndata:
    @pytest.fixture
    def rna_merged(self, make_clinical, make_rna_df):
        return {
            'Tm1': {'cd': make_clinical('C1', 'F1'), 'rna': make_rna_df(0)},
            'Tm2': {'cd': make_clinical('C2', 'F2'), 'rna': make_rna_df(1)},
            'Tm3': {'cd': make_clinical('C3', 'F3'), 'rna': make_rna_df(2)},
        }

    def test_x_shape(self, rna_merged):
        adata = build_anndata(rna_merged, 'rna', 'gene_id', 'unstranded')
        assert adata.X.shape == (3, 5)  # 3 samples × 5 features

    def test_obs_names_match_dict_keys(self, rna_merged):
        adata = build_anndata(rna_merged, 'rna', 'gene_id', 'unstranded')
        assert list(adata.obs_names) == ['Tm1', 'Tm2', 'Tm3']

    def test_var_names_match_features(self, rna_merged):
        from tests.conftest import FEATURES
        adata = build_anndata(rna_merged, 'rna', 'gene_id', 'unstranded')
        assert list(adata.var_names) == FEATURES

    def test_obs_contains_clinical_columns(self, rna_merged):
        adata = build_anndata(rna_merged, 'rna', 'gene_id', 'unstranded')
        assert 'Case ID' in adata.obs.columns

    def test_x_values_match_source(self, rna_merged):
        adata = build_anndata(rna_merged, 'rna', 'gene_id', 'unstranded')
        expected = rna_merged['Tm1']['rna'].set_index('gene_id')['unstranded'].values.astype(np.float32)
        assert np.allclose(adata.X[0], expected)

    def test_extra_layers_shape(self, rna_merged):
        adata = build_anndata(
            rna_merged, 'rna', 'gene_id', 'unstranded',
            extra_layers={'tpm': 'tpm_unstranded'}
        )
        assert 'tpm' in adata.layers
        assert adata.layers['tpm'].shape == (3, 5)

    def test_extra_layers_values(self, rna_merged):
        adata = build_anndata(
            rna_merged, 'rna', 'gene_id', 'unstranded',
            extra_layers={'tpm': 'tpm_unstranded'}
        )
        expected = rna_merged['Tm1']['rna'].set_index('gene_id')['tpm_unstranded'].values.astype(np.float32)
        assert np.allclose(adata.layers['tpm'][0], expected)

    def test_dask_dataframes_are_computed(self, make_clinical, make_rna_df):
        merged = {
            'Tm1': {
                'cd': make_clinical('C1', 'F1'),
                'rna': dd.from_pandas(make_rna_df(0), npartitions=1),
            },
            'Tm2': {
                'cd': make_clinical('C2', 'F2'),
                'rna': dd.from_pandas(make_rna_df(1), npartitions=1),
            },
        }
        adata = build_anndata(merged, 'rna', 'gene_id', 'unstranded')
        assert adata.X.shape == (2, 5)

    def test_no_extra_layers_by_default(self, rna_merged):
        adata = build_anndata(rna_merged, 'rna', 'gene_id', 'unstranded')
        assert len(adata.layers) == 0
