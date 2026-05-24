import os
import pandas as pd
import pytest
import src

from src.gparse import _omic_generator, parse_gdc


# ---------------------------------------------------------------------------
# Helpers to write minimal TSV files mimicking GDC output formats
# ---------------------------------------------------------------------------

RNA_HEADER = "# comment line\n"
RNA_COLS = "gene_id\tgene_name\tgene_type\tunstranded\ttpm_unstranded\n"
RNA_ROWS = (
    "N_unmapped\t\t\tN_unmapped\tN_unmapped\n"
    "ENSG001\tGENE1\tprotein_coding\t100\t1.5\n"
    "ENSG002\tGENE2\tprotein_coding\t200\t2.5\n"
)

GENE_COLS = "gene_id\tcopy_number\n"
GENE_ROWS = "ENSG001\t1\nENSG002\t-1\n"

METH_ROWS = "cg00001\t0.85\ncg00002\t0.42\n"

MIRNA_COLS = "miRNA_ID\tread_count\tcross_mapped\n"
MIRNA_ROWS = "hsa-mir-1\t500\tN\nhsa-mir-2\t300\tN\n"


def _write_sample_folder(base, folder_name, content, ext, with_annotation=False):
    folder = base / folder_name
    folder.mkdir(parents=True)
    (folder / f"data{ext}").write_text(content)
    if with_annotation:
        (folder / "annotations.txt").write_text("flagged\n")


def _clinical(folder_name, file_col='File ID'):
    return pd.DataFrame({file_col: [folder_name], 'Case ID': ['C1'], 'Sample ID': ['S1']})


# ---------------------------------------------------------------------------
# parse_gdc — RNA-seq
# ---------------------------------------------------------------------------

class TestParseGdcRna:
    def test_basic_rna_returns_dict_with_rna_key(self, tmp_path):
        folder = 'folder_rna1'
        _write_sample_folder(tmp_path, folder, RNA_HEADER + RNA_COLS + RNA_ROWS, '.tsv')
        clinical = _clinical(folder)
        result = parse_gdc([folder], str(tmp_path), 'Tumor', clinical, '.tsv', 'rna')
        assert len(result) == 1
        assert 'Tumor1' in result
        assert 'rna' in result['Tumor1']
        assert 'cd' in result['Tumor1']

    def test_rna_dataframe_has_expected_columns(self, tmp_path):
        folder = 'folder_rna2'
        _write_sample_folder(tmp_path, folder, RNA_HEADER + RNA_COLS + RNA_ROWS, '.tsv')
        clinical = _clinical(folder)
        result = parse_gdc([folder], str(tmp_path), 'T', clinical, '.tsv', 'rna')
        rna_df = result['T1']['rna']
        if hasattr(rna_df, 'compute'):
            rna_df = rna_df.compute()
        for col in ['gene_id', 'gene_name', 'gene_type', 'unstranded', 'tpm_unstranded']:
            assert col in rna_df.columns

    def test_rna_drops_na_gene_id_rows(self, tmp_path):
        folder = 'folder_rna3'
        _write_sample_folder(tmp_path, folder, RNA_HEADER + RNA_COLS + RNA_ROWS, '.tsv')
        clinical = _clinical(folder)
        result = parse_gdc([folder], str(tmp_path), 'T', clinical, '.tsv', 'rna')
        rna_df = result['T1']['rna']
        if hasattr(rna_df, 'compute'):
            rna_df = rna_df.compute()
        # N_unmapped row should be dropped (it's mapped as NaN via na_values)
        assert rna_df['gene_id'].isna().sum() == 0

    def test_multiple_rna_folders(self, tmp_path):
        for i in range(1, 4):
            folder = f'rna_folder_{i}'
            _write_sample_folder(tmp_path, folder, RNA_HEADER + RNA_COLS + RNA_ROWS, '.tsv')
        clinical = pd.DataFrame({
            'File ID': [f'rna_folder_{i}' for i in range(1, 4)],
            'Case ID': [f'C{i}' for i in range(1, 4)],
            'Sample ID': [f'S{i}' for i in range(1, 4)],
        })
        result = parse_gdc(
            [f'rna_folder_{i}' for i in range(1, 4)],
            str(tmp_path), 'T', clinical, '.tsv', 'rna'
        )
        assert len(result) == 3


# ---------------------------------------------------------------------------
# parse_gdc — annotation skipping
# ---------------------------------------------------------------------------

class TestParseGdcAnnotationSkipping:
    def test_folder_with_annotations_is_skipped(self, tmp_path):
        good_folder = 'good_folder'
        bad_folder = 'bad_folder'
        _write_sample_folder(tmp_path, good_folder, RNA_HEADER + RNA_COLS + RNA_ROWS, '.tsv')
        _write_sample_folder(tmp_path, bad_folder, RNA_HEADER + RNA_COLS + RNA_ROWS, '.tsv',
                             with_annotation=True)
        clinical = pd.DataFrame({
            'File ID': [good_folder, bad_folder],
            'Case ID': ['C1', 'C2'],
            'Sample ID': ['S1', 'S2'],
        })
        result = parse_gdc(
            [good_folder, bad_folder], str(tmp_path), 'T', clinical, '.tsv', 'rna'
        )
        assert len(result) == 1
        assert 'T1' in result

    def test_all_annotated_returns_empty_dict(self, tmp_path):
        folder = 'annotated_folder'
        _write_sample_folder(tmp_path, folder, RNA_HEADER + RNA_COLS + RNA_ROWS, '.tsv',
                             with_annotation=True)
        clinical = _clinical(folder)
        result = parse_gdc([folder], str(tmp_path), 'T', clinical, '.tsv', 'rna')
        assert result == {}


# ---------------------------------------------------------------------------
# parse_gdc — deduplication
# ---------------------------------------------------------------------------

class TestParseGdcDedup:
    def test_same_folder_listed_twice_parsed_once(self, tmp_path):
        folder = 'dup_folder'
        _write_sample_folder(tmp_path, folder, RNA_HEADER + RNA_COLS + RNA_ROWS, '.tsv')
        clinical = _clinical(folder)
        # Pass the same folder twice
        result = parse_gdc([folder, folder], str(tmp_path), 'T', clinical, '.tsv', 'rna')
        assert len(result) == 1


# ---------------------------------------------------------------------------
# parse_gdc — miRNA
# ---------------------------------------------------------------------------

class TestParseGdcMirna:
    def test_mirna_returns_dict_with_mirna_key(self, tmp_path):
        folder = 'mirna_folder'
        _write_sample_folder(tmp_path, folder, MIRNA_COLS + MIRNA_ROWS, '.tsv')
        clinical = _clinical(folder)
        result = parse_gdc([folder], str(tmp_path), 'T', clinical, '.tsv', 'mirna')
        assert 'T1' in result
        assert 'mirna' in result['T1']

    def test_mirna_is_pandas_not_dask(self, tmp_path):
        folder = 'mirna_folder2'
        _write_sample_folder(tmp_path, folder, MIRNA_COLS + MIRNA_ROWS, '.tsv')
        clinical = _clinical(folder)
        result = parse_gdc([folder], str(tmp_path), 'T', clinical, '.tsv', 'mirna')
        # miRNA is loaded with pandas, not dask
        import pandas as pd
        assert isinstance(result['T1']['mirna'], pd.DataFrame)


# ---------------------------------------------------------------------------
# parse_gdc — methylation
# ---------------------------------------------------------------------------

class TestParseGdcMeth:
    def test_meth_returns_dict_with_meth_key(self, tmp_path):
        folder = 'meth_folder'
        _write_sample_folder(tmp_path, folder, METH_ROWS, '.txt')
        clinical = _clinical(folder, file_col='MFile ID')
        result = parse_gdc([folder], str(tmp_path), 'T', clinical, '.txt', 'meth')
        assert 'T1' in result
        assert 'meth' in result['T1']

    def test_meth_has_site_and_betav_columns(self, tmp_path):
        folder = 'meth_folder2'
        _write_sample_folder(tmp_path, folder, METH_ROWS, '.txt')
        clinical = _clinical(folder, file_col='MFile ID')
        result = parse_gdc([folder], str(tmp_path), 'T', clinical, '.txt', 'meth')
        meth_df = result['T1']['meth']
        if hasattr(meth_df, 'compute'):
            meth_df = meth_df.compute()
        assert 'site' in meth_df.columns
        assert 'betav' in meth_df.columns
