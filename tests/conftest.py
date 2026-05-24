import pandas as pd
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Minimal clinical DataFrames
# ---------------------------------------------------------------------------

@pytest.fixture
def make_clinical():
    """Return a factory that builds a one-row clinical DataFrame."""
    def _make(case_id, file_id, gfile_id=None, mfile_id=None, sample_id=None):
        data = {'Case ID': [case_id], 'File ID': [file_id]}
        if gfile_id is not None:
            data['GFile ID'] = [gfile_id]
        if mfile_id is not None:
            data['MFile ID'] = [mfile_id]
        if sample_id is not None:
            data['Sample ID'] = [sample_id]
        return pd.DataFrame(data)
    return _make


# ---------------------------------------------------------------------------
# Small omic DataFrames (n_features=5)
# ---------------------------------------------------------------------------

FEATURES = ['G1', 'G2', 'G3', 'G4', 'G5']

@pytest.fixture
def make_rna_df():
    def _make(seed=0):
        rng = np.random.default_rng(seed)
        return pd.DataFrame({
            'gene_id': FEATURES,
            'unstranded': rng.integers(0, 1000, 5).astype(float),
            'tpm_unstranded': rng.random(5).astype(np.float32),
        })
    return _make


@pytest.fixture
def make_gene_df():
    def _make(seed=0):
        rng = np.random.default_rng(seed)
        return pd.DataFrame({
            'gene_id': FEATURES,
            'copy_number': rng.integers(-2, 4, 5).astype(float),
        })
    return _make


@pytest.fixture
def make_meth_df():
    def _make(seed=0):
        rng = np.random.default_rng(seed)
        return pd.DataFrame({
            'site': [f'cg{i:05d}' for i in range(5)],
            'betav': rng.random(5).astype(np.float32),
        })
    return _make


@pytest.fixture
def make_mirna_df():
    def _make(seed=0):
        rng = np.random.default_rng(seed)
        return pd.DataFrame({
            'miRNA_ID': [f'hsa-mir-{i}' for i in range(5)],
            'read_count': rng.integers(0, 500, 5).astype(float),
        })
    return _make


# ---------------------------------------------------------------------------
# Pre-built 3-sample merged dicts for aggreg tests
# ---------------------------------------------------------------------------

@pytest.fixture
def three_sample_rna_dict(make_clinical, make_rna_df):
    """rna_dict with 3 samples: case IDs C1, C2, C3."""
    return {
        'Tumor1': {'cd': make_clinical('C1', 'F1'), 'rna': make_rna_df(0)},
        'Tumor2': {'cd': make_clinical('C2', 'F2'), 'rna': make_rna_df(1)},
        'Tumor3': {'cd': make_clinical('C3', 'F3'), 'rna': make_rna_df(2)},
    }


@pytest.fixture
def three_sample_cnv_dict(make_clinical, make_gene_df):
    return {
        'GTW1': {'cd': make_clinical('C1', 'GF1', gfile_id='GF1'), 'gene': make_gene_df(0)},
        'GTW2': {'cd': make_clinical('C2', 'GF2', gfile_id='GF2'), 'gene': make_gene_df(1)},
        'GTW3': {'cd': make_clinical('C3', 'GF3', gfile_id='GF3'), 'gene': make_gene_df(2)},
    }


@pytest.fixture
def three_sample_meth_dict(make_clinical, make_meth_df):
    return {
        'Meth1': {'cd': make_clinical('C1', 'MF1', mfile_id='MF1'), 'meth': make_meth_df(0)},
        'Meth2': {'cd': make_clinical('C2', 'MF2', mfile_id='MF2'), 'meth': make_meth_df(1)},
        'Meth3': {'cd': make_clinical('C3', 'MF3', mfile_id='MF3'), 'meth': make_meth_df(2)},
    }


@pytest.fixture
def alignment_df_3():
    """Alignment DataFrame joining C1, C2, C3 across all three omics."""
    return pd.DataFrame({
        'Case ID':   ['C1', 'C2', 'C3'],
        'File ID':   ['F1', 'F2', 'F3'],
        'GFile ID':  ['GF1', 'GF2', 'GF3'],
        'MFile ID':  ['MF1', 'MF2', 'MF3'],
    })


@pytest.fixture
def three_sample_mirna_dict(make_clinical, make_mirna_df):
    return {
        'MiR1': {'cd': make_clinical('C1', 'MiRF1', sample_id='S1'), 'mirna': make_mirna_df(0)},
        'MiR2': {'cd': make_clinical('C2', 'MiRF2', sample_id='S2'), 'mirna': make_mirna_df(1)},
        'MiR3': {'cd': make_clinical('C3', 'MiRF3', sample_id='S3'), 'mirna': make_mirna_df(2)},
    }


@pytest.fixture
def alignment_df_2():
    """Alignment DataFrame for 2-omic (RNA + miRNA) merge on Case ID."""
    return pd.DataFrame({
        'Case ID':    ['C1', 'C2', 'C3'],
        'File ID':    ['F1', 'F2', 'F3'],
        'MiRFile ID': ['MiRF1', 'MiRF2', 'MiRF3'],
    })
