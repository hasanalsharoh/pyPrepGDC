import pandas as pd
import numpy as np
import anndata as ad


def merge_3dicts(rna_dict, cnv_dict, meth_dict, alignment_df, classific):
    """
    Merge three omics dicts (RNA, CNV, methylation) using a pre-built alignment DataFrame.

    O(1) lookup via reverse indices built once before the loop. The caller supplies
    alignment_df (e.g. tma/twa from the calling notebook) as the single source of truth
    for which samples belong together — avoids re-deriving alignment by scanning.

    Args:
        rna_dict:     parsed RNA-seq dict from parse_gdc
        cnv_dict:     parsed CNV dict from parse_gdc
        meth_dict:    parsed methylation dict from parse_gdc
        alignment_df: DataFrame with at least a 'Case ID' column joining all three omics
        classific:    string prefix for output keys (e.g. 'Tumor')

    Returns:
        dict keyed classific+i with 'cd', 'rna', 'gene', 'meth' entries
    """
    rna_idx  = {next(iter(v['cd']['Case ID'])): k for k, v in rna_dict.items()}
    cnv_idx  = {next(iter(v['cd']['Case ID'])): k for k, v in cnv_dict.items()}
    meth_idx = {next(iter(v['cd']['Case ID'])): k for k, v in meth_dict.items()}

    merged = {}
    for i, (_, row) in enumerate(alignment_df.iterrows(), 1):
        case_id  = row['Case ID']
        rna_key  = rna_idx.get(case_id)
        cnv_key  = cnv_idx.get(case_id)
        meth_key = meth_idx.get(case_id)
        if rna_key and cnv_key and meth_key:
            merged[f'{classific}{i}'] = {
                'cd':   alignment_df[alignment_df['Case ID'] == case_id].reset_index(drop=True),
                'rna':  rna_dict[rna_key]['rna'],
                'gene': cnv_dict[cnv_key]['gene'],
                'meth': meth_dict[meth_key]['meth'],
            }
    return merged


def merge_2dicts(rna_dict, omic2_dict, omic2_cat, alignment_df, classific, join_key='Case ID'):
    """
    Merge two omics dicts (RNA + one secondary omic) using a pre-built alignment DataFrame.

    O(1) lookup via reverse indices. Generic omic2_cat lets the caller name the second omic
    key (e.g. 'mirna', 'gene', 'meth').

    Args:
        rna_dict:     parsed RNA-seq dict from parse_gdc
        omic2_dict:   parsed second-omic dict from parse_gdc
        omic2_cat:    key name for the second omic in the output dict (e.g. 'mirna', 'gene')
        alignment_df: DataFrame with at least a join_key column joining both omics
        classific:    string prefix for output keys (e.g. 'Tumor')
        join_key:     column to join on (default 'Case ID'; use 'Sample ID' to preserve
                      tumor/normal pairs within the same case)

    Returns:
        dict keyed classific+i with 'cd', 'rna', and omic2_cat entries
    """
    rna_idx   = {next(iter(v['cd'][join_key])): k for k, v in rna_dict.items()}
    omic2_idx = {next(iter(v['cd'][join_key])): k for k, v in omic2_dict.items()}

    merged = {}
    for i, (_, row) in enumerate(alignment_df.iterrows(), 1):
        key_val   = row[join_key]
        rna_key   = rna_idx.get(key_val)
        omic2_key = omic2_idx.get(key_val)
        if rna_key and omic2_key:
            merged[f'{classific}{i}'] = {
                'cd':      alignment_df[alignment_df[join_key] == key_val].reset_index(drop=True),
                'rna':     rna_dict[rna_key]['rna'],
                omic2_cat: omic2_dict[omic2_key][omic2_cat],
            }
    return merged


def create_count_table(data_dict1, data_dict2, data_cat, cols, label):
    """
    Build a count/value matrix from two merged dicts (e.g. cases and controls).

    Iterates by dict key directly — no counter reconstruction, single concat at the end.

    Args:
        data_dict1: first merged dict (e.g. tumor samples)
        data_dict2: second merged dict (e.g. normal samples)
        data_cat:   omic key inside each sample ('rna', 'gene', 'meth', 'mirna')
        cols:       column holding count/value to extract (e.g. 'unstranded', 'betav')
        label:      column to use as the feature label (e.g. 'gene_id', 'site', 'miRNA_ID')

    Returns:
        DataFrame with label column followed by one column per sample
    """
    cols1   = {k: v[data_cat][cols].rename(k) for k, v in data_dict1.items()}
    cols2   = {k: v[data_cat][cols].rename(k) for k, v in data_dict2.items()}
    label_s = next(iter(data_dict1.values()))[data_cat][label]
    return pd.concat([label_s, *cols1.values(), *cols2.values()], axis=1)


def build_clinical_df(merged_dict):
    """
    Concatenate clinical DataFrames from a merged dict, adding a 'key' column.

    Non-mutating — original DataFrames in merged_dict are not modified.

    Args:
        merged_dict: output of merge_2dicts or merge_3dicts

    Returns:
        Single long-form DataFrame with all samples stacked; 'key' column holds the dict key
    """
    return pd.concat(
        [v['cd'].assign(key=k) for k, v in merged_dict.items()],
        axis=0,
        ignore_index=True,
    )


def build_anndata(merged_dict, data_cat, feature_col, count_col, extra_layers=None):
    """
    Convert a merged dict-of-dicts into an AnnData object (samples × features).

    Materializes any dask DataFrames lazily, aligns feature order across samples, and
    stacks into a single X matrix without concat-in-loop. Saves with .write_h5ad() for
    backed/lazy access and interoperability with scanpy / pydeseq2.

    Args:
        merged_dict:   output of merge_2dicts or merge_3dicts
        data_cat:      omic key inside each sample dict ('rna', 'meth', 'gene', 'mirna')
        feature_col:   column to use as var index ('gene_id', 'miRNA_ID', 'site')
        count_col:     column to use as X values ('unstranded', 'read_count', 'betav')
        extra_layers:  dict of {layer_name: column_name} for additional matrices,
                       e.g. {'tpm': 'tpm_unstranded'}

    Returns:
        AnnData object: X shape (n_samples × n_features), obs=clinical, var=feature metadata
    """
    keys = list(merged_dict.keys())

    first_df = merged_dict[keys[0]][data_cat]
    if hasattr(first_df, 'compute'):
        first_df = first_df.compute()
    features = first_df[feature_col].values

    X_rows = []
    layer_rows = {ln: [] for ln in (extra_layers or {})}
    obs_rows = []

    for key in keys:
        df = merged_dict[key][data_cat]
        if hasattr(df, 'compute'):
            df = df.compute()
        df = df.set_index(feature_col).reindex(features)
        X_rows.append(df[count_col].values.astype(np.float32))
        for layer_name, col in (extra_layers or {}).items():
            layer_rows[layer_name].append(df[col].values.astype(np.float32))

        cd = merged_dict[key]['cd']
        if hasattr(cd, 'compute'):
            cd = cd.compute()
        obs_rows.append(cd.iloc[0])

    X = np.vstack(X_rows)
    obs = pd.DataFrame(obs_rows, index=keys)
    var = pd.DataFrame(index=features)
    var.index.name = feature_col

    adata = ad.AnnData(X=X, obs=obs, var=var)
    for layer_name, rows in layer_rows.items():
        adata.layers[layer_name] = np.vstack(rows)

    return adata
