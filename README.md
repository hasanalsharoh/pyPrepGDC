# pyPrepGDC

Python package for preprocessing multi-omics data downloaded from the [GDC (Genomic Data Commons)](https://gdc.cancer.gov/) portal via the GDC-Client into analysis-ready structures.

## Overview

GDC-Client downloads each file into its own folder (named by File ID). pyPrepGDC reads those folders, attaches clinical metadata, merges samples across omic types, and produces either a dict-of-dicts or an [AnnData](https://anndata.readthedocs.io/) object ready for downstream tools such as [scanpy](https://scanpy.readthedocs.io/) and [pydeseq2](https://pydeseq2.readthedocs.io/).

Supported omic types: **RNA-seq**, **miRNA-seq**, **DNA methylation (EPIC/450K)**, **gene-level CNV**.

## Installation

```bash
pip install pyPrepGDC
```

**Requirements:** Python ≥ 3.11, pandas, numpy, dask, pyarrow, anndata, scipy.

## Quick start

### 1 — Download data with GDC-Client

Your download directory should look like:

```
data/
├── <File ID 1>/
│   └── <data file>.tsv
├── <File ID 2>/
│   └── <data file>.tsv
│   annotations.txt   ← flagged/duplicate samples are skipped automatically
└── ...
```

You also need the **Sample Sheet** and **Clinical** files exported from the GDC portal.

### 2 — Parse a single omic type

```python
import pandas as pd
from pyPrepGDC import parse_gdc

# Load the sample sheet (contains File ID → Case ID / Sample ID mapping)
sample_sheet = pd.read_csv('sample_sheet.tsv', sep='\t')

# Parse RNA-seq files
rna_tumor = parse_gdc(
    target_folders = sample_sheet[sample_sheet['Sample Type'] == 'Primary Tumor']['File ID'].tolist(),
    base_dir       = 'data/',
    classific      = 'Tumor',
    grouped_list   = sample_sheet,
    file_ext       = '.tsv',
    data_cat       = 'rna',
)
# rna_tumor = {'Tumor1': {'cd': <clinical df>, 'rna': <dask df>}, 'Tumor2': ...}
```

`data_cat` options:

| Value | File type |
|---|---|
| `'rna'` | RNA-seq (STAR counts) |
| `'mirna'` | miRNA-seq |
| `'meth'` | DNA methylation beta values |
| `'gene'` | Gene-level CNV (WGS) |

### 3 — Merge two omic types

```python
from pyPrepGDC import merge_2dicts

# alignment_df must have a column matching join_key (default 'Case ID')
# and one column per File ID type
alignment = sample_sheet[['Case ID', 'File ID', 'MiRFile ID']].drop_duplicates()

merged = merge_2dicts(
    dict1        = rna_tumor,
    dict2        = mirna_tumor,
    omic1_cat    = 'rna',
    omic2_cat    = 'mirna',
    alignment_df = alignment,
    classific    = 'Tumor',
    join_key     = 'Case ID',   # use 'Sample ID' to preserve tumor/normal pairs
)
# merged = {'Tumor1': {'cd': ..., 'rna': ..., 'mirna': ...}, ...}
```

### 4 — Merge three omic types

```python
from pyPrepGDC import merge_3dicts

alignment = sample_sheet[['Case ID', 'File ID', 'GFile ID', 'MFile ID']].drop_duplicates()

merged = merge_3dicts(
    rna_dict     = rna_tumor,
    cnv_dict     = cnv_tumor,
    meth_dict    = meth_tumor,
    alignment_df = alignment,
    classific    = 'Tumor',
)
# merged = {'Tumor1': {'cd': ..., 'rna': ..., 'gene': ..., 'meth': ...}, ...}
```

### 5 — Build a count/beta matrix

```python
from pyPrepGDC import create_count_table

count_matrix = create_count_table(
    data_dict1 = tumor_merged,
    data_dict2 = normal_merged,
    data_cat   = 'rna',
    cols       = 'unstranded',
    label      = 'gene_id',
)
# Returns a DataFrame: rows = genes, columns = [gene_id, Tumor1, Tumor2, ..., Normal1, ...]
```

### 6 — Build a clinical DataFrame

```python
from pyPrepGDC import build_clinical_df

clinical = build_clinical_df(merged)
# Returns a long-form DataFrame with all samples stacked and a 'key' column
```

### 7 — Export to AnnData (.h5ad)

```python
from pyPrepGDC import build_anndata
import anndata as ad

adata = build_anndata(
    merged_dict  = merged,
    data_cat     = 'rna',
    feature_col  = 'gene_id',
    count_col    = 'unstranded',
    extra_layers = {'tpm': 'tpm_unstranded'},   # optional additional matrices
)

# Save — HDF5 format, compressed, supports backed/lazy loading
adata.write_h5ad('merged_rna.h5ad', compression='gzip')

# Load full
adata = ad.read_h5ad('merged_rna.h5ad')

# Load in backed (lazy) mode — only reads requested slices into RAM
adata = ad.read_h5ad('merged_rna.h5ad', backed='r')
```

**AnnData layout:**

| Slot | Contents |
|---|---|
| `adata.X` | Count / beta matrix (samples × features), `float32` |
| `adata.obs` | Clinical metadata (one row per sample) |
| `adata.var` | Feature metadata (gene / CpG site index) |
| `adata.layers['tpm']` | TPM values (if `extra_layers` supplied) |

### Downstream use with pydeseq2

```python
import pandas as pd
from pydeseq2.dds import DeseqDataSet

counts_df = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
meta_df   = adata.obs[['condition']]
dds = DeseqDataSet(counts=counts_df, metadata=meta_df, design_factors='condition')
```

## API reference

| Function | Description |
|---|---|
| `parse_gdc(target_folders, base_dir, classific, grouped_list, file_ext, data_cat)` | Parse GDC download folders into a dict-of-dicts |
| `merge_2dicts(dict1, dict2, omic1_cat, omic2_cat, alignment_df, classific, join_key)` | Merge any two omic dicts using a pre-built alignment DataFrame |
| `merge_3dicts(rna_dict, cnv_dict, meth_dict, alignment_df, classific)` | Merge RNA, CNV, and methylation dicts |
| `create_count_table(data_dict1, data_dict2, data_cat, cols, label)` | Build a count/value matrix from two merged dicts |
| `build_clinical_df(merged_dict)` | Concatenate clinical DataFrames into a single long-form table |
| `build_anndata(merged_dict, data_cat, feature_col, count_col, extra_layers)` | Convert a merged dict to an AnnData object |

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
