"""
pyPrepGDC: Python package for preprocessing --omics data obtained from GDC (Genomic Data Commons) Portal for downstream analysis.
This package is designed for preprocessing data downloaded through the GDC-Client python tool.
The main processing steps are:
1) After creating a metadata file, and downloading the data through the GDC-Client, 
 the data is created and organized in a folder structure as follows:
    - data/
        - sample1/
            - file1
        - sample2/
            - file2
            *- annotation file // in case errors are found in the data, they are often accompanied by annotation files that provide details about the errors. These files can be used to identify and address issues in the data.
2) Download from the GDC Portal, similarly to when obtaining the metadata file, Clinical.csv and Sample Sheet.csv must also be obtained.
3) The data is then preprocessed using the pyPrepGDC package.
"""
from .gparse import _omic_generator, parse_gdc
from .aggreg import create_count_table


__version__ = "0.0.1"
__all__= [
    "parse_gdc",
    "create_count_table"
]