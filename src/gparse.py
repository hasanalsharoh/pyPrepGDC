# Parsing function for TCGA data
import os
import pandas as pd
import dask
import pyarrow
import dask.dataframe as dd
import warnings

def _omic_generator(target_folders, base_dir, classific, grouped_list, file_ext, data_cat):
    """
    Target folders is a list of GDC folders containing files of either RNAseq, DNA methylation, or CNV data.
    base_dir is the directory where the target folders are located.
    classific is the string class of the samples in the target folders. In this example we have TMAs (tumors with metastatic
    potential) and TWAs (tumors without metastatic potential).
    grouped_list is a DataFrame containing clinical data for the samples in the target folders. This is to implement clinical data.
    file_ext is the string file extension of the files in the target folders.
    data_cat is the string category of the data in the target folders. It can be one of 'gene', 'meth', or 'rna'.
    """
    counter = 1 # Counter for naming the DataFrames
    processed_files = set()  # Set to keep track of processed files, thiis avoids processing the same file multiple times in case of duplicates
    
    for folder_name in target_folders: 
        folder_path = os.path.join(base_dir, folder_name)
        if "annotations.txt" in os.listdir(folder_path): # Automatically skip folders with annotations.txt which are usually duplicates
            print(f"Skipping folder: {folder_name} (contains 'annotations.txt')")
            continue  # Skip this folder
        
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)

                # Check if the file has already been processed
                if file_path in processed_files:
                    continue  # Skip this file

                if os.path.isfile(file_path) and file_name.endswith(file_ext):
                    
                    # Create a DataFrame from the file
                    # For gene-level copy number variation (CNV) data
                    if data_cat == "gene": # Genomic data for WGS files
                        df = dd.read_csv(file_path, sep='\t')
                        df = df.dropna(subset=['gene_id'])
                        cdsubs = grouped_list[grouped_list['GFile ID'] == folder_name] # Get the corresponding clinical data

                    # DNA methylation beta-values
                    elif data_cat == "meth": # DNA methylation beta values data 
                        df = dd.read_csv(file_path, sep='\t', header=None)
                        df.columns = ['site', 'betav']
                        cdsubs = grouped_list[grouped_list['MFile ID'] == folder_name] # Get the corresponding clinical data
                    
                    # RNAseq
                    elif data_cat == "rna": # RNA-seq data
                        df = dd.read_csv(file_path, sep='\t', comment='#', 
                                         na_values=['N_unmapped', 'N_multimapping', 
                                                    'N_noFeature', 'N_ambiguous'],
                                         usecols=['gene_id','gene_name','gene_type',
                                                  'unstranded','tpm_unstranded']) # read only columns useful for analysis, for managing memory
                        df = df.dropna(subset=['gene_id'])
                        cdsubs = grouped_list[grouped_list['File ID'] == folder_name] # Get the corresponding clinical data
                    
                    # miRNA-seq
                    elif data_cat == "mirna": # miRNA-seq data
                        df = pd.read_csv(file_path, sep='\t', 
                                         usecols=['miRNA_ID','read_count']) # read only columns useful for analysis, for managing memory
                        df = df.dropna(subset=['miRNA_ID'])
                        cdsubs = grouped_list[grouped_list['File ID'] == folder_name] # Get the corresponding clinical data
                    
                    # Give the DataFrame a name based on the file name
                    dataframe_name = f"{classific}{counter}" # e.g. 'Tumor1', dictionary key corresponding to the parsed sample data
                    counter += 1 # Add 1 to the counter for the next DataFrame to have dictionary keys in order

                    # Add to parsed_data dictionary
                    yield dataframe_name, {'cd': cdsubs, data_cat: df}
                    
                    # Mark the file as processed
                    processed_files.add(file_path)

def parse_gdc(target_folders, base_dir, classific, grouped_list, file_ext, data_cat):
    """
    This function takes parameters from _omic_generator to parse the data
    Similarly, the arguments are duplicated to maintain the same functionality as _omic_generator

    output: dictionary [classific1: {'cd': clinical data, data_cat: omic data}, classific2: {'cd': clinical data, data_cat: omic data}, ...]
    """
    dict = {}
    generator = _omic_generator(target_folders, base_dir, classific, grouped_list, file_ext, data_cat)
    for name, data in generator:
        dict[f'{name}'] = data
    return dict