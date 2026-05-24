import pandas as pd

def merge_3dicts(dict1, dict2, dict3, classific):
    """
    Function to merge 3 omics data dictionaries based on Case ID while avoiding column duplication.
    
    Args:
        dict1: Dictionary containing RNA-seq data
        dict2: Dictionary containing genomic data
        dict3: Dictionary containing methylation data
        classific: String prefix for the merged dictionary keys
    
    Returns:
        Dictionary containing merged data with unified clinical information
    """
    merged_dict = {}
    counter = 1
    
    # Concatenate clinical data from each dictionary
    cd_df_gtw_all = pd.concat([val['cd'] for val in dict2.values()])
    cd_df_meth_all = pd.concat([val['cd'] for val in dict3.values()])
    
    # Get unique clinical data entries, keeping essential columns
    cd_df_gtw_unique = (cd_df_gtw_all[['Case ID', 'GFile ID']]
                       .drop_duplicates(subset=['Case ID'])
                       .set_index('Case ID'))
    
    cd_df_meth_unique = (cd_df_meth_all[['Case ID', 'MFile ID']]
                        .drop_duplicates(subset=['Case ID'])
                        .set_index('Case ID'))
    
    # Merge genomic and methylation clinical data
    merged_gtw_meth = (cd_df_gtw_unique
                      .join(cd_df_meth_unique, how='inner')
                      .reset_index())
    
    for akey, val in dict1.items():
        adf = val['rna']
        
        # Get clinical data for RNA-seq, keeping all columns except duplicate IDs
        cd_df_tw = (val['cd']
                   .drop_duplicates(subset=['Case ID'])
                   .reset_index(drop=True))
        
        # Find matching entries in other dictionaries
        matching_akey_dict2 = next((bkey for bkey, val2 in dict2.items() 
                                  if val2['cd']['Case ID'].isin(cd_df_tw['Case ID']).any()), None)
        matching_akey_dict3 = next((ckey for ckey, val3 in dict3.items() 
                                  if val3['cd']['Case ID'].isin(cd_df_tw['Case ID']).any()), None)
        
        if matching_akey_dict2 and matching_akey_dict3:
            gend_df = dict2[matching_akey_dict2]['gene']
            meth_df = dict3[matching_akey_dict3]['meth']
            
            # Keep all columns from cd_df_tw except GFile ID and MFile ID
            base_columns = [col for col in cd_df_tw.columns 
                          if col not in ['GFile ID', 'MFile ID']]
            cd_df_tw_filtered = cd_df_tw[base_columns]
            
            # Merge with the pre-merged gtw and meth clinical data
            merged_cd_df = pd.merge(cd_df_tw_filtered, merged_gtw_meth, 
                                  on='Case ID', how='inner')
            
            # Store merged data in the new dictionary
            merged_dict[f'{classific}{counter}'] = {
                'cd': merged_cd_df,
                'rna': adf,
                'gene': gend_df,
                'meth': meth_df
            }
            counter += 1
            
    return merged_dict

def merge_2dicts(dict1, dict2, classific):
    """
    Function to merge RNA-seq and miRNA-seq data dictionaries based on Sample ID.
    
    Args:
        dict1: Dictionary containing RNA-seq data
        dict2: Dictionary containing miRNA-seq data
        classific: String prefix for the merged dictionary keys
    
    Returns:
        Dictionary containing merged data with unified clinical information
    """
    merged_dict = {}
    counter = 1
    
    # Concatenate clinical data from dict2
    cd_df_mirna_all = pd.concat([val['cd'] for val in dict2.values()])
    
    # Get unique clinical data entries based on Sample ID instead of Case ID to preserve control groups
    cd_df_mirna_unique = (cd_df_mirna_all[['Sample ID', 'MiRFile ID']]
                        .drop_duplicates(subset=['Sample ID'])
                        .set_index('Sample ID'))
    
    for akey, val in dict1.items():
        adf = val['rna']
        
        # Get clinical data for RNA-seq, using Sample ID to keep both tumor and normal
        cd_df_tw = (val['cd']
                   .drop_duplicates(subset=['Sample ID'])
                   .reset_index(drop=True))
        
        # Find matching entries in dict2 using Sample ID
        matching_akey_dict2 = next((bkey for bkey, val2 in dict2.items() 
                                  if val2['cd']['Sample ID'].isin(cd_df_tw['Sample ID']).any()), None)
        
        if matching_akey_dict2:
            mirna_df = dict2[matching_akey_dict2].get('mirna', dict2[matching_akey_dict2].get('rna'))
            
            # Keep all columns from cd_df_tw except MiRFile ID to avoid duplication
            base_columns = [col for col in cd_df_tw.columns if col != 'MiRFile ID']
            cd_df_tw_filtered = cd_df_tw[base_columns]
            
            # Merge with dict2's File ID info
            merged_cd_df = pd.merge(cd_df_tw_filtered, cd_df_mirna_unique, 
                                  on='Sample ID', how='inner')
            
            # Store merged data in the new dictionary
            merged_dict[f'{classific}{counter}'] = {
                'cd': merged_cd_df,
                'rna': adf,
                'mirna': mirna_df
            }
            counter += 1
            
    return merged_dict


def create_count_table(data_dict1, data_dict2, data_subdict1, data_subdict2, data_cat, cols, label):
    """
    This function takes the dictionary of parsed data and creates a count matrix for the samples.
    data_dict1 is the dictionary of the first set of parsed data, or cases.
    data_dict2 is the dictionary of the second set of parsed data, or controls.
    data_subdict1 is the string subdictionary key for the first set of parsed data.
    data_subdict2 is the string subdictionary key for the second set of parsed data.
    data_cat is the string category of the data in the parsed data dictionaries. It can be one of 'gene', 'meth', or 'rna'.
    cols is the string column name of the data in the parsed data dictionaries. It can be one of 'gene_id', 'site', or 'gene_name',
      depending on the data category.
    label is the string column name of the data labels, such as gene_id, site, or gene_name.
    """
    dict1df= pd.DataFrame()
    for i in range(1, len(data_dict1)+1):
        new_col= pd.DataFrame(zip(data_dict1[f'{data_subdict1}{i}'][f'{data_cat}'][f'{cols}']),columns=[f'{data_subdict1}{i}'])
        dict1df= pd.concat([dict1df,new_col],axis=1)
    dict2df= pd.DataFrame()
    for i in range(1, len(data_dict2)+1):
        new_col= pd.DataFrame(zip(data_dict2[f'{data_subdict2}{i}'][f'{data_cat}'][f'{cols}']),columns=[f'{data_subdict2}{i}'])
        dict2df= pd.concat([dict2df,new_col],axis=1)

    labeldict= pd.DataFrame(zip(data_dict1[f'{data_subdict1}1'][f'{data_cat}'][f'{label}']),columns=[f'{label}'])
    combdict= pd.concat([labeldict,dict1df,dict2df],axis=1)
    return combdict