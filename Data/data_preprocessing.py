
import pandas as pd
import numpy as np
import os
import zipfile
from datetime import datetime

# Create Data directory if it doesn't exist
if not os.path.exists('Data'):
    os.makedirs('Data')

def process_maf_file(maf_file):
    """
    Process the MAF file to extract metabolite data
    """
    print(f"Processing MAF file: {maf_file}")
    
    # Read the MAF file
    maf_data = pd.read_csv(maf_file, sep='\t')
    
    # Extract metabolite names and sample IDs
    metabolite_names = maf_data['metabolite_identification'].tolist()
    
    # Get sample columns (those starting with '16K-' or 'N-')
    sample_columns = [col for col in maf_data.columns if col.startswith('16K-') or col.startswith('N-')]
    
    # Create a dictionary to store metabolite data for each sample
    metabolite_data = {}
    
    # Extract metabolite values for each sample
    for sample_id in sample_columns:
        metabolite_values = maf_data[sample_id].tolist()
        metabolite_data[sample_id] = metabolite_values
    
    # Create a dataframe with metabolite data
    metabolite_df = pd.DataFrame(metabolite_data)
    
    # Add metabolite names as index
    metabolite_df.index = metabolite_names
    
    # Transpose the dataframe to have samples as rows and metabolites as columns
    metabolite_df = metabolite_df.transpose()
    
    # Reset index to have sample IDs as a column
    metabolite_df.reset_index(inplace=True)
    metabolite_df.rename(columns={'index': 'Source Name'}, inplace=True)
    
    return metabolite_df

def process_excel_file(excel_file):
    """
    Process the Excel file to extract clinical data
    """
    print(f"Processing Excel file: {excel_file}")
    
    # Read the Excel file
    clinical_data = pd.read_excel(excel_file)
    
    # Extract relevant columns
    clinical_data_subset = clinical_data[['Sample', 'Grade', 'Status (1 = dead, 0 = censored)', 
                                         'Survival (days)', 'ER status', 'HER2 score', 
                                         'Age (yrs)', 'Sex', 'Neuter status', 'Breed', 
                                         'Weight (kg)', 'Histology']]
    
    # Rename columns for consistency
    clinical_data_subset = clinical_data_subset.rename(columns={
        'Sample': 'Source Name',
        'Status (1 = dead, 0 = censored)': 'Status',
        'Survival (days)': 'Survival',
        'Age (yrs)': 'Age',
        'ER status': 'ER Status',
        'HER2 score': 'HER2 Status',
        'Weight (kg)': 'Weight (Kg)'
    })
    
    # Convert age from years to months
    clinical_data_subset['Age[Month]'] = clinical_data_subset['Age'] * 12
    
    return clinical_data_subset

def create_mvb_file(metabolite_df, clinical_data):
    """
    Create the MVB (Metabolite vs Benign/Malignant) file
    """
    print("Creating MVB file")
    
    # Merge metabolite data with clinical data
    mvb_data = metabolite_df.copy()
    
    # Add tumor state column (1 for malignant, 0 for benign)
    def get_tumor_state(sample_id):
        if sample_id in clinical_data['Source Name'].values:
            histology = clinical_data[clinical_data['Source Name'] == sample_id]['Histology'].values[0]
            if isinstance(histology, str) and 'Benign' in histology:
                return 0
            else:
                return 1
        return 1  # Default to malignant if not found
    
    mvb_data['Tumor State'] = mvb_data['Source Name'].apply(get_tumor_state)
    
    # Reorder columns to have Source Name and Tumor State first
    cols = mvb_data.columns.tolist()
    cols.remove('Source Name')
    cols.remove('Tumor State')
    mvb_data = mvb_data[['Source Name', 'Tumor State'] + cols]
    
    # Save to CSV
    output_file = 'Data/mvb_unadjmetaboliteprofiling.csv'
    mvb_data.to_csv(output_file, index=False)
    print(f"MVB file saved to {output_file}")
    
    return mvb_data

def create_mg_file(metabolite_df, clinical_data):
    """
    Create the MG (Metabolite vs Grade) file
    """
    print("Creating MG file")
    
    # Merge metabolite data with clinical data
    mg_data = metabolite_df.copy()
    
    # Create a mapping of sample names to grades
    grade_mapping = {}
    for _, row in clinical_data.iterrows():
        if pd.notna(row['Grade']):
            try:
                grade = int(row['Grade'])
                grade_mapping[row['Source Name']] = grade
            except:
                # Handle non-numeric grades (e.g., "Benign")
                if row['Grade'] == 'Benign':
                    grade_mapping[row['Source Name']] = 0
    
    # Add tumor grade column
    mg_data['Tumor Grade'] = mg_data['Source Name'].map(grade_mapping)
    
    # Remove rows with missing grades
    mg_data = mg_data.dropna(subset=['Tumor Grade'])
    
    # Reorder columns to have Source Name and Tumor Grade first
    cols = mg_data.columns.tolist()
    cols.remove('Source Name')
    cols.remove('Tumor Grade')
    mg_data = mg_data[['Source Name', 'Tumor Grade'] + cols]
    
    # Rename Source Name to SourceName for consistency with existing file
    mg_data = mg_data.rename(columns={'Source Name': 'SourceName'})
    
    # Save to CSV
    output_file = 'Data/mg_unadjmetaboliteprofiling.csv'
    mg_data.to_csv(output_file, index=False)
    print(f"MG file saved to {output_file}")
    
    return mg_data

def create_hvt_file(metabolite_df, clinical_data):
    """
    Create the HVT (High vs Low) file based on HER2 status
    """
    print("Creating HVT file")
    
    # Merge metabolite data with clinical data
    hvt_data = metabolite_df.copy()
    
    # Create a mapping of sample names to HER2 status
    her2_mapping = {}
    for _, row in clinical_data.iterrows():
        if pd.notna(row['HER2 Status']) and row['HER2 Status'] != 'Not applicable':
            try:
                her2_status = int(row['HER2 Status'])
                # Classify as high (1) if HER2 score >= 2, otherwise low (0)
                her2_mapping[row['Source Name']] = 1 if her2_status >= 2 else 0
            except:
                pass
    
    # Add tumor state column based on HER2 status
    hvt_data['Tumor State'] = hvt_data['Source Name'].map(her2_mapping)
    
    # Remove rows with missing HER2 status
    hvt_data = hvt_data.dropna(subset=['Tumor State'])
    
    # Reorder columns to have Source Name and Tumor State first
    cols = hvt_data.columns.tolist()
    cols.remove('Source Name')
    cols.remove('Tumor State')
    hvt_data = hvt_data[['Source Name', 'Tumor State'] + cols]
    
    # Save to CSV
    output_file = 'Data/hvt_unadjmetaboliteprofiling.csv'
    hvt_data.to_csv(output_file, index=False)
    print(f"HVT file saved to {output_file}")
    
    return hvt_data

def create_cox_regression_file(clinical_data):
    """
    Create the Cox regression file with age, MG, and Pena grade
    """
    print("Creating Cox regression file")
    
    # Extract relevant columns for Cox regression
    cox_data = clinical_data[['Source Name', 'Grade', 'Survival', 'Status', 'Age']].copy()
    
    # Add MG (Metabolic Grade) column - this would typically come from a model
    # For now, we'll use a placeholder based on existing data
    cox_data['MG'] = cox_data['Grade']
    
    # Add Pena Grade column - this would typically come from a model
    # For now, we'll use a placeholder based on existing data
    cox_data['Pena Grade'] = cox_data['Grade']
    
    # Clean up sample names (remove trailing spaces)
    cox_data['Sample'] = cox_data['Source Name'].str.strip()
    
    # Select and reorder columns
    cox_data = cox_data[['Sample', 'Grade', 'Survival', 'Status', 'Age', 'MG', 'Pena Grade']]
    
    # Save to CSV
    output_file = 'Data/coxregression_agemgpena.csv'
    cox_data.to_csv(output_file, index=False)
    print(f"Cox regression file saved to {output_file}")
    
    # Save to CSV with UTF-8 encoding (for compatibility)
    output_file_utf8 = 'Data/coxregression_agemgpena_utf8.csv'
    cox_data.to_csv(output_file_utf8, index=False, encoding='utf-8')
    print(f"Cox regression file (UTF-8) saved to {output_file_utf8}")
    
    return cox_data

def create_dist_age_os_weight_file(clinical_data):
    """
    Create the distribution of age, overall survival, and weight file
    """
    print("Creating distribution file")
    
    # Extract relevant columns
    dist_data = clinical_data[['Source Name', 'Survival', 'Age', 'Weight (Kg)']].copy()
    
    # Rename Source Name to Sample for consistency
    dist_data = dist_data.rename(columns={'Source Name': 'Sample'})
    
    # Save to CSV
    output_file = 'Data/dist_ageosweight.csv'
    dist_data.to_csv(output_file, index=False)
    print(f"Distribution file saved to {output_file}")
    
    return dist_data

def create_heatmap_file(metabolite_df, clinical_data):
    """
    Create the heatmap file with metabolite data and clinical information
    """
    print("Creating heatmap file")
    
    # Start with metabolite data
    heatmap_data = metabolite_df.copy()
    
    # Add clinical information
    clinical_subset = clinical_data[['Source Name', 'Age[Month]', 'ER Status', 'HER2 Status', 'Weight (Kg)', 'Survival']].copy()
    clinical_subset = clinical_subset.dropna(subset=['Age[Month]', 'ER Status', 'HER2 Status'])
    
    # Merge with metabolite data
    heatmap_data = pd.merge(heatmap_data, clinical_subset, on='Source Name', how='inner')
    
    # Add tumor state column (1 for malignant, 0 for benign)
    def get_tumor_state(sample_id):
        if sample_id in clinical_data['Source Name'].values:
            histology = clinical_data[clinical_data['Source Name'] == sample_id]['Histology'].values[0]
            if isinstance(histology, str) and 'Benign' in histology:
                return 0
            else:
                return 1
        return 1  # Default to malignant if not found
    
    heatmap_data['Tumor State'] = heatmap_data['Source Name'].apply(get_tumor_state)
    
    # Add placeholder columns for Pena Grade (PG) and Metabolic Grade (MG)
    heatmap_data['Pena Grade (PG)'] = heatmap_data['Source Name'].map(
        {row['Source Name']: row['Grade'] for _, row in clinical_data.iterrows() if pd.notna(row['Grade'])}
    )
    heatmap_data['Metabolic Grade (MG)'] = heatmap_data['Pena Grade (PG)']  # Placeholder
    
    # Add OS (Days) column
    heatmap_data['OS (Days)'] = heatmap_data['Survival']
    
    # Reorder columns to have Source Name, Tumor State, and clinical data first
    metabolite_cols = [col for col in heatmap_data.columns if col not in 
                      ['Source Name', 'Tumor State', 'Age[Month]', 'ER Status', 'HER2 Status', 
                       'Weight (Kg)', 'Survival', 'Pena Grade (PG)', 'Metabolic Grade (MG)', 'OS (Days)']]
    
    heatmap_data = heatmap_data[['Source Name', 'Tumor State', 'Age[Month]', 'ER Status', 'HER2 Status'] + 
                               metabolite_cols + ['OS (Days)', 'Weight (Kg)', 'Pena Grade (PG)', 'Metabolic Grade (MG)']]
    
    # Save to CSV
    output_file = 'Data/heatmap_main.csv'
    heatmap_data.to_csv(output_file, index=False)
    print(f"Heatmap file saved to {output_file}")
    
    return heatmap_data

def create_readme():
    """
    Create a README file with information about the preprocessing
    """
    print("Creating README file")
    
    readme_content = f"""# Data Preprocessing

This directory contains processed data files derived from the original MAF and Excel files.

## Files

1. **mvb_unadjmetaboliteprofiling.csv**: Metabolite data with tumor state (benign/malignant)
2. **mg_unadjmetaboliteprofiling.csv**: Metabolite data with tumor grade
3. **hvt_unadjmetaboliteprofiling.csv**: Metabolite data with HER2 status (high/low)
4. **coxregression_agemgpena.csv**: Data for Cox regression analysis with age, metabolic grade, and Pena grade
5. **dist_ageosweight.csv**: Distribution data for age, overall survival, and weight
6. **heatmap_main.csv**: Combined metabolite and clinical data for heatmap visualization

## Source Files

- MAF file: m_MTBLS2550_NMR_metabolite_profiling_v2_maf(1).tsv
- Excel file: 41467_2020_17458_MOESM4_ESM.xlsx

## Processing Date

{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    # Save README file
    with open('Data/README.md', 'w') as f:
        f.write(readme_content)
    
    print("README file created")

def create_zip_file():
    """
    Create a ZIP file with all the data files
    """
    print("Creating ZIP file")
    
    # Create a ZIP file
    with zipfile.ZipFile('Data/processed_data.zip', 'w') as zipf:
        # Add all files in the Data directory to the ZIP file
        for file in os.listdir('Data'):
            if file != 'processed_data.zip':  # Avoid adding the ZIP file to itself
                zipf.write(os.path.join('Data', file), file)
    
    print("ZIP file created")

def main():
    """
    Main function to run the preprocessing pipeline
    """
    print("Starting data preprocessing")
    
    # Define input files
    maf_file = 'm_MTBLS2550_NMR_metabolite_profiling_v2_maf(1).tsv'
    excel_file = '41467_2020_17458_MOESM4_ESM.xlsx'
    
    # Process input files
    metabolite_df = process_maf_file(maf_file)
    clinical_data = process_excel_file(excel_file)
    
    # Create output files
    create_mvb_file(metabolite_df, clinical_data)
    create_mg_file(metabolite_df, clinical_data)
    create_hvt_file(metabolite_df, clinical_data)
    create_cox_regression_file(clinical_data)
    create_dist_age_os_weight_file(clinical_data)
    create_heatmap_file(metabolite_df, clinical_data)
    
    # Create README file
    create_readme()
    
    # Create ZIP file
    create_zip_file()
    
    print("Data preprocessing completed")

if __name__ == "__main__":
    main()
