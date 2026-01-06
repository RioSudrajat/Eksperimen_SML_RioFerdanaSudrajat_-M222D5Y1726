import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE  # NEW IMPORT
import os

def load_data(path):
    # Using sep=';' as identified in EDA
    return pd.read_csv(path, sep=';')

def preprocess_and_save(df, output_dir):
    print("Starting Feature Engineering...")
    
    # 1. Feature Engineering
    # Binning 'age'
    age_bins = [0, 29, 39, 49, 59, 100]
    age_labels = ['<30', '30-39', '40-49', '50-59', '60+']
    df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)
    
    # Konsolidasi 'education'
    df['education'] = df['education'].replace(['basic.4y', 'basic.6y', 'basic.9y'], 'basic')
    
    # Transformasi 'pdays'
    df['pernah_kontak'] = df['pdays'].apply(lambda x: 'no' if x == 999 else 'yes')
    
    # Fitur Rasio
    total_contacts = df['previous'] + df['campaign']
    df['previous_ratio'] = df['previous'] / total_contacts
    df['previous_ratio'] = df['previous_ratio'].fillna(0)
    
    # Drop duration (Leakage)
    if 'duration' in df.columns:
        df = df.drop(columns=['duration'])
        print("Dropped 'duration' column.")
    
    # Handle Duplicates
    duplicates = df.duplicated().sum()
    df = df.drop_duplicates()
    print(f"Dropped {duplicates} duplicate rows.")
    
    # 2. Encoding Target
    le = LabelEncoder()
    df['y'] = le.fit_transform(df['y'])
    
    # 3. SPLIT DATA FIRST (Anti-Leakage)
    # Drop columns that were transformed or are not needed
    X = df.drop(['y', 'age', 'pdays'], axis=1)
    y = df['y']
    
    print(f"Total Features shape: {X.shape}")
    
    # Stratified Split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Raw Train shape: {X_train_raw.shape}")
    print(f"Raw Test shape: {X_test_raw.shape}")

    # 4. Define Transformers
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    
    print(f"Numerical cols: {len(numerical_cols)}")
    print(f"Categorical cols: {len(categorical_cols)}")

    num_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, numerical_cols),
            ('cat', cat_transformer, categorical_cols)
        ],
        remainder='passthrough'
    )
    
    # 5. Transform (Fit on Train, Transform Both)
    print("Applying ColumnTransformer...")
    X_train_processed = preprocessor.fit_transform(X_train_raw)
    X_test_processed = preprocessor.transform(X_test_raw)
    
    # Get feature names
    feature_names = preprocessor.get_feature_names_out()
    
    # Convert to DataFrame
    if hasattr(X_train_processed, 'toarray'):
        X_train_processed = X_train_processed.toarray()
        X_test_processed = X_test_processed.toarray()
        
    X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_processed, columns=feature_names)
    
    # 6. Apply SMOTE (Train Only)
    print("Applying SMOTE to Training Data...")
    print(f"Before SMOTE (Train): {pd.Series(y_train).value_counts().to_dict()}")
    
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_df, y_train)
    
    print(f"After SMOTE (Train): {pd.Series(y_train_resampled).value_counts().to_dict()}")
    
    # 7. Save Data
    os.makedirs(output_dir, exist_ok=True)
    
    # A. Train (SMOTE)
    train_export = pd.concat([pd.DataFrame(X_train_resampled, columns=feature_names), 
                              pd.Series(y_train_resampled, name='y')], axis=1)
    
    # B. Test (Original)
    test_export = pd.concat([X_test_df, 
                             pd.Series(y_test.values, name='y')], axis=1)
    
    # C. Full Processed (No SMOTE, combined)
    full_X = pd.concat([X_train_df, X_test_df], ignore_index=True)
    full_y = pd.concat([pd.Series(y_train.values), pd.Series(y_test.values)], ignore_index=True)
    full_export = pd.concat([full_X, full_y.rename('y')], axis=1)
    
    train_path = os.path.join(output_dir, 'train.csv')
    test_path = os.path.join(output_dir, 'test.csv')
    full_path = os.path.join(output_dir, 'data_processed.csv')
    
    train_export.to_csv(train_path, index=False)
    test_export.to_csv(test_path, index=False)
    full_export.to_csv(full_path, index=False)
    
    print(f"Data saved to {output_dir}")
    print(f"Train (SMOTE): {train_export.shape}")
    print(f"Test: {test_export.shape}")
    print(f"Full Processed (No SMOTE): {full_export.shape}")

def main():
    # Adjust path as needed for the GitHub Action environment
    # Assuming the script is run from the root of the repo
    dataset_path = '../dataset/bank-additional-full.csv' 
    output_dir = 'output'
    
    # If running locally in the specific folder or similar structure
    # Checking absolute path or relative path
    possible_paths = [
        '../dataset/bank-additional-full.csv',
        'dataset/bank-additional-full.csv',
        'bank-additional-full.csv',
        'd:/Submission_MA5/Eksperimen_SML_RioFerdanaSudrajat_ M222D5Y1726/dataset/bank-additional-full.csv'
    ]
    
    final_path = None
    for path in possible_paths:
        if os.path.exists(path):
            final_path = path
            break
            
    if not final_path:
        print("Error: Dataset not found in common paths.")
        return

    print(f"Loading data from {final_path}...")
    df = load_data(final_path)
    
    preprocess_and_save(df, output_dir)

if __name__ == '__main__':
    main()
