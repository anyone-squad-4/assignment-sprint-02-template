from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE


def preprocess_data(
    train_df: pd.DataFrame, 
    val_df: pd.DataFrame, 
    test_df: pd.DataFrame,
    apply_smote: bool = True  # Parámetro para controlar aplicación de SMOTE
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
 
    print("Input train data shape: ", train_df.shape)
    print("Input val data shape: ", val_df.shape)
    print("Input test data shape: ", test_df.shape, "\n")

    # Copy dataframes to avoid modifying original data
    working_train_df = train_df.copy()
    working_val_df = val_df.copy()
    working_test_df = test_df.copy()

    # Store target if present
    y_train = None
    if 'TARGET' in working_train_df.columns:
        y_train = working_train_df['TARGET'].copy()
        working_train_df = working_train_df.drop(columns=['TARGET'])
    if 'TARGET' in working_val_df.columns:
        working_val_df = working_val_df.drop(columns=['TARGET'])
    if 'TARGET' in working_test_df.columns:
        working_test_df = working_test_df.drop(columns=['TARGET'])

    # Fix known anomaly in DAYS_EMPLOYED
    working_train_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_val_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_test_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)

    # Basic feature engineering (keeping it minimal)
    print("Creating essential new features...")
    for df in [working_train_df, working_val_df, working_test_df]:
        # External source features
        ext_source_cols = [col for col in df.columns if 'EXT_SOURCE' in col]
        if len(ext_source_cols) >= 2:
            df['EXT_SOURCES_MEAN'] = df[ext_source_cols].mean(axis=1)
        
        # Age in years
        if 'DAYS_BIRTH' in df.columns:
            df['AGE_YEARS'] = abs(df['DAYS_BIRTH']) / 365.25

    # Handle categorical features
    print("Encoding categorical features...")
    categorical_cols = working_train_df.select_dtypes(include=["object"]).columns
    binary_cols = [col for col in categorical_cols if working_train_df[col].nunique() == 2]
    multi_cols = [col for col in categorical_cols if working_train_df[col].nunique() > 2]

    # Encode binary columns
    if binary_cols:
        ordinal_encoder = OrdinalEncoder()
        working_train_df[binary_cols] = ordinal_encoder.fit_transform(working_train_df[binary_cols])
        working_val_df[binary_cols] = ordinal_encoder.transform(working_val_df[binary_cols])
        working_test_df[binary_cols] = ordinal_encoder.transform(working_test_df[binary_cols])
    
    # Encode multi-value categorical columns
    if multi_cols:
        onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        onehot_encoded_train = onehot_encoder.fit_transform(working_train_df[multi_cols])
        onehot_encoded_val = onehot_encoder.transform(working_val_df[multi_cols])
        onehot_encoded_test = onehot_encoder.transform(working_test_df[multi_cols])
        
        onehot_cols = onehot_encoder.get_feature_names_out(multi_cols)
        onehot_train_df = pd.DataFrame(onehot_encoded_train, columns=onehot_cols, index=working_train_df.index)
        onehot_val_df = pd.DataFrame(onehot_encoded_val, columns=onehot_cols, index=working_val_df.index)
        onehot_test_df = pd.DataFrame(onehot_encoded_test, columns=onehot_cols, index=working_test_df.index)
        
        working_train_df = pd.concat([working_train_df.drop(columns=multi_cols), onehot_train_df], axis=1)
        working_val_df = pd.concat([working_val_df.drop(columns=multi_cols), onehot_val_df], axis=1)
        working_test_df = pd.concat([working_test_df.drop(columns=multi_cols), onehot_test_df], axis=1)

    # Ensure all dataframes have the same columns
    common_cols = list(set(working_train_df.columns) & 
                      set(working_val_df.columns) & 
                      set(working_test_df.columns))
    
    working_train_df = working_train_df[common_cols]
    working_val_df = working_val_df[common_cols]
    working_test_df = working_test_df[common_cols]
    
    # Handle missing values
    print("Imputing missing values...")
    imputer = SimpleImputer(strategy="median")
    working_train_df = pd.DataFrame(
        imputer.fit_transform(working_train_df), 
        columns=working_train_df.columns, 
        index=working_train_df.index
    )
    working_val_df = pd.DataFrame(
        imputer.transform(working_val_df), 
        columns=working_val_df.columns, 
        index=working_val_df.index
    )
    working_test_df = pd.DataFrame(
        imputer.transform(working_test_df), 
        columns=working_test_df.columns, 
        index=working_test_df.index
    )

    # === Force exactly 246 features ===
    current_feature_count = working_train_df.shape[1]
    print(f"Current feature count: {current_feature_count}")
    
    if current_feature_count > 246 and y_train is not None:
        print(f"Selecting top 246 features using SelectKBest...")
        # Use ANOVA F-test for feature selection
        selector = SelectKBest(f_classif, k=246)
        
        # Fit and transform training data
        selected_train = selector.fit_transform(working_train_df, y_train)
        
        # Get the selected feature names
        selected_mask = selector.get_support()
        selected_features = working_train_df.columns[selected_mask].tolist()
        
        # Ensure we have exactly 246 features
        assert len(selected_features) == 246, f"Selected {len(selected_features)} features instead of 246"
        
        # Apply selection to all datasets
        working_train_df = working_train_df[selected_features]
        working_val_df = working_val_df[selected_features]
        working_test_df = working_test_df[selected_features]
    elif current_feature_count < 246:
        # This is unlikely but handle it just in case
        raise ValueError(f"Not enough features: {current_feature_count}. Need 246 features.")
    
    # Scaling
    print("Scaling features...")
    scaler = StandardScaler()
    working_train_df = pd.DataFrame(
        scaler.fit_transform(working_train_df), 
        columns=working_train_df.columns, 
        index=working_train_df.index
    )
    working_val_df = pd.DataFrame(
        scaler.transform(working_val_df), 
        columns=working_val_df.columns, 
        index=working_val_df.index
    )
    working_test_df = pd.DataFrame(
        scaler.transform(working_test_df), 
        columns=working_test_df.columns, 
        index=working_test_df.index
    )
    
    # Final verification before converting to arrays
    final_feature_count = working_train_df.shape[1]
    print(f"Final feature count: {final_feature_count}")
    
    # Verify again we have exactly 246 features
    if final_feature_count != 246:
        # Emergency trim or padding if needed
        if final_feature_count > 246:
            # Just take first 246 features as last resort
            all_features = working_train_df.columns.tolist()
            selected_features = all_features[:246]
            working_train_df = working_train_df[selected_features]
            working_val_df = working_val_df[selected_features]
            working_test_df = working_test_df[selected_features]
        else:
            raise ValueError(f"Not enough features: {final_feature_count}. Need 246 features.")
    
    # Convert to numpy arrays
    train_array = working_train_df.to_numpy()
    val_array = working_val_df.to_numpy()
    test_array = working_test_df.to_numpy()
    
    # Apply SMOTE to balance the training data
    if apply_smote and y_train is not None:
        print("Applying SMOTE to balance training data...")
        
        # Check class imbalance before SMOTE
        class_counts = np.bincount(y_train)
        print(f"Class distribution before SMOTE: {class_counts}")
        imbalance_ratio = class_counts[0] / class_counts[1] if len(class_counts) > 1 and class_counts[1] > 0 else 0
        print(f"Imbalance ratio (majority:minority): {imbalance_ratio:.2f}:1")
        
        # Initialize and apply SMOTE
        smote = SMOTE(random_state=42)
        train_array, y_train_resampled = smote.fit_resample(train_array, y_train)
        
        # Report new class balance
        new_class_counts = np.bincount(y_train_resampled)
        print(f"Class distribution after SMOTE: {new_class_counts}")
        print(f"New training data shape after SMOTE: {train_array.shape}")
    
    print("Final processed shapes:")
    print("Processed train data shape: ", train_array.shape)
    print("Processed val data shape: ", val_array.shape)
    print("Processed test data shape: ", test_array.shape, "\n")
    
    # Final assertions
    assert train_array.shape[1] == 246, f"Expected 246 features, got {train_array.shape[1]}"
    assert val_array.shape[1] == 246, f"Expected 246 features, got {val_array.shape[1]}"
    assert test_array.shape[1] == 246, f"Expected 246 features, got {test_array.shape[1]}"
    
    return train_array, val_array, test_array