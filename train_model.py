import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

def train():
    """
    Trains a Random Forest model by generating synthetic training data 
    based on the plant metadata found in Dataset.csv.
    """
    if not os.path.exists('Dataset.csv'):
        print("Error: Dataset.csv not found in backend folder.")
        return

    try:
        # Load the raw CSV
        df = pd.read_csv('Dataset.csv')
        
        # Clean column names to handle duplicates and hidden strings
        # Given: ['Plant', 'Plant', 'Image', 'Contaminants', 'Accumlation', ...]
        new_cols = []
        for i, col in enumerate(df.columns):
            clean_name = col.split(' ')[0].strip()
            # Handle potential duplicates by appending index
            if clean_name in new_cols:
                new_cols.append(f"{clean_name}_{i}")
            else:
                new_cols.append(clean_name)
        df.columns = new_cols
        
        print(f"Detected and cleaned columns: {list(df.columns)}")

        # Map our required data
        # We look for 'Plant' (or 'Plant_1') and 'Contaminants'
        plant_col = 'Plant' if 'Plant' in df.columns else 'Plant_0'
        contam_col = 'Contaminants'

        if plant_col not in df.columns or contam_col not in df.columns:
            print(f"Error: Required columns not found. Need '{plant_col}' and '{contam_col}'.")
            return

    except Exception as e:
        print(f"Failed to parse CSV: {e}")
        return

    # --- SYNTHETIC DATA GENERATION ---
    # Since the CSV is a lookup table, we generate numeric training data
    # so the Random Forest can learn 'if Lead is high, pick X'.
    
    data = []
    print("Generating synthetic training samples based on plant specialties...")
    
    # Extract unique plants and their targets
    plant_list = df[[plant_col, contam_col]].dropna().values.tolist()

    for _ in range(10000):
        # Generate random sensor readings
        cu = np.random.uniform(0, 300)
        cd = np.random.uniform(0, 15)
        pb = np.random.uniform(0, 800)
        
        # Determine which metal is the most "hazardous" relative to safe limits
        # Limits: Cu: 50, Cd: 1, Pb: 100
        ratios = {
            'Copper': cu / 50,
            'Cadmium': cd / 1,
            'Lead': pb / 100
        }
        main_contaminant = max(ratios, key=ratios.get)
        
        # Find plants from your CSV that target this specific metal
        eligible_plants = [p[0] for p in plant_list if main_contaminant.lower() in str(p[1]).lower()]
        
        if eligible_plants:
            chosen_plant = np.random.choice(eligible_plants)
            data.append([cu, cd, pb, chosen_plant])

    # Create the training DataFrame
    training_df = pd.DataFrame(data, columns=['Copper', 'Cadmium', 'Lead', 'Recommended_Plant'])

    # --- MODEL TRAINING ---
    X = training_df[['Copper', 'Cadmium', 'Lead']]
    y = training_df['Recommended_Plant']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Training Random Forest on {len(training_df)} samples...")
    model = RandomForestClassifier(n_estimators=500, max_depth=15, random_state=42)
    
    # Use the DataFrame to preserve feature names for the API
    model.fit(X_train, y_train)

    # Save the model
    joblib.dump(model, 'phytorem_rf_model.pkl')
    
    accuracy = model.score(X_test, y_test)
    print(f"Model saved successfully. Accuracy on synthetic test set: {accuracy * 100:.2f}%")
    print(f"Feature Names: {list(model.feature_names_in_)}")

if __name__ == "__main__":
    train()