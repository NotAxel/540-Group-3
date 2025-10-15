import argparse
import os
import pandas as pd
import numpy as np

def target_encode(train_series, test_series, target_series, min_samples_leaf=1, smoothing=1):
    """
    Manually performs target encoding.
    Replaces a category with the smoothed average of the target variable for that category.
    """
    # Calculate statistics from the training data
    temp = pd.concat([train_series, target_series], axis=1)
    averages = temp.groupby(train_series.name)[target_series.name].agg(["mean", "count"])

    # Calculate the overall mean of the target
    smoothing_component = target_series.mean()

    # Apply smoothing
    averages["smoothed"] = (averages["mean"] * averages["count"] + smoothing_component * smoothing) / (averages["count"] + smoothing)

    # Create the mapping and apply it
    mapping = averages["smoothed"]

    # Apply the learned mapping to both train and test series
    train_encoded = train_series.map(mapping)
    test_encoded = test_series.map(mapping)

    # Fill any new/unseen categories in the test set with the global average
    test_encoded.fillna(smoothing_component, inplace=True)

    return train_encoded, test_encoded


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-input", type=str)
    parser.add_argument("--validation-input", type=str)
    args, _ = parser.parse_known_args()

    base_dir = "/opt/ml/processing"
    train_input_path = os.path.join(base_dir, "input/train", args.train_input)
    validation_input_path = os.path.join(base_dir, "input/validation", args.validation_input)

    print("Loading data...")
    df_train = pd.read_csv(train_input_path)
    df_val = pd.read_csv(validation_input_path)

    print("Applying manual Target Encoding...")
    categorical_cols = ['brand', 'category_code']
    target_column = 'purchased'

    for col in categorical_cols:
        # Learn from train, apply to both train and val
        train_encoded, val_encoded = target_encode(
            df_train[col], 
            df_val[col], 
            df_train[target_column]
        )
        # Add the new encoded columns
        df_train[f"{col}_encoded"] = train_encoded
        df_val[f"{col}_encoded"] = val_encoded

    print("Cleaning and reordering columns...")
    columns_to_drop = ['user_id', 'product_id', 'first_interaction', 'last_interaction'] + categorical_cols
    df_train_processed = df_train.drop(columns=columns_to_drop)
    df_val_processed = df_val.drop(columns=columns_to_drop)

    feature_columns = [col for col in df_train_processed.columns if col != target_column]
    final_train_df = df_train_processed[[target_column] + feature_columns]
    final_val_df = df_val_processed[[target_column] + feature_columns]

    print("Splitting validation data...")
    rand_split = np.random.rand(len(final_val_df))
    new_validation_set = final_val_df[rand_split < 0.5]
    batch_set = final_val_df[rand_split >= 0.5]

    print("Saving processed files...")
    train_output_path = os.path.join(base_dir, "output/train/train_commerce.csv")
    validation_output_path = os.path.join(base_dir, "output/validation/validation_commerce.csv")
    batch_output_path = os.path.join(base_dir, "output/batch/batch_commerce.csv")

    final_train_df.to_csv(train_output_path, header=False, index=False)
    new_validation_set.to_csv(validation_output_path, header=False, index=False)
    batch_set.to_csv(batch_output_path, header=False, index=False)