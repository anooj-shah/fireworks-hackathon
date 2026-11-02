## SimBench: Double-blind compliant code

import os
import sys
import glob
import pandas as pd
import numpy as np
from pathlib import Path


def calc_total_variation(human_dist, model_dist):
    """Calculate Total Variation distance between two distributions."""
    if not isinstance(human_dist, (list, np.ndarray)) or not isinstance(model_dist, (list, np.ndarray)):
        return np.nan
    
    # Convert to numpy arrays
    p = np.array(human_dist, dtype=float)
    q = np.array(model_dist, dtype=float)
    
    # Check if arrays have same length
    if len(p) != len(q) or len(p) == 0:
        return np.nan
    
    # Normalize distributions (in case they're not already)
    p = p / p.sum() if p.sum() > 0 else p
    q = q / q.sum() if q.sum() > 0 else q
    
    # Calculate Total Variation: 0.5 * sum(|p_i - q_i|)
    return 0.5 * np.sum(np.abs(p - q))


def uniform_distribution(n):
    """Create uniform distribution with n options."""
    if n <= 0:
        return []
    return [1.0 / n] * n


def process_results_file(file_path):
    """Process a single results pickle file."""
    try:
        df = pd.read_pickle(file_path)
        print(f"Loaded {len(df)} results from {Path(file_path).name}")
        
        # Extract human distribution from human_answer if it's a dict
        if 'human_answer' in df.columns:
            df['Human_Distribution'] = df['human_answer'].apply(
                lambda x: list(x.values()) if isinstance(x, dict) else []
            )
        
        # Calculate Total Variation for each row
        df['Total_Variation'] = df.apply(
            lambda row: calc_total_variation(
                row['Human_Distribution'], 
                row['Response_Distribution']
            ), axis=1
        )
        
        # Calculate uniform baseline TV for each question (TV distance from uniform to human)
        df['TV_Uniform'] = df.apply(
            lambda row: calc_total_variation(
                row['Human_Distribution'],
                uniform_distribution(len(row['Human_Distribution']))
            ) if len(row['Human_Distribution']) > 1 else np.nan, axis=1
        )
        
        return df
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def main():
    if len(sys.argv) != 2:
        print("Usage: python calculate_simbench_score.py <results_folder>")
        print("\nExample: python calculate_simbench_score.py ./results/")
        sys.exit(1)
    
    results_folder = sys.argv[1]
    
    if not os.path.exists(results_folder):
        print(f"Error: Results folder '{results_folder}' does not exist.")
        sys.exit(1)
    
    # Find all pickle files in the results folder
    pickle_files = glob.glob(os.path.join(results_folder, "*.pkl"))
    
    if not pickle_files:
        print(f"No .pkl files found in {results_folder}")
        sys.exit(1)
    
    print(f"Found {len(pickle_files)} result files")
    print("=" * 60)
    
    all_results = []
    
    # Process each file
    for file_path in pickle_files:
        df = process_results_file(file_path)
        if df is not None:
            all_results.append(df)
    
    if not all_results:
        print("No valid results to process.")
        sys.exit(1)
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Check if we have dataset_name column (crucial for correct scoring)
    if 'dataset_name' not in combined_df.columns:
        print("ERROR: 'dataset_name' column not found in results!")
        print("Available columns:", list(combined_df.columns))
        print("Cannot calculate SimBench scores without dataset information.")
        sys.exit(1)
    
    # Calculate dataset-specific uniform baselines (this is the key step!)
    print("Calculating dataset-specific uniform baselines...")
    dataset_norms = combined_df.groupby('dataset_name')['TV_Uniform'].mean()
    print("Dataset norms:")
    for dataset, norm in dataset_norms.items():
        print(f"  {dataset}: {norm:.4f}")
    
    # Calculate rescaled SimBench scores using dataset-specific baselines
    combined_df['SimBench_Score'] = combined_df.apply(
        lambda row: 100 * (1 - (row['Total_Variation'] / dataset_norms[row['dataset_name']])) 
        if not pd.isna(row['Total_Variation']) and row['dataset_name'] in dataset_norms and dataset_norms[row['dataset_name']] > 0
        else np.nan, axis=1
    )
    
    # Calculate final SimBench scores by model
    print("\n" + "=" * 60)
    print("SIMBENCH SCORES (Higher is Better)")
    print("=" * 60)
    
    model_scores = combined_df.groupby('Model')['SimBench_Score'].agg(['mean', 'count']).round(2)
    model_scores.columns = ['SimBench_Score', 'Num_Questions']
    
    # Sort by score (descending - higher is better)
    model_scores = model_scores.sort_values('SimBench_Score', ascending=False)
    
    print(f"{'Model':<40} {'SimBench Score':<15} {'Questions':<10}")
    print("-" * 65)
    
    for model, row in model_scores.iterrows():
        score = row['SimBench_Score']
        count = int(row['Num_Questions'])
        print(f"{model:<40} {score:<15.2f} {count:<10}")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total questions processed: {len(combined_df)}")
    print(f"Models evaluated: {len(model_scores)}")
    print(f"Best performing model: {model_scores.index[0]} ({model_scores.iloc[0]['SimBench_Score']:.2f})")
    print(f"Average score across all models: {model_scores['SimBench_Score'].mean():.2f}")
    
    # Save detailed results
    output_file = os.path.join(results_folder, "simbench_scores.csv")
    model_scores.to_csv(output_file)
    print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    main() 