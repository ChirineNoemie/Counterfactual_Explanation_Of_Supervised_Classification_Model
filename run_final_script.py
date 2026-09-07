# -*- coding: utf-8 -*-
import argparse
from itertools import product

import numpy as np
import time

import pandas as pd
import torch
from joblib import Parallel, delayed

from experiment.exp1 import CounterfactualExperiment1

# Dataset classification based on number of classes
BINARY_DATASETS = ['breast_cancer', 'moons', 'boston', 'magic', 'banknote']
MULTICLASS_DATASETS = ['iris', 'wine']


def _run_exp_by_idx(idx, experiments):
    return experiments[idx].run()

def cached_run_all(datasets=None,
                   method_list=None, num_iter_list=None, n_counterfactuals_list=None,
                   n_clusters_list=None, strategy_list=None, beta_list=None, lambda_list=None,
                   clustering_method_list=None, cv_grid_size=20, max_samples=3000, n_jobs=1,n_folds=2):
    if method_list is None:
        method_list = ['wachter', 'Medoid-based', 'dice', 'lvq']
    if num_iter_list is None:
        num_iter_list = [100]
    if n_counterfactuals_list is None:
        n_counterfactuals_list = [1]
    if n_clusters_list is None:
        n_clusters_list = [-1]
    if strategy_list is None:
        strategy_list = ['greedy']
    if beta_list is None:
        beta_list = [10.]
    if lambda_list is None:
        lambda_list = [10.]
    if clustering_method_list is None:
        clustering_method_list = ['kmedoids']
    if datasets is None:
        datasets = ['iris', 'breast_cancer']

    all_result = []
    experiments = []

    for dataset in datasets:
        for method, num_iter, n_counterfactuals, n_clusters, clustering_method, strategy, beta in list(product(
                method_list, num_iter_list, n_counterfactuals_list, n_clusters_list, clustering_method_list,
                strategy_list, beta_list)):
            experiment = CounterfactualExperiment1(
                dataset_name=dataset,
                n_clusters=n_clusters if method in ['Medoid-based', 'lvq'] else -1,
                cv_grid_size=cv_grid_size,
                num_iter=num_iter,
                max_samples=max_samples,
                method=method,
                n_counterfactuals=n_counterfactuals,
                clustering_method=clustering_method if method in ['Medoid-based', 'lvq'] and n_clusters > 0 else None,
                strategy=strategy,
                beta=beta,
                compute_diversity=(n_counterfactuals > 1),
                n_folds=n_folds 
            )
            experiments.append(experiment)

    # use joblib to parallelize the experiments
    results = Parallel(n_jobs=n_jobs, verbose=1)(delayed(_run_exp_by_idx)(idx, experiments) for idx in range(len(experiments)))

    return pd.concat(results)

if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)

    # create parser
    parser = argparse.ArgumentParser(description='Run counterfactual experiments')

    parser.add_argument('--datasets', nargs='+', type=str, default= ['iris','wine'],
                        #['breast_cancer', 'moons', 'boston', 'magic', 'banknote'],
                        help='Dataset to use')
    parser.add_argument('--dataset_type', type=str, choices=['binary', 'multiclass', 'all'], default='all',
                       help='Type of datasets to run: binary, multiclass or all')
    parser.add_argument('--methods', nargs='+', type=str, default=['wachter', 'dice', 'Medoid-based', 'lvq'], help='Method to use')
    parser.add_argument('--num_iters', nargs='+', type=int, default=[100], help='Number of iterations')
    parser.add_argument('--n_clusters', nargs='+', type=int, default=[32],
                        help='Number of clusters, ignored by all expect Medoid-based')
    parser.add_argument('--cv_grid_size', type=int, default=20, help='CV grid size')
    parser.add_argument('--max_samples', type=int, default=3000, help='Max samples for the train+test set')
    parser.add_argument('--n_counterfactuals', nargs='+', type=int, default=[1],
                        help='Number of counterfactuals to generate')
    parser.add_argument('--clustering_method', nargs='+', type=str, default=['kmedoids'],
                        help='Clustering method to use')
    parser.add_argument('--n_jobs', type=int, default=1, help='Number of parallel jobs')
    parser.add_argument('--n_folds', type=int, default=2,
                        help='Number of cross-validation folds')
    args = parser.parse_args()

    # Determine which datasets to use based on the type
    if args.dataset_type == 'binary':
        selected_datasets = [d for d in args.datasets if d in BINARY_DATASETS]
        print(f"🎯 RUNNING BINARY DATASETS: {selected_datasets}")
    elif args.dataset_type == 'multiclass':
        selected_datasets = [d for d in args.datasets if d in MULTICLASS_DATASETS]
        print(f"🎯 RUNNING MULTICLASS DATASETS: {selected_datasets}")
    else:  # all
        selected_datasets = args.datasets
        print(f"🎯 RUNNING ALL DATASETS: {selected_datasets}")

    if not selected_datasets:
        print(f"❌ No dataset selected for type '{args.dataset_type}'")
        exit(1)

    datasets = selected_datasets
    cv_grid_size = args.cv_grid_size
    method_list = args.methods
    num_iter_list = args.num_iters
    n_counterfactuals_list = args.n_counterfactuals
    n_clusters_list = args.n_clusters
    clustering_method_list = args.clustering_method

    start = time.time()
    all_result = cached_run_all(datasets=datasets, method_list=method_list, num_iter_list=num_iter_list,
                                n_counterfactuals_list=n_counterfactuals_list, n_clusters_list=n_clusters_list,
                                clustering_method_list=clustering_method_list, cv_grid_size=cv_grid_size,
                                max_samples=args.max_samples, n_jobs=args.n_jobs,n_folds=args.n_folds)
    
    # Display detailed results
    print("\n" + "="*80)
    print("DETAILED RESULTS")
    print("="*80)
    print(all_result.describe())

    # Averages by method and dataset
    print("\n" + "="*80)
    print("AVERAGES BY METHOD AND DATASET")
    print("="*80)

    # Split results by type
    binary_datasets = [d for d in datasets if d in BINARY_DATASETS]
    multiclass_datasets = [d for d in datasets if d in MULTICLASS_DATASETS]

    if binary_datasets:
        print(f"\n📊 BINARY DATASETS ({len(binary_datasets)}): {', '.join(binary_datasets)}")
        print("-" * 60)

    if multiclass_datasets:
        print(f"\n📊 MULTICLASS DATASETS ({len(multiclass_datasets)}): {', '.join(multiclass_datasets)}")
        print("-" * 60)
    
    summary_stats = []
    for method in method_list:
        method_data = all_result[all_result['Method'].str.lower() == method.lower()]
        if len(method_data) > 0:
            print(f"\n{method.upper()}:")
            for dataset in datasets:
                dataset_data = method_data[method_data['Dataset'] == dataset]
                if len(dataset_data) > 0:
                    valid_data = dataset_data[~dataset_data['Invalid']]
                    
                    accuracy = dataset_data['Correct_Prediction'].mean() * 100
                    validity = (1 - dataset_data['Invalid'].mean()) * 100
                    time_mean = dataset_data['Computation_Time'].mean()
                    
                    print(f"  {dataset}:")
                    print(f"    Accuracy: {accuracy:.1f}% | Validity: {validity:.1f}%")
                    print(f"    Time: {time_mean:.4f}s")
                    def to_scalar(v):
                        import numpy as np
                        return np.mean(v) if isinstance(v, np.ndarray) else v

                    if len(valid_data) > 0:
                        dist_mean = valid_data['Distance'].apply(to_scalar).mean()
                        dist_l1_mean = valid_data['Distance_L1'].apply(to_scalar).mean()
                        density_mean = valid_data['Density'].apply(to_scalar).mean()
                        print(f"    Distance L2: {dist_mean:.4f} | L1: {dist_l1_mean:.4f}")
                        print(f"    Density: {density_mean:.4f}")
                        
                        # Diversity if applicable
                        # FIX: diversity_mean was computed/displayed here but
                        # never fed back into the summary_stats.append(...) dict
                        # below -> the 'Diversity' column never existed in
                        # summary_final_*.csv, regardless of the success rate.
                        diversity_mean = None
                        if 'Diversity' in valid_data.columns and valid_data['Diversity'].notna().any():
                            diversity_mean = valid_data['Diversity'].mean()
                            print(f"    Diversity: {diversity_mean:.4f}")

                        # Add to summary statistics
                        dataset_type = "Binary" if dataset in BINARY_DATASETS else "Multiclass"
                        summary_stats.append({
                            'Dataset': dataset,
                            'Type': dataset_type,
                            'Method': method,
                            'Accuracy': f"{accuracy:.1f}%",
                            'Validity': f"{validity:.1f}%",
                            'Distance_L2': f"{dist_mean:.4f}",
                            'Distance_L1': f"{dist_l1_mean:.4f}",
                            'Density': f"{density_mean:.4f}",
                            'Diversity': f"{diversity_mean:.4f}" if diversity_mean is not None else "N/A",
                            'Time': f"{time_mean:.4f}s"
                        })
    
    # Save results
    output_file = f"results_{'_'.join(datasets)}_{'_'.join(method_list)}_{args.dataset_type}.csv"
    all_result.to_csv(output_file, index=False)
    print(f"\n✅ Detailed results saved to: {output_file}")

    # Create and save the summary
    if summary_stats:
        summary_df = pd.DataFrame(summary_stats)
        print(f"\n📊 FINAL SUMMARY:")
        print("="*80)
        print(summary_df.to_string(index=False))

        # Save the summary
        # FIX: naming aligned with output_file (datasets+methods+type) instead
        # of relying solely on args.dataset_type. Without this, running iris then
        # wine then digits separately (with --dataset_type='all' default each time)
        # would overwrite the same summary_final_all.csv on every run.
        summary_file = f"summary_{'_'.join(datasets)}_{'_'.join(method_list)}_{args.dataset_type}.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"\n💾 Summary saved to: {summary_file}")

        # Display separate summaries if both types are present
        if binary_datasets and multiclass_datasets:
            print(f"\n📊 BINARY DATASETS SUMMARY ({len(binary_datasets)} datasets):")
            binary_summary = summary_df[summary_df['Type'] == 'Binary']
            print(binary_summary.to_string(index=False))

            print(f"\n📊 MULTICLASS DATASETS SUMMARY ({len(multiclass_datasets)} datasets):")
            multiclass_summary = summary_df[summary_df['Type'] == 'Multiclass']
            print(multiclass_summary.to_string(index=False))
    
    print(f'\nElapsed time: {time.time() - start}')