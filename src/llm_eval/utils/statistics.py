"""
Statistics and plotting functions
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from statsmodels.stats.power import TTestIndPower
from ..evaluation.prompts import list_of_metrics

def calculate_statistics(values):
    """Calculate mean, standard error, and confidence intervals"""
    mean_value = np.mean(values) # Mean of the metric over single run and over single metric (but over all questions)
    std_error = np.std(values, ddof=1) / np.sqrt(len(values)) # ddof=1 to divide by n-1 to calculate the sample sd
    
    assert np.std(values, ddof=1) == np.sqrt(np.sum((values-mean_value)**2)/(len(values)-1)), "Standard deviation calculation mismatch"
    
    margin_of_error = 1.96 * std_error # didn't use t_critical=t.ppf(0.975, df=len(values)-1) since we're using sample standard deviation

    return {
        'mean': mean_value,
        'std_error': std_error,
        'ci_low': mean_value - margin_of_error,
        'ci_high': mean_value + margin_of_error
    }

def plot_metric_distributions(metric_values, axes, colors, bin_edges, metric_names):
    """Plot individual metric distributions with error bars"""
    error_bars = []
    run_stats = {}
    
    for metric_idx, (metric_name, values) in enumerate(metric_values.items()): # Loop over runs' metric names and values
        clean_metric_name = metric_name.replace('_descr', '') # This is over one run and over one metric (but over all questions)
        metric_name = metric_names[metric_idx]
        assert clean_metric_name == metric_name, "Metric name mismatch"
        
        stats = calculate_statistics(values)
        
        # Handle both single axis and array of axes
        if hasattr(axes, '__len__') and len(axes) > 1:
            current_ax = axes[metric_idx]
        else:
            current_ax = axes
            
        sns.histplot(values, bins=bin_edges, color=colors[metric_idx], ax=current_ax, kde=False)
        
        #Store error bars
        if metric_idx == 0:
            error_bars = []
        error_bars.append((stats['mean'], current_ax.get_ylim()[1]/2, stats['ci_high'] - stats['mean']))
        
        run_stats[metric_name] = stats

        current_ax.set_title(f'{metric_name} (Mean: {stats["mean"]:.2f} ± {stats["std_error"]:.2f} SE, CI: {stats["ci_low"]:.2f}-{stats["ci_high"]:.2f})')
        current_ax.set_xlim(0, 5.5) # Keep 0 in case of errors
        current_ax.set_ylabel('Frequency')
        current_ax.set_xlabel('Values' if metric_idx == len(metric_values)-1 else '')
        
    return error_bars, run_stats

def plot_question_scores(metric_names, grouped_values, colors):
    """Plot scores for each question across metrics"""
    
    plt.figure(figsize=(10, 6))

    # Define colors for each metric
    colors = plt.cm.Set3(np.linspace(0, 1, len(metric_names)))

    # First count all frequencies per score (1-5) per metric for one run over all questions
    question_scores_by_metric = {metric: [] for metric in metric_names}
    score_metric_counts = {}

    #Plot each metric's values and store question scores
    for i, (metric, question_scores) in enumerate(zip(metric_names, grouped_values)):
        width = 0.8 / len(question_scores)  # Width of each metric's bar
        
        for j, val in enumerate(question_scores): #Create a bar for each question's score
            plt.bar(i + j * width, val, width=width, color=colors[i], alpha=0.5, 
                    label=metric if j == 0 else "")
                    # i is the index of metric and determines the base position of a group of bars corresponding to that metric.
                    # j*width adds an offset to the base position to separate individual bars within the same group (metric). 
                    # Each j corresponds to a different value in question_scores, creating distinct bars for the values of question_scores for the same metric.
                    # By combining the above two, we get the exact x-position of a specific bar     
            question_scores_by_metric[metric].append((j, val))

        counts = Counter(question_scores)
        for score, freq in counts.items():
            if score not in score_metric_counts:
                score_metric_counts[score] = {}
            score_metric_counts[score][metric] = freq #Keeps track of how many times each metric gets a specific score over all questions (for one run)
            # {4: {'completeness': 1, 'confidence': 1, 'factuality': 1, 'judgement': 1}, 3: {'completeness': 1, 'relevance': 2, 'conciseness': 2, ....}

    return question_scores_by_metric, score_metric_counts

#https://python.langchain.com/v0.2/docs/integrations/chat/openai/
def load_model_stats(judge_model): #In case we had to restart the loop - some models didn't run - Keep track of all model stats
    """
    Loads previously saved model statistics and run metrics from JSON files.
    This allows the evaluation to be resumed or extended without re-calculating past results.
    """
    judge_name = "_".join(judge_model.split('/')[1:])
    print(f"Checking if we can load stats for {judge_name}")
    
    try:
        with open(f'stats_{judge_name}.json', 'r') as f:
            all_models_stats = json.load(f)
    except FileNotFoundError:
        all_models_stats = {}  # Used in comparison between models

    try: # a dict with num_models keys, each having a list with one dict. The dict has num_metrics keys,
        # and each key has a list with number_questions values like so: {"openai/o1-mini": [{"completeness_descr": [5, 0, 0],...
        with open(f'all_runs_model_metrics_{judge_name}.json', 'r') as f:
            all_runs_model_metrics = json.load(f)
    except FileNotFoundError:
        all_runs_model_metrics = {}  # Used in plotting metrics
        
    return all_models_stats, all_runs_model_metrics

def save_model_stats(all_models_stats, all_runs_model_metrics, judge_model):
    """
    Saves the aggregated model statistics and run metrics to JSON files.
    """
    judge_name = "_".join(judge_model.split('/')[1:])
    stats_file = f'all_models_stats_{judge_name}.json'
    metrics_file = f'all_runs_model_metrics_{judge_name}.json'
    
    with open(stats_file, 'w') as f:
        json.dump(all_models_stats, f, indent=4)
        
    with open(metrics_file, 'w') as f:
        json.dump(all_runs_model_metrics, f, indent=4)

def calculate_model_stats(model_name, n_resamples, judge_model):
    """
    Calculates and returns aggregated statistics for a given model across all its resamples.
    It reads the individual Excel result files for each resample, computes the mean scores,
    standard deviations, and other metrics, and aggregates them.
    """
    model_param = model_name.replace('/', '_')
    judge_param = judge_model.replace('/', '_')
    
    all_runs_for_model = []
    
    for i in range(1, n_resamples + 1):
        folder_name = f"resample_{i}"
        file_path = os.path.join(folder_name, f"resample_{i}_with_{model_param}.xlsx")
        if os.path.exists(file_path):
            df = pd.read_excel(file_path)
            all_runs_for_model.append(df)
    
    if not all_runs_for_model:
        return {}, {}

    combined_df = pd.concat(all_runs_for_model, ignore_index=True)
    
    metric_cols = [f"metric_{m.replace('_descr', '')}_{judge_param}" for m in list_of_metrics]
    
    # Calculate overall stats
    all_scores = combined_df[metric_cols].values.flatten()
    mean_score = np.mean(all_scores)
    std_dev = np.std(all_scores)
    total_runs = len(combined_df)
    
    # Calculate mean for each metric
    mean_metrics = {metric.replace(f"_{judge_param}", ""): combined_df[metric].mean() for metric in metric_cols}

    model_total_stats = {
        "mean_score": mean_score,
        "std_dev": std_dev,
        "total_runs": total_runs,
        "mean_metrics": mean_metrics
    }
    
    # This structure is kept to align with the notebook's data aggregation logic
    model_runs_metrics = {col: combined_df[col].tolist() for col in metric_cols}
    
    return model_runs_metrics, model_total_stats 

def aggregate_metrics_by_model(all_runs_model_metrics):
    """
    Calculate mean and std of each metric over all runs for each model
    
    Args:
        all_runs_model_metrics (dict): Dictionary containing metrics for all model runs
        
    Returns:
        dict: Aggregated metrics by model
    """
    aggregated_metrics_by_model = {}

    for model, model_data in all_runs_model_metrics.items():
        if model not in aggregated_metrics_by_model:
            aggregated_metrics_by_model[model] = {}
        
        for run_data in model_data:
            for metric_name, metric_values in run_data.items():
                if metric_name not in aggregated_metrics_by_model[model]:
                    aggregated_metrics_by_model[model][metric_name] = []
                
                if isinstance(metric_values, list) and all(isinstance(x, (int, float)) for x in metric_values):
                    aggregated_metrics_by_model[model][metric_name].extend(metric_values)
                else:
                    print(metric_values)
                    with open(f"aggregated_metrics_by_model_{model}.txt", "a") as f:
                        f.write(f"metric_values: {metric_values}\n")
                    
    return aggregated_metrics_by_model

def print_aggregated_metrics(aggregated_metrics, judge_model):
    for model, metrics in aggregated_metrics.items():
        print(f"\nModel: {model}")
        print("-" * (len(model) + 8))
        
        # Create file first
        model_name="_".join(model.split("/")[1:])
        judge_name = "_".join(judge_model.split("/")[1:])
        with open(f"aggregated_metrics_{model_name}_{judge_name}.txt", "w") as f:
            pass
            
        with open(f"aggregated_metrics_{model_name}_{judge_name}.txt", "a") as f:
            f.write(f"Model: {model_name}\n")
            f.write("-" * (len(model_name) + 8) + "\n")
        
        for metric_name, values in metrics.items():
            if values:
                mean_value = np.mean(values)
                std_value = np.std(values)
                print(f"{metric_name}:")
                print(f"  Mean: {mean_value:.4f}")
                print(f"  Std:  {std_value:.4f}")
                with open(f"aggregated_metrics_{model_name}_{judge_name}.txt", "a") as f:
                    f.write(f"{metric_name}: {mean_value:.4f} {std_value:.4f}\n")

def calculate_model_metrics(list_of_metric_names, aggregated_metrics):
    metric_means = {m: [] for m in list_of_metric_names}
    metric_stds = {m: [] for m in list_of_metric_names}
    model_names = []

    for model, model_metrics in aggregated_metrics.items():
        model_names.append(model.split('/')[-1].replace('-descr', '').replace('_descr', ''))
        for m in list_of_metric_names:
            key = f"{m}_descr"
            if key in model_metrics and model_metrics[key]:
                values = model_metrics[key]
                metric_means[m].append(np.mean(values))
                metric_stds[m].append(np.std(values))
            else:
                metric_means[m].append(0.0)
                metric_stds[m].append(0.0)
    return model_names, metric_means, metric_stds

def create_comparison_table(comparison_results, metrics):
    """
    Creates a formatted table from comparison results.
    
    Args:
        comparison_results (dict): The comparison results dictionary
        metrics (list): List of metrics to include
        
    Returns:
        str: Formatted markdown table
    """
    # Table header
    table = "| Metric | Model | Baseline | Model - Baseline | 95% Conf. Interval | Correlation |\n"
    table += "|--------|--------|-----------|-----------------|-------------------|-------------|\n"
    
    # Add rows for each comparison and metric
    for pair in comparison_results:
        model1, model2 = pair.split('_vs_')
        for metric in metrics:
            results = comparison_results[pair][metric]
            
            row = f"| {metric} | {model1} | {model2} | "
            row += f"{results['mean_difference']:.1%} | "
            row += f"({results['ci_low']:.1%}, {results['ci_high']:.1%}) | "
            row += f"{results['pearson_correlation']:.2f} |\n"
            
            table += row
            
    return table

def compare_model_performances(all_runs_model_metrics, judge_model): 
    """
    Performs statistical comparison between models using paired differences, standard errors,
    and Pearson correlation coefficients following section 4.2 methodology.
    
    Args:
        all_runs_model_metrics (dict): Dictionary containing raw metrics for each model/run/question
        
    Returns:
        dict: Dictionary containing pairwise comparison results
    """
    import numpy as np
    from scipy import stats
    import itertools
    
    # Get all model pairs for comparison
    models = list(all_runs_model_metrics.keys())
    model_pairs = list(itertools.combinations(models, 2))
    
    # Store results
    comparison_results = {}
    
    for model1, model2 in model_pairs:
        judge_name = "_".join(judge_model.split('/')[1:])
        comparison_key = f"{model1.split('/')[-1]}_vs_{model2.split('/')[-1]}_with_{judge_name}"
        comparison_results[comparison_key] = {}
        
        # Get metrics (removing '_descr' suffix)
        metrics = [metric.replace('_descr', '') for metric in list(all_runs_model_metrics[model1][0].keys())]
        
        # Create file for this model comparison
        variance_results_text = f"\n=== Variance Analysis Results for {comparison_key} ===\n"
        
        for metric in metrics:
            # Calculate differences and correlations for each resample
            resample_differences = []
            resample_ses = []
            correlations = []
            model1_variances = []  # Initialize list
            model2_variances = []  # Initialize list
            
            # Iterate through resamples - Same number for both models
            for resample_idx in range(len(all_runs_model_metrics[model1])):
                # Get scores for both models for this resample
                scores1 = all_runs_model_metrics[model1][resample_idx][f'{metric}_descr']
                scores2 = all_runs_model_metrics[model2][resample_idx][f'{metric}_descr']
                
                # Calculate differences for each question
                question_differences = np.array(scores1) - np.array(scores2)
                
                # Calculate mean difference for this resample
                mean_diff = np.mean(question_differences) #Same as the formula in the paper since mean(a-b)=mean(a)-mean(b)
                
                # Calculate standard error for this resample - Paired analysis (section 4.2)
                n = len(question_differences)
                se = np.sqrt(np.sum((question_differences - mean_diff)**2) / (n * (n-1))) if n > 1 else np.nan

                # # Calculate standard errors for each model - Unpaired analysis (section 4.1)
                # n = len(scores1)
                # sea = np.sqrt(np.sum((scores1 - np.mean(scores1))**2) / (n * (n - 1))) if n > 1 else np.nan
                # seb = np.sqrt(np.sum((scores2 - np.mean(scores2))**2) / (n * (n - 1))) if n > 1 else np.nan

                # # Calculate the combined standard error as sqrt(sea^2 + seb^2)
                # se = np.sqrt(sea**2 + seb**2)

                # Calculate variances for each model
                var1 = np.var(scores1, ddof=1)  # Using ddof=1 for sample variance
                var2 = np.var(scores2, ddof=1)
                model1_variances.append(var1)
                model2_variances.append(var2)
                
                # Calculate Pearson correlation
                correlation, _ = stats.pearsonr(scores1, scores2)
                
                resample_differences.append(mean_diff)
                resample_ses.append(se)
                correlations.append(correlation)
            
            # Convert to numpy arrays
            resample_differences = np.array(resample_differences)
            resample_ses = np.array(resample_ses)
            correlations = np.array(correlations)
            model1_variances = np.array(model1_variances)
            model2_variances = np.array(model2_variances)
            print("resample_differences",resample_differences)
            print("resample_ses",resample_ses)
            print("correlations",correlations)
            print(f"Model 1 variances: {model1_variances}")
            print(f"Model 2 variances: {model2_variances}")
            with open(f"model_variances_{comparison_key}.txt", "a") as f:
                f.write(f"Model 1 variances: {model1_variances}\n")
                f.write(f"Model 2 variances: {model2_variances}\n")
                f.write(f"resample_differences: {resample_differences}\n")
                f.write(f"resample_ses: {resample_ses}\n")
                f.write(f"correlations: {correlations}\n")
          
            # Calculate overall mean difference over all resamples
            overall_mean_diff = np.mean(resample_differences)
            print("overall_mean_diff",overall_mean_diff)
            with open(f"model_variances_{comparison_key}.txt", "a") as f:
                f.write(f"overall_mean_diff: {overall_mean_diff}\n")
            
            #We want an aggregated SE across all resamples for the same questions (same paired differences)
            #This approach accounts for the fact that each resampling provides a different estimate of the variance of the same underlying distribution, 
            # and averaging these estimates gives a better representation of the overall uncertainty.

            # Calculate pooled standard error across resamples
            R = len(resample_differences)
            pooled_se = np.sqrt(np.sum(resample_ses**2) / (R**2))
            print("pooled_se",pooled_se)
            with open(f"model_variances_{comparison_key}.txt", "a") as f:
                f.write(f"pooled_se: {pooled_se}\n")
            
            # # If the resampling results are independent estimates of variance (i.e., combining uncertainty estimates from independent sources), the combined variance is
            # # the sum of all individual variances, and the combined standard error is given below (goal to capture total variability)
            # # Calculate the overall combined SE across all resamples
            # combined_se = np.sqrt(np.nansum(np.array(resample_ses)**2))

            # Calculate overall variance reduction across all resamples
            n = len(scores1)
            
            # Calculate mean variances across resamples
            mean_var1 = np.mean(model1_variances)  # Var(sA)
            mean_var2 = np.mean(model2_variances)  # Var(sB)
            
            # Calculate mean correlation across resamples
            mean_correlation = np.mean(correlations)
            
            # Calculate covariance between model scores
            mean_cov = mean_correlation * np.sqrt(mean_var1 * mean_var2)  # Cov(sA, sB)
            
            # Calculate variance for unpaired case: Var(μA-B,unpaired) = (Var(sA) + Var(sB))/n
            var_unpaired = (mean_var1 + mean_var2) / n
            
            # Calculate variance for paired case: Var(μA-B,paired) = (Var(sA) + Var(sB) - 2Cov(sA,sB))/n
            var_paired = (mean_var1 + mean_var2 - 2 * mean_cov) / n
            
            # The reduction in variance is: Var(μA-B,unpaired) - Var(μA-B,paired) = 2Cov(xA,xB)/n
            variance_reduction = 2 * mean_cov / n  # This should equal var_unpaired - var_paired
            
            # Calculate percentage reduction in variance
            percent_reduction = (variance_reduction / var_unpaired) * 100 if var_unpaired != 0 else 0

            # Add results for this metric to the text
            variance_results_text += f"\nMetric: {metric}\n"
            variance_results_text += f"Mean Model 1 variance (Var(sA)): {mean_var1:.6f}\n"
            variance_results_text += f"Mean Model 2 variance (Var(sB)): {mean_var2:.6f}\n"
            variance_results_text += f"Mean covariance (Cov(sA,sB)): {mean_cov:.6f}\n"
            variance_results_text += f"Unpaired variance: {var_unpaired:.6f}\n"
            variance_results_text += f"Paired variance: {var_paired:.6f}\n"
            variance_results_text += f"Variance reduction (2Cov(xA,xB)/n): {variance_reduction:.6f}\n"
            variance_results_text += f"Percent reduction: {percent_reduction:.1f}%\n"

            # # Calculate t-statistic and p-value
            # t_stat = overall_mean_diff / pooled_se if pooled_se != 0 else np.nan
            # df = R - 1  # degrees of freedom
            # p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df)) if not np.isnan(t_stat) else np.nan
            
            # # Calculate confidence interval
            # t_crit = stats.t.ppf(0.975, df)  # 95% CI
            # ci_margin = t_crit * pooled_se

            # Calculate z-statistic and CI using standard normal distribution
            z_stat = overall_mean_diff / pooled_se if pooled_se != 0 else np.nan
            
            # Calculate confidence interval using 1.96 for 95% CI
            ci_margin = 1.96 * pooled_se
            
            # Calculate p-value using standard normal distribution
            #For a two-tailed test p = 2 × (1 − Φ(|z|)), where Φ(z) is the cumulative distribution function (CDF) of the standard normal distribution.
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat))) if not np.isnan(z_stat) else np.nan
            
            # # Calculate average Pearson correlation - not accurate when correlations close to 1 or -1, variances differences across resamples, sample size is small.
            # avg_correlation = np.mean(correlations)

            #Apply Fisher z-transformation
            z_values = [0.5 * np.log((1 + r) / (1 - r)) for r in correlations]

            # Compute the mean Fisher z-value
            z_mean = np.mean(z_values)

            #Back-transform to Pearson correlation scale
            overall_correlation = (np.exp(2 * z_mean) - 1) / (np.exp(2 * z_mean) + 1)
            
            # Store results
            comparison_results[comparison_key][metric] = {
                "mean_difference": overall_mean_diff,
                "pooled_standard_error": pooled_se,
                "ci_low": overall_mean_diff - ci_margin,
                "ci_high": overall_mean_diff + ci_margin,
                # "t_statistic": t_stat,
                "z_statistic": z_stat,
                "p_value": p_value,
                "significant": p_value < 0.05 if not np.isnan(p_value) else None,
                "better_model": model1.split('/')[-1] if overall_mean_diff > 0 else model2.split('/')[-1],
                "pearson_correlation": overall_correlation
            }
        
        # Write all metrics results for this model comparison to a single file
        with open(f'variance_results_{comparison_key}.txt', 'w') as f:
            variance_results_text += f"Overall Variance Reduction Analysis:\n"
            f.write(variance_results_text)
    
    return comparison_results

def perform_power_analysis(effect_size=0.5, alpha=0.05, power=0.8):
    """
    Perform power analysis to determine required sample size.
    
    Args:
        effect_size (float): Expected effect size (Cohen's d)
        alpha (float): Significance level
        power (float): Desired statistical power
        
    Returns:
        int: Required sample size per group
    """
    analysis = TTestIndPower()
    sample_size = analysis.solve_power(
        effect_size=effect_size,
        alpha=alpha,
        power=power,
        alternative='two-sided'
    )
    return int(np.ceil(sample_size))