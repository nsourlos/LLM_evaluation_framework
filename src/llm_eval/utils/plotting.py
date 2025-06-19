import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.colors as mcolors
from .statistics import plot_metric_distributions, plot_question_scores

def plot_ordered_scores(metric_names, question_scores_by_metric, colors):
    """Plot metrics ordered by score values"""
    plt.figure(figsize=(15, 10))
    
    for i, metric in enumerate(metric_names):
        plt.subplot(len(metric_names), 1, i+1)
        sorted_questions = sorted(question_scores_by_metric[metric], key=lambda x: x[1]) #Sort questions by score
        
        #Plot bars
        x_pos = range(len(sorted_questions))
        scores = [q[1] for q in sorted_questions]
        plt.bar(x_pos, scores, color=colors[i], alpha=0.5)

        #Add question indices as x-axis labels
        plt.xticks(x_pos, [str(q[0]) for q in sorted_questions])
        
        plt.ylabel(metric)
        plt.ylim(0, 5.5)
        plt.yticks(range(6)) # Set y-axis ticks from 0 to 5

        if i == len(metric_names)-1:
            plt.xlabel('Question number (ordered by score)')

def plot_accumulated_distributions(score_metric_counts, metric_names, colors):
    """Plot accumulated distribution of scores by metric"""
    legend_added = set()

    #For each score, plot metrics in order of frequency (highest frequency at bottom)
    for score in sorted(score_metric_counts.keys()):
        #Sort metrics by frequency for this score
        sorted_metrics = sorted(score_metric_counts[score].items(),
                            key=lambda x: x[1], #Use the frequency (second element of each tuple) as the sorting key
                            reverse=True) # highest frequency first
        bottom = 0
        for metric, freq in sorted_metrics:
            i = metric_names.index(metric) #get index for color
            plt.bar(score, freq,
                    width=0.4,
                    color=colors[i],
                    alpha=0.5,
                    label=metric if metric not in legend_added else "",
                    bottom=bottom)
            bottom += freq
            legend_added.add(metric)

def plot_figures_metrics(all_runs_model_metrics, metric_names, model_name, judge_model):
    """
    Creates visualizations and calculates statistics for evaluation metrics across multiple runs.

    Args:
        all_runs_model_metrics (dict): Nested dictionary containing evaluation metrics for each model and run.
            Structure: {model_id: [{metric1_descr_run1: [q1_score, q2_score, ...], 
                                  metric2_descr_run1: [q1_score, q2_score, ...], ...}, 
                                 {metric1_descr_run2: [q1_score, q2_score, ...],
                                  metric2_descr_run2: [q1_score, q2_score, ...], ...},
                                 ...num_runs]}
            Example: {'model1': [{'completeness_descr_run1': [4.5, 3.0, 4.0], 
                                'relevance_descr_run1': [3.5, 4.0, 3.0]}, ...,
                               {'completeness_descr_run2': [4.0, 3.5, 4.5],
                                'relevance_descr_run2': [3.0, 4.5, 3.5], ...},
                               ...num_runs]}
            Where each inner dictionary represents one run containing scores for each metric across all questions
        metric_names (list): Names of metrics to analyze and plot (e.g. ['completeness', 'relevance'])
        model_name (str): Name/identifier of the model being evaluated
        judge_model (str): Name/identifier of the model used for judging the evaluations

    Returns:
        dict: Summary statistics for each model, run and metric.
            Structure: {model_name: {run_idx: {metric_name: {
                'mean': float,
                'std_error': float, 
                'ci_low': float,
                'ci_high': float
            }}}}
            Example: {'anthropic/claude-3-5-sonnet': {
                '0': {'completeness': {'mean': 4.5, 'std_error': 0.5, 
                                     'ci_low': 3.52, 'ci_high': 5.48},
                      'relevance': {'mean': 3.5, 'std_error': 0.5,
                                  'ci_low': 2.52, 'ci_high': 4.48} , ...},
                '1': {'completeness': {'mean': 4.5, 'std_error': 0.5,
                                     'ci_low': 3.52, 'ci_high': 5.48},
                      'relevance': {'mean': 3.5, 'std_error': 0.5,
                                  'ci_low': 2.52, 'ci_high': 4.48}, ...},
                ...num_runs}}

    The function generates several visualization types:
    - Individual histograms for each metric showing score distributions
    - Error bars indicating means and confidence intervals
    - Overlapping bar plots comparing metrics
    - Stacked distribution plots showing relative frequencies of scores

    All plots are saved as PNG files with names indicating the judge model,
    evaluated model, run index, and plot type.
    """

    summary_stats_all_runs = {}  # Keep track of summary statistics over all runs

    for run_idx, metric_values_run in enumerate(all_runs_model_metrics[model_name]): #Loop over runs

        colors = sns.color_palette("Set3", len(metric_names))
        
        # Create two figures - one with separate subplots and one overlaid
        fig, axes = plt.subplots(len(metric_names), 1, figsize=(10, 18))
        plt.subplots_adjust(hspace=0.6, top=0.94)
        fig.suptitle(f'Metric Distributions for {model_name} (Run {run_idx})', fontsize=16)
        
        bin_edges = np.arange(0.0, 5.6, 0.2)  # Bins for range 0-5
        metric_names = [name.replace('_descr', '') for name in metric_values_run]
        
        error_bars, run_stats = plot_metric_distributions(metric_values_run, axes, colors, bin_edges, metric_names)
        
        # Save version without error bars
        plt.figure(fig.number)
        judge_name = "_".join(judge_model.split('/')[1:])
        plt.savefig(f"{judge_name}_judge_with_{model_name.replace('/', '_')}_run_{run_idx}_metric_distributions_no_error_bars.png")
        
        # Add error bars and save updated version
        for i, (mean, ylim, margin) in enumerate(error_bars):
            # Handle both single axis and array of axes for error bars
            if hasattr(axes, '__len__') and len(axes) > 1:
                current_ax = axes[i]
            else:
                current_ax = axes
            current_ax.errorbar(mean, ylim, xerr=margin, color='black', capsize=5, 
                               capthick=1, elinewidth=2, marker='o')
        
        plt.savefig(f"{judge_name}_judge_with_{model_name.replace('/', '_')}_run_{run_idx}_metric_distributions.png")
        plt.close('all')

        # Print summary statistics - Can also be seen in txt file. 
        print(f"\nSummary Statistics over run {run_idx}:")
        print("-" * 50)
        for metric, stats in run_stats.items():
            print(f"{metric}:")
            for key, value in stats.items():
                print(f"  {key}: {value:.2f}")
            print("-" * 50)

        summary_stats_all_runs[run_idx] = run_stats #For one run

        grouped_values=list(metric_values_run.values()) #Values of all metrics for one run over all questions. There are num_metrics lists in that list. 
        values = [val for sublist in grouped_values for val in sublist] #Flatten the list - Size is num_questions*num_metrics (1st metric questions, 2nd metric questions, etc)
        
        question_scores_by_metric, score_metric_counts = plot_question_scores(metric_names, grouped_values, colors)
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('Per-Metric Question Scores Distribution')
        plt.xticks(np.arange(len(metric_names)) + 0.1, metric_names)
        plt.yticks(range(6)) #Set y-ticks to 0-5
        plt.savefig(f"{judge_name}_judge_with_{model_name.replace('/', '_')}_run_{run_idx}_per_metric_question_scores.png")
        plt.close('all')

        # Plot ordered scores
        plot_ordered_scores(metric_names, question_scores_by_metric, colors)
        plt.suptitle('Question indices ordered by metric value')
        plt.tight_layout()
        plt.savefig(f"{judge_name}_judge_with_{model_name.replace('/', '_')}_run_{run_idx}_question_indices_ordered_by_metric_value.png")
        plt.close('all')

        # Plot accumulated distributions
        plt.figure(figsize=(10, 6))
        plot_accumulated_distributions(score_metric_counts, metric_names, colors)
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        plt.title('Score Distribution Histogram by Metric') 
        plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(np.arange(0, 6))
        plt.tight_layout()
        plt.savefig(f"{judge_name}_judge_with_{model_name.replace('/', '_')}_run_{run_idx}_score_distribution_histogram_by_metric.png")
        plt.close('all')

    return summary_stats_all_runs

def plot_model_comparison(models, metrics, metric_means, metric_stds, save_prefix=""):
    """
    Plot comparison charts for multiple models across different metrics.
    
    Args:
        models (list): List of model names to compare
        metrics (list): List of metric names to display
        metric_means (dict): Dictionary mapping metrics to lists of mean values for each model
        metric_stds (dict): Dictionary mapping metrics to lists of standard deviation values for each model
        save_prefix (str, optional): Prefix for saved image filenames
    
    Returns:
        None: Plots are displayed and saved to files
    """
    # Your preferred colors first
    base_colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2']

    # If we need more colors than in the base list
    if len(models) > len(base_colors):
        # Generate the additional colors needed
        extra_needed = len(models) - len(base_colors)
        hsv_colors = [(i/extra_needed, 0.8, 0.8) for i in range(extra_needed)]
        rgb_colors = [mcolors.hsv_to_rgb(hsv) for hsv in hsv_colors]
        extra_colors = [mcolors.to_hex(rgb) for rgb in rgb_colors]
        
        # Combine base colors with extra colors
        model_colors = base_colors + extra_colors
    else:
        # Use just the base colors up to the number needed
        model_colors = base_colors[:len(models)]

    # Plot 1: Grid of metrics
    # Calculate the number of rows and columns needed for the subplots
    num_metrics = len(metrics)
    num_cols = 3  # Keep 3 columns as in original
    num_rows = (num_metrics + num_cols - 1) // num_cols  # Ceiling division to ensure enough subplots
    
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(16, 5 * num_rows))
    axs = axs.flatten()

    for i, metric in enumerate(metrics):
        means = metric_means[metric]
        stds = metric_stds[metric]
        
        # Filter out zero values
        valid_indices = [j for j, mean in enumerate(means) if mean > 0]
        valid_means = [means[j] for j in valid_indices]
        valid_stds = [stds[j] for j in valid_indices]
        valid_colors = [model_colors[j] for j in valid_indices]
        valid_model_labels = [models[j] for j in valid_indices]
        
        # Create new x positions without gaps
        valid_x = np.arange(len(valid_indices))
        
        bars = axs[i].bar(valid_x, valid_means, yerr=valid_stds, capsize=5, color=valid_colors)
        axs[i].set_title(metric)
        axs[i].set_xticks(valid_x)
        axs[i].set_xticklabels(valid_model_labels, rotation=45, ha='right')
        axs[i].set_ylim(0, 6.2)  # higher to accommodate error bar labels
        axs[i].set_yticks(np.arange(0, 6, 1))
        axs[i].grid(axis='y', linestyle='dotted', color='gray', linewidth=0.8)
        if i in [0, 3]:
            axs[i].set_ylabel("Score")
        if i >= 3:
            axs[i].set_xlabel("LLM")
        for j, bar in enumerate(bars):
            top = valid_means[j] + valid_stds[j]
            axs[i].text(bar.get_x() + bar.get_width() / 2, top + 0.05, f"{valid_means[j]:.2f}",
                        ha='center', va='bottom', fontsize=9, rotation=90)

    # Hide any unused subplots
    for i in range(num_metrics, len(axs)):
        axs[i].set_visible(False)

    fig.suptitle("Metric Comparison Across LLMs (± std dev)", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"metric_comparison_grid_judge_{save_prefix}.png", dpi=300, bbox_inches='tight')
    # plt.show()

    # Plot 2: Grouped bar chart
    width = 0.12
    fig, ax = plt.subplots(figsize=(18, 7))

    # Ensure we have enough colors for all metrics
    metric_colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b']
    if len(metrics) > len(metric_colors):
        # Generate additional colors if needed
        extra_needed = len(metrics) - len(metric_colors)
        hsv_colors = [(i/extra_needed, 0.8, 0.8) for i in range(extra_needed)]
        rgb_colors = [mcolors.hsv_to_rgb(hsv) for hsv in hsv_colors]
        extra_colors = [mcolors.to_hex(rgb) for rgb in rgb_colors]
        metric_colors = metric_colors + extra_colors
    else:
        metric_colors = metric_colors[:len(metrics)]
        
    max_y = 0

    # Get all valid model indices (models with at least one non-zero metric)
    all_valid_indices = set()
    for metric in metrics:
        for j, mean in enumerate(metric_means[metric]):
            if mean > 0:
                all_valid_indices.add(j)

    all_valid_indices = sorted(list(all_valid_indices))
    valid_models = [models[j] for j in all_valid_indices]

    # Create new x positions without gaps
    x = np.arange(len(all_valid_indices))

    for i, metric in enumerate(metrics):
        means = metric_means[metric]
        stds = metric_stds[metric]
        
        # Filter out zero values but maintain position for valid models
        valid_means = []
        valid_stds = []
        valid_positions = []
        
        for idx, j in enumerate(all_valid_indices):
            if means[j] > 0:
                valid_means.append(means[j])
                valid_stds.append(stds[j])
                valid_positions.append(x[idx])
        
        # Skip if no valid data for this metric
        if not valid_means:
            continue
            
        offset = (i - len(metrics)/2 + 0.5) * width
        positions = [pos + offset for pos in valid_positions]
        
        bars = ax.bar(positions, valid_means, width, yerr=valid_stds, label=metric, color=metric_colors[i], capsize=4)
        for j, bar in enumerate(bars):
            top = valid_means[j] + valid_stds[j]
            max_y = max(max_y, top)
            ax.text(bar.get_x() + bar.get_width() / 2, top + 0.1, f"{valid_means[j]:.2f}",
                    ha='center', va='bottom', fontsize=9, rotation=90)

    ax.set_ylabel('Score')
    ax.set_title('LLM Metric Comparison (Mean ± Std Dev)')
    ax.set_xticks(x)
    ax.set_xticklabels(valid_models, rotation=45, ha='right')
    ax.set_yticks(np.arange(0, 6, 1))
    ax.set_ylim(0, max_y + 0.5)
    ax.grid(axis='y', linestyle='dotted', color='gray', linewidth=0.8)
    ax.legend(title='Metric', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout(rect=[0, 0, 0.88, 0.97])
    plt.savefig(f"metric_comparison_summary_by_LLM_judge_{save_prefix}.png", dpi=300, bbox_inches='tight')
    # plt.show()

def plot_spider_chart(models, metrics, metric_means, save_prefix=""):
    """
    Plot a spider chart comparing multiple models across different metrics.
    
    Args:
        models (list): List of model names to compare
        metrics (list): List of metric names to display
        metric_means (dict): Dictionary mapping metrics to lists of mean values for each model
        save_prefix (str, optional): Prefix for saved image filenames
    """
    # Filter out models with zero values
    all_valid_indices = set()
    for metric in metrics:
        means = metric_means[metric]
        for j, mean in enumerate(means):
            if mean > 0:
                all_valid_indices.add(j)

    all_valid_indices = sorted(list(all_valid_indices))
    valid_models = [models[j] for j in all_valid_indices]

    N = len(metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    plt.xticks(angles[:-1], metrics, fontsize=12)

    ax.set_rlabel_position(0)
    plt.yticks([0, 1, 2, 3, 4, 5], ["0", "1", "2", "3", "4", "5"], color="grey", size=8)
    plt.ylim(0, 5)

    markers = ['o', 's', '^', 'D', 'v', 'p', 'h', '*', 'x', '+'][:len(valid_models)]
    
    for i, model_idx in enumerate(all_valid_indices):
        values = []
        for metric in metrics:
            values.append(metric_means[metric][model_idx])
        values += values[:1]  # repeat first value to close the loop
        ax.plot(angles, values, linewidth=2, linestyle='solid', marker=markers[i], label=valid_models[i])
        ax.fill(angles, values, alpha=0.1)

    plt.title('LLM Performance Comparison', size=16, y=1.08)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    plt.savefig(f"spider_chart_judge_{save_prefix}.png", dpi=300, bbox_inches='tight')
    # plt.show()

def plot_and_save_model_comparisons(comparison_results, list_of_metrics, suffix):
    # Extract metrics and models from comparison_results
    metrics = [metric.replace('_descr', '') for metric in list_of_metrics]
    model_pairs = list(comparison_results.keys())

    # Create figure with subplots for each metric
    # Calculate number of rows needed based on number of metrics
    num_metrics = len(metrics)
    num_rows = (num_metrics + 2) // 3  # Using 3 columns, calculate rows needed (ceiling division)
    fig, axes = plt.subplots(num_rows, 3, figsize=(25, 20 * num_rows / 2), dpi=600)  # Adjusted figsize proportionally
    fig.suptitle('Model Comparison Results by Metric for judge model '+suffix, fontsize=16, y=1.05)
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Extract data for this metric
        means = []
        cis = []
        labels = []
        
        for pair in model_pairs:
            metric_data = comparison_results[pair][metric]
            means.append(metric_data['mean_difference'])
            # ci_margin = metric_data['ci_margin']
            cis.append([metric_data['ci_low'], 
                       metric_data['ci_high']])
            labels.append(pair.split('with')[0].strip()) #Append only the model name comparisons

        # Create bar plot
        bars = ax.bar(range(len(means)), means)
        
        # Add error bars for confidence intervals
        ax.errorbar(range(len(means)), means, 
                   yerr=[[m - ci[0] for m, ci in zip(means, cis)],
                         [ci[1] - m for m, ci in zip(means, cis)]],
                   fmt='none', color='black', capsize=5)
        
        # Add horizontal line at y=0
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Customize plot
        ax.set_title(f'{metric.capitalize()}')
        ax.set_xticks(range(len(means)))
        ax.set_xticklabels(labels, rotation=90) # Changed to vertical labels
        ax.set_ylabel('Mean Difference')
        
        # Color bars based on statistical significance
        for j, bar in enumerate(bars):
            if comparison_results[model_pairs[j]][metric]['p_value'] < 0.05:
                bar.set_color('darkred')
            else:
                bar.set_color('lightgray')

    # Hide any unused subplots
    for i in range(num_metrics, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    # Save plot before showing with high resolution
    plt.savefig(f'model_comparisons_{suffix}.png', bbox_inches='tight', dpi=600)  # Increased DPI for higher resolution

    # Show plot after saving
    # plt.show()