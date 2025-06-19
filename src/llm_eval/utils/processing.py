import pandas as pd
import numpy as np
from termcolor import colored
import traceback
from langsmith import Client

from src.llm_eval.config import embedding_model, reranker_model_name, n_resamples, judge_model
from .rag import initialize_vectorstore, initialize_reranker

def process_evaluation_results(langsmith_api_key, dataset_langsmith, use_RAG=False):
    """Extract questions and answers from evaluation results."""
    #https://docs.smith.langchain.com/tutorials/Developers/evaluation

    # Get unique questions/answers
    client = Client(api_key=langsmith_api_key)
    questions_answers=[x for x in client.list_examples(dataset_id=dataset_langsmith.id)]
    list_of_questions=[x.inputs['question'] for x in questions_answers]
    list_of_answers=[x.outputs['answer'] for x in questions_answers]
        
    # with open('list_of_questions.txt', 'w') as f:
    #     f.write(str(list_of_questions))
    # with open('list_of_answers.txt', 'w') as f:
    #     f.write(str(list_of_answers))

    if use_RAG==True:
        # Initialize vectorstore
        vectorstore = initialize_vectorstore(list_of_questions, list_of_answers, embedding_model)
        # Initialize reranker
        reranker = initialize_reranker(reranker_model_name)
    
    results_df = pd.DataFrame({
        'questions': list_of_questions,
        'answers': list_of_answers
    })

    if use_RAG:
        return results_df, list_of_questions, vectorstore, reranker
    else:
        return results_df, list_of_questions, None, None

def process_metrics(resample_results, list_of_metrics, list_of_questions, resample_idx, results_df, model_name, indices_to_reorder):
    """
    Process metrics for a single resample and update results DataFrame.
    
    Args:
        resample_results: Results from current resample
        list_of_metrics: List of metrics to process
        resample_idx: Current resample index
        results_df: DataFrame to update with metrics
        model_name: Name of the model being evaluated
        indices_to_reorder: Indices to reorder the metrics
        
    Returns:
        individual_run_metric_scores, metrics, results_df
    """

    metrics = [] #This should be the same as resample_results (list) except when there are 'traceback' errors where it will be 0.
    # metrics format will be:[[EvaluationResult(key='completeness', score=4, value='To evaluate the .... - It has num_questions sublists, each with num_metrics values

    model_parameter = "_".join(model_name.split('/')[1:])

    for result in resample_results:
        if result['run'].outputs['output'] is None or not result['evaluation_results']['results']: #or result['run'].error is not None - Same as first condition
            metrics.append(0)  # Use 0 to indicate failed evaluation - We might even get in here when LangSmith API connection issues
            print("Error: No metric value found!")
            #Also print which condition is true
            print("result['run'].outputs['output'] is None",result['run'].outputs['output'] is None)
            print("not result['evaluation_results']['results']",not result['evaluation_results']['results'])
            # Log the error conditions to a file
            with open('error_conditions_'+str(resample_idx)+'_'+str(model_parameter)+'.txt', 'a', encoding='utf-8') as f:
                f.write("Error: No metric value found! \n")
                f.write(f"result['run'].outputs['output'] is None: {result['run'].outputs['output'] is None}\n")
                f.write(f"not result['evaluation_results']['results']: {not result['evaluation_results']['results']}\n")
                f.write(f"result['evaluation_results']['results'] {result['evaluation_results']['results']}\n")
                f.write("\n")
        else:
            metrics.append(result['evaluation_results']['results'])
    
    # Reorder metrics based on indices_to_reorder
    reordered_metrics = [metrics[i] for i in indices_to_reorder]
    metrics = reordered_metrics

    assert len(resample_results)==len(list_of_questions), f"Number of resample results not matching num_questions. Got {len(resample_results)} resample \
        results but expected {len(list_of_questions)}"
    #Format is [{'run': RunTree(id=UUID('b7aea73... and there are num_questions runs. There are multiple files, one for each model and one for each resample (for main judge only)
    
    #A list with num_questions sublists, each with num_metrics values in the format:
    #[EvaluationResult(key='completeness', score=5, value='To evaluate
    with open('process_metrics_'+str(resample_idx)+'_'+str(model_parameter)+'.txt', 'w', encoding='utf-8') as f:
        f.write(str(metrics))

    assert len(metrics)==len(list_of_questions), f"Number of metrics not matching num_questions. Got {len(metrics)} metrics but expected {len(list_of_questions)}"
    
    #This is at the end a dict with num_metrics keys and each key has num_questions values.
    #Example: {'completeness_descr': [4, 3, 3, 5, 5, 4, 3], 'relevance_descr': [4, 3, 3, 3, 4, 3, 1], ....} assuming 7 questions
    individual_run_metric_scores = {} #Keep track of scores of all metrics over all questions for one resample

    for metric_idx, metric_name in enumerate(list_of_metrics): #Get specific metric name and values over all questions for the current resample

        clean_metric_names, metric_scores, metric_prompts = [], [], [] #Metric scores and prompts for all questions for a given resample - num_questions elements each time
        
        #Get all metric keys for the current resample over all questions, handling potential missing keys (values set to 0 for those - they are errors)
        for m in metrics:
            if m == 0: #If there is an error
                key = metric_name.replace('_descr','')
                score = 0
                prompt=""
            else:
                try:
                    key = m[metric_idx].key #Metric name
                    score = m[metric_idx].score ##Scores of a given metric over all questions for a given resample
                    prompt = m[metric_idx].value #Prompt used for the evaluation
                except:
                    print("Error: Metric not found - Shouldn't get here")
                    with open('error_conditions_'+str(resample_idx)+'_'+str(model_parameter)+'.txt', 'a') as f:
                        f.write("Error: Metric not found - Shouldn't get here \n")
                    key = metric_name.replace('_descr','')
                    score = 0
                    prompt = ""
                
            clean_metric_names.append(key)
            metric_scores.append(score)
            metric_prompts.append(prompt)
            
        assert all(name == metric_name.replace('_descr','') for name in clean_metric_names), f"Metric keys not matching: clean_metric_names={clean_metric_names}, \
            expected={metric_name.replace('_descr','')} and their values: {metric_scores}"
            
        assert len(metric_scores)==len(list_of_questions), f"Number of metric scores not matching num_questions. Got {len(metric_scores)} metric scores \
            but expected {len(list_of_questions)}"
            
        assert len(metric_prompts)==len(list_of_questions), f"Number of metric prompts not matching num_questions. Got {len(metric_prompts)} metric prompts \
            but expected {len(list_of_questions)}"

        # Update results DataFrame
        clean_metric_name = clean_metric_names[0] #Just one metric name without the _descr
        results_df[f'metric_{clean_metric_name}_{resample_idx+1}'] = metric_scores
        results_df[f'prompt_{clean_metric_name}_{resample_idx+1}'] = metric_prompts
        
        # Store scores for return
        individual_run_metric_scores[metric_name] = metric_scores #len is num_metrics

    return individual_run_metric_scores, metrics, results_df

def process_metrics_second_judge(excel_path, list_of_metrics, n_resamples, model_name, question_col="questions", judge_model_2=judge_model):
    """
    Loads the Excel file and processes metrics and prompts for the second judge model.
    Returns:
        - all_run_metric_scores: list of dicts, one per resample, {metric: [scores for all questions]}
        - all_run_metric_prompts: list of dicts, one per resample, {metric: [prompts for all questions]}
    """
    if judge_model_2!='openai/gpt4o-mini':
        df = pd.read_excel(excel_path)
        all_run_metric_scores = []
        all_run_metric_prompts = []
        num_questions = len(df[question_col])

        at_least_one_nan = False

        for resample_idx in range(n_resamples):
            run_scores = {}
            run_prompts = {}
            for metric_name in list_of_metrics:
                clean_name = metric_name.replace('_descr', '')
                judge_name = "_".join(judge_model_2.split('/')[1:])
                score_col = f"metric_{clean_name}_{resample_idx+1}_{judge_name}"
                prompt_col = f"prompt_{clean_name}_{resample_idx+1}_{judge_name}"
                if score_col in df.columns:
                    # flag_nan = False
                    if df[score_col].isna().any():
                        # flag_nan = True
                        at_least_one_nan = True
                        with open(f"Column_scores_nan_{'_'.join(judge_model_2.split('/')[1:])}.txt", "a") as col_file:
                            col_file.write(f"Score col: {score_col} and values are \n {df[score_col]} \n \n")
                    
                    df[score_col] = df[score_col].fillna(0)
                    run_scores[metric_name] = df[score_col].astype(int).tolist()

                    # if flag_nan: #Final output of this function saved elsewhere
                    #     with open(f"Column_scores_{'_'.join(judge_model_2.split('/')[1:])}.txt", "a") as col_file:
                    #         col_file.write(f"After conversion to int Score col: {score_col} and values are \n {df[score_col]} \n \n")
   
                else:
                    run_scores[metric_name] = [None] * num_questions
                    print(f"Metric {metric_name} not found in column {score_col}")
                    with open(f"Column_scores_{'_'.join(judge_model_2.split('/')[1:])}.txt", "a") as col_file:
                        col_file.write(f"Metric {metric_name} not found in column {score_col} \n \n")
                if prompt_col in df.columns:
                    run_prompts[metric_name] = df[prompt_col].tolist()
                else:
                    run_prompts[metric_name] = [None] * num_questions
                    print(f"Prompt {metric_name} not found in column {prompt_col}")
                    with open(f"Column_scores_{'_'.join(judge_model_2.split('/')[1:])}.txt", "a") as col_file:
                        col_file.write(f"Prompt {metric_name} not found in column {prompt_col} \n \n")

            all_run_metric_scores.append(run_scores)
            all_run_metric_prompts.append(run_prompts)
        
        if at_least_one_nan:
            model_parameter = "_".join(model_name.split('/')[1:])
            with open(f"Column_scores_{'_'.join(judge_model_2.split('/')[1:])}.txt", "a") as col_file:
                col_file.write(f"####### Judge model: {model_parameter}...... \n \n")

        return all_run_metric_scores, all_run_metric_prompts
    else:
        return [], []

def calculate_metric_statistics(all_runs_metric_scores, list_of_metrics, num_questions, model_name, judge_model, n_resamples=n_resamples):
    """Calculate statistical metrics across resamples (reduce variance - step 3.1)."""
    metric_stats_resampling = {} # Calculate mean and standard error for each metric and question across K resamples
    #The above dict will have num_metrics elements, each with metric keys (e.g. mean, std, etc), that will have num_questions values
    #Example: {'completeness': {'mean':[4, 3, 3, 5, 5, 4, 3]}, #here for a dataset with 7 questions
    #          'relevance': {'mean':[4, 3, 3, 3, 4, 3, 2]}, ...}
    
    for metric in list_of_metrics:
        metric_stats_resampling[metric] = {
            'means': [],  # Mean score across K resamples for each question
            'standard_errors': [],  # Standard error of the mean for each question
            'conditional_vars': []  # Conditional variance reduced by factor of K
        }
        
        # For each question
        for q in range(num_questions):
            # Get K scores for this metric/question across all resamples (num_resamples elements each time in that list)
            scores = [run[metric][q] for run in all_runs_metric_scores]
            K = len(scores)  # Number of resamples
            assert len(scores)==n_resamples, f"Number of scores not matching num_resamples. Got {len(scores)} scores but expected {n_resamples}"
            
            # Calculate statistics
            mean = np.mean(scores) #Average score of each question for a given metric over all resamples
            var = np.var(scores) #Variance of the scores of each question for a given metric over all resamples
            # Calculate conditional variance reduced by factor of K. Var(mean) = σ²/K where σ² is the variance of individual scores
            conditional_var = var / K if K > 0 else 0
            standard_error = np.sqrt(conditional_var)
            
            # Store results
            metric_stats_resampling[metric]['means'].append(mean)
            metric_stats_resampling[metric]['standard_errors'].append(standard_error)
            metric_stats_resampling[metric]['conditional_vars'].append(conditional_var)

    model_parameter = "_".join(model_name.split('/')[1:])
    try:
        judge_name = "_".join(judge_model.split('/')[1:]) if '/' in judge_model else judge_model
        with open('metric_stats_resampling_'+str(model_parameter)+'_judge_'+str(judge_name)+'.txt', 'w') as f:
            f.write(f"judge is: {judge_name} \n")
            f.write(str(metric_stats_resampling))
    except:
        print(f"ERROR: Metric stats resampling not found for {model_parameter} and judge")
        print("metric_stats_resampling is",metric_stats_resampling)
        with open(f"ERROR_Calculate_metric_statistics_{model_parameter}.txt", "a") as col_file:
            col_file.write(f"ERROR: Metric stats resampling not found for {model_parameter} and judge \n")
        with open('metric_stats_resampling_'+str(model_parameter)+'.txt', 'w', encoding='utf-8') as f:
            f.write(f"judge in the except is (and that's why error): {judge_model} \n")
            f.write("try to write metric_stats_resampling as simple string \n")
            f.write(str(metric_stats_resampling))

    assert len(metric_stats_resampling)==len(list_of_metrics), f"Number of metric_stats_resampling not matching num_metrics. \
        Got {len(metric_stats_resampling)} metric_stats_resampling but expected {len(list_of_metrics)}"
    
    for metric in list_of_metrics:
        assert len(metric_stats_resampling[metric]['means']) == num_questions, f"Number of values for metric '{metric}' ({len(metric_stats_resampling[metric]['means'])}) \
            not matching expected number of questions ({num_questions})"

    return metric_stats_resampling

def handle_zero_values(results_df, n_resamples, list_of_metrics, model_name): #Need to be changed for second judge
    """
    Handle zero values in results.
    
    Args:
        results_df (pd.DataFrame): DataFrame containing results
        n_resamples (int): Number of resamples
        list_of_metrics (list): List of metrics to check
        model_name (str): Name of the model being evaluated
        
    Returns:
        dict: Indices of rows containing zero values for each metric
    """
    zero_rows_columns = {}
    
    try:
        # Handle 0 values across all resamples - These are errors
        for resample_idx in range(n_resamples):
            for metric in list_of_metrics:
                try:
                    simple_metric_name = metric.replace('_descr','')
                    metric_col = f'metric_{simple_metric_name}_{resample_idx+1}'
                    
                    # Check if column exists
                    if metric_col not in results_df.columns:
                        print(colored(f"Warning: Column {metric_col} not found in DataFrame", 'yellow'))
                        continue
                    
                    zero_indices = results_df[metric_col] == 0 #series with True/False
                    
                    if zero_indices.any(): #If any of the values of that column are 0
                        zero_rows_columns[metric_col] = []
                        for idx in zero_indices[zero_indices].index: #Loop over True indices (rows with 0s)
                            try:
                                print(colored(f"Missing value for metric '{simple_metric_name}' in resample {resample_idx+1} of model {'_'.join(model_name.split('/')[1:])}", 'red'))
                                print(colored(f"Question: {results_df.loc[idx, 'questions']}", 'green'))
                                zero_rows_columns[metric_col].append(idx) #Keep track of columns and rows with zero values

                                with open(f"Handle_zero_values.txt", "a") as col_file:
                                    # Write all zero value information to file
                                    col_file.write(f"\nMissing value for metric '{simple_metric_name}' in resample {resample_idx+1} of model {'_'.join(model_name.split('/')[1:])}\n")
                                    col_file.write(f"Question: {results_df.loc[idx, 'questions']}\n")

                            except Exception as e:
                                print(colored(f"Unexpected error processing zero value at row {idx} of model {'_'.join(model_name.split('/')[1:])}: {e}", 'red'))
                                with open(f"Handle_zero_values.txt", "a") as col_file:
                                    col_file.write(f"Unexpected error processing zero value at row {idx} of model {'_'.join(model_name.split('/')[1:])}: {e}\n")
                
                except Exception as e:
                    print(colored(f"Error processing metric {metric} in resample {resample_idx} of model {'_'.join(model_name.split('/')[1:])}: {e}", 'red'))
                    with open(f"Handle_zero_values.txt", "a") as col_file:
                        col_file.write(f"Error processing metric {metric} in resample {resample_idx} of model {'_'.join(model_name.split('/')[1:])}: {e}\n")
        
        return zero_rows_columns # Return column names and rows with zero values
    
    except Exception as e:
        print(colored(f"Critical error in handle_zero_values for model {'_'.join(model_name.split('/')[1:])}: {e}", 'red'))
        with open(f"Handle_zero_values.txt", "a") as col_file:
            col_file.write(f"Critical error in handle_zero_values for model {'_'.join(model_name.split('/')[1:])}: {e}\n")
        traceback.print_exc()
        return {}  # Return empty dict in case of critical error

def process_zero_values(results_df, zero_rows_columns, list_of_metrics, model_name): #TO BE ACTIVATED - Need to be changed for second judge
    """Process and optionally replace zero values in results."""
    row_zero_counts = {}
    col_zero_counts = {}

    # Force a copy to ensure changes are applied properly below when replace with mean value
    # results_df_copy = results_df.copy()
    model_parameter = "_".join(model_name.split('/')[1:])

    for column_name, row_indices in zero_rows_columns.items():
        for row_idx in row_indices:
                
            # Get values for this metric for this row and column (one resample per time)
            values = results_df.loc[row_idx, column_name]

            assert values==0, "Values should be 0"
            
            if values != 0: #We should never get here
                with open('values_'+str(model_parameter)+'_'+str(column_name)+'_'+str(row_idx)+'.txt', 'w') as f:
                    f.write(str(values))
                
            #Given that values are 0, replace with mean of non-zero values
            df_values=results_df.loc[:, column_name].values
            non_zero_values = [x for x in df_values if x != 0]

            if len(non_zero_values) > 0:
                mean_value = np.mean(non_zero_values)

                if results_df.loc[row_idx, column_name] == 0 and mean_value != 0:
                    print(colored(f"0 value in row {row_idx}, column {column_name} should be replaced with mean {mean_value:.2f}", 'yellow'))
                    # Uncomment to actually replace values:
                    # results_df.loc[row_idx, column_name] = mean_value#round(mean_value, 1)

                    row_zero_counts[row_idx] = row_zero_counts.get(row_idx, 0) + 1
                    col_zero_counts[column_name] = col_zero_counts.get(column_name, 0) + 1

    print("\nZero values replaced per row:")
    for row in sorted(row_zero_counts):
        print(f"Row/question {row}: {row_zero_counts[row]} replacements")
        with open(f"process_zero_values_{model_parameter}.txt", "a") as col_file:
            col_file.write(f"Row/question {row}: {row_zero_counts[row]} replacements \n")

    print("\nZero values replaced per column:")
    for col in sorted(col_zero_counts):
        print(f"Column/metric {col}: {col_zero_counts[col]} replacements")
        with open(f"process_zero_values_{model_parameter}.txt", "a") as col_file:
            col_file.write(f"Column/metric {col}: {col_zero_counts[col]} replacements \n")

def reorganize_evaluation_metrics(all_resamples_metrics, list_of_metrics, model_name, list_of_questions, n_resamples, judge_model):
    """    
    This function takes evaluation metrics from multiple resampling runs and reorganizes them into
    a structured dictionary where each metric's scores are grouped together. It handles cases where
    some evaluations may have failed (represented by 0s).
    
    Args:
        all_resamples_metrics (list): List of evaluation results for each resample. Each resample contains
                                     scores for multiple questions and metrics.
        list_of_metrics (list): List of metric names to process (e.g., ['completeness_descr', 'relevance_descr']).
        model_name (str): Name of the model being evaluated, used for logging.
        list_of_questions (list): List of questions that were evaluated.
        n_resamples (int): Number of resampling iterations performed.
        judge_model (str): Name of the judge model being used.
    
    Returns:
        dict: Dictionary where keys are metric names (without '_descr' suffix) and values are lists
              containing all scores for that metric across all resamples and questions.
              
    Note:
        The function assumes each resample has scores for all questions and metrics.
    """
    metric_scores_all_resamples = {metric.replace('_descr', ''): [] for metric in list_of_metrics}
    #The above dict will have num_metrics elements, with their value for each question, over each run (first num_questions for run 1, then next num_questions for next run, etc)
    #Example: {'completeness': {'mean':[4, 3, 3, 5, 5, 4, 3, 4, 3, 3, 5, 5, 0, 3, 5, 3, 3, 5, 5, 4, 3]},  #assuming 3 runs and 7 questions
    #          'relevance': {'mean':[4, 3, 3, 3, 4, 3, 2, 4, 3, 3, 3, 3, 0, 2, 4, 3, 3, 3, 3, 3, 2]}, ...}
    #In case of error, there will be num_questions less elements in the sublist for which there was an error
    
    for metric_name in list_of_metrics:
        clean_name = metric_name.replace('_descr', '')
    
        #Each resample_metrics (num_resamples in total) has a list of num_questions lists, each having num_metrics values
        #format of each sublist: [EvaluationResult(key='completeness', score=4, value='To evaluate the ...
        #If error, instead of the above list we have just a 0.
        for resample_idx, resample_metrics in enumerate(all_resamples_metrics):

            # judge_name="_".join(judge_model.split('/')[1:])
            # #Information exists in all_resamples_metrics_main
            # with open('resample_metrics_'+str(resample_idx)+'_'+str(metric_name)+'_'+str(model_name.split('/')[1])+"_with_judge_"+judge_name+'.txt', 'w') as f:
            #     f.write(str(resample_metrics))

            metric_idx = list_of_metrics.index(metric_name) #0-num_metrics the range of values of this. 

            scores = [m[metric_idx].score if m!=0 and m!=[] else 0 
                     for m in resample_metrics] #num_questions elements each time
            assert len(scores)==len(list_of_questions), "Scores length not matching num_questions"

            metric_scores_all_resamples[clean_name].extend(scores) #Every time we add one metric for one resample (num_questions elements)

    assert [len(x) for x in metric_scores_all_resamples.values()]==[len(list_of_questions)*n_resamples]*len(list_of_metrics), "Metric stats length not matching"

    return metric_scores_all_resamples

def reorganize_evaluation_metrics_second_judge(
    excel_path, list_of_metrics, n_resamples, question_col="questions", judge_model_2=judge_model):
    """
    Loads the Excel file and reorganizes metrics for the second judge model.
    Returns a dict: {metric: [all scores for that metric across all resamples/questions]}
    """

    if judge_model_2!='openai/gpt-4o-mini':
        df = pd.read_excel(excel_path)
        metric_scores_all_resamples = {m.replace('_descr', ''): [] for m in list_of_metrics}
        questions = df[question_col].tolist()
        num_questions = len(questions)

        for resample_idx in range(n_resamples):
            for metric_name in list_of_metrics:
                clean_name = metric_name.replace('_descr', '')
                judge_name = "_".join(judge_model_2.split('/')[1:])
                col = f"metric_{clean_name}_{resample_idx+1}_{judge_name}"
                if col not in df.columns:
                    with open(f"warning_{'_'.join(judge_model_2.split('/')[1:])}.txt", "a") as col_file:
                        col_file.write(f"Warning excel: {col} not found in DataFrame \n")
                    print(f"Warning excel: {col} not found in DataFrame")
                    continue
                scores = df[col].tolist()
                initial_scores = scores
                assert len(scores) == num_questions, f"Scores length not matching num_questions for {col}"
                scores=[0 if value is None or (isinstance(value, float) and np.isnan(value)) else value for value in scores]
                final_scores = scores
                metric_scores_all_resamples[clean_name].extend(scores)

                if final_scores!=initial_scores:
                    with open(f"warning_{'_'.join(judge_model_2.split('/')[1:])}.txt", "a") as col_file:
                        col_file.write(f"Initial scores for col {col} are {initial_scores} \n")
                        col_file.write(f"Final scores for col {col} are {scores} \n")

        return metric_scores_all_resamples

def save_results(results_df, judge_model, model_id, save_file=True):
    """Save results DataFrame to Excel."""

    filename = (f"results_{'_'.join(judge_model.split('/')[1:])}_judge_with_"
                    f"{model_id.replace('/','_')}.xlsx")
    try:
        #Extract reasoning traces and final answers from predicted answers
        # Check for <think> tags in predicted answer columns and split them if found
        for col in results_df.columns:
            if 'predicted_answer' in col:
                # Create new column names
                number_pred_col = col.split('_')[2]
                reasoning_col = f'reasoning_trace_{number_pred_col}'
                
                # Check if we need to split this column
                has_think_tags = results_df[col].astype(str).str.contains('</think>', na=False).any()
                has_think_start_tags = results_df[col].astype(str).str.contains('<think>', na=False).any()
                
                if has_think_tags or has_think_start_tags: 
                    print("Has </think> or <think> in answer")
                    with open(f"warning_{'_'.join(judge_model.split('/')[1:])}.txt", "a", encoding='utf-8') as col_file:
                        col_file.write(f"Has </think> or <think> in answer for model {model_id} and judge {judge_model} at column {col} \n")

                    # Extract reasoning traces and final answers
                    reasoning_traces = []
                    final_answers = []
                    
                    for answer in results_df[col]:
                        if isinstance(answer, str):
                            if '<think>' in answer and '</think>' in answer:
                                # Extract the reasoning trace between <think> and </think>
                                think_start = answer.find('<think>') + len('<think>')
                                think_end = answer.find('</think>')
                                
                                if think_start >= 0 and think_end >= 0:
                                    reasoning = answer[think_start:think_end].strip()
                                    final_answer = answer[think_end + len('</think>'):].strip()
                                    reasoning_traces.append(reasoning)
                                    final_answers.append(final_answer)
                                    try:
                                        print("Reasoning:",reasoning)
                                        print("Final answer:",final_answer)
                                        with open(f"warning_{'_'.join(judge_model.split('/')[1:])}.txt", "a", encoding='utf-8') as col_file:
                                            col_file.write(f"Reasoning normal both thinks: {reasoning[:300] + '...' + reasoning[-300:]} \n")
                                            col_file.write(f"Final answer normal both thinks: {final_answer[:300] + '...' + final_answer[-300:]} \n")
                                    except:
                                        with open(f"warning_{'_'.join(judge_model.split('/')[1:])}.txt", "a", encoding='utf-8') as col_file:
                                            col_file.write(f"Error in printing reasoning or final answer for both thinks in answer for model {model_id} and judge {judge_model} \
                                                           in column {col} \n")
                                            col_file.write(f"Reasoning error for both thinks: {reasoning[:300] + '...' + reasoning[-300:]} \n \n")
                                            col_file.write(f"Final answer error for both thinks: {final_answer[:300] + '...' + final_answer[-300:]} \n \n")
                                    
                            elif '</think>' in answer:
                                print("Only </think> in answer")
                                with open(f"warning_{'_'.join(judge_model.split('/')[1:])}.txt", "a", encoding='utf-8') as col_file:
                                    col_file.write(f"Only </think> in answer for model {model_id} and judge {judge_model} \n")
                                # Handle case where only </think> is present
                                think_end = answer.find('</think>')
                                reasoning = answer[:think_end].strip()
                                final_answer = answer[think_end + len('</think>'):].strip()
                                reasoning_traces.append(reasoning)
                                final_answers.append(final_answer)
                                try:
                                    print("Reasoning:",reasoning)
                                    print("Final answer:",final_answer)
                                    with open(f"warning_{'_'.join(judge_model.split('/')[1:])}.txt", "a", encoding='utf-8') as col_file:
                                        col_file.write(f"Reasoning normal only </think>: {reasoning[:300] + '...' + reasoning[-300:]} \n")
                                        col_file.write(f"Final answer normal only </think>: {final_answer[:300] + '...' + final_answer[-300:]} \n")
                                except:
                                    with open(f"warning_{'_'.join(judge_model.split('/')[1:])}.txt", "a", encoding='utf-8') as col_file:
                                        col_file.write(f"Error in printing reasoning or final answer for only </think> in answer for model {model_id} and judge {judge_model} \
                                                       in column {col} \n")
                                        col_file.write(f"Reasoning error for only </think>: {reasoning[:300] + '...' + reasoning[-300:]} \n")
                                        col_file.write(f"Final answer error for only </think>: {final_answer[:300] + '...' + final_answer[-300:]} \n")
                                
                            elif '<think>' in answer:
                                print("Only <think> in answer")
                                with open(f"warning_{'_'.join(judge_model.split('/')[1:])}.txt", "a", encoding='utf-8') as col_file:
                                    col_file.write(f"Only <think> in answer for model {model_id} and judge {judge_model} \n")
                                # Handle case where only <think> is present
                                think_start = answer.find('<think>') + len('<think>')
                                reasoning = answer[think_start:].strip()
                                final_answer = ""  # No final answer if only <think> tag is present
                                reasoning_traces.append(reasoning)
                                final_answers.append(final_answer)
                                try:
                                    print("Reasoning:",reasoning)
                                    print("Final answer: (empty)")
                                    with open(f"warning_{'_'.join(judge_model.split('/')[1:])}.txt", "a", encoding='utf-8') as col_file:
                                        col_file.write(f"Reasoning normal only <think>: {reasoning[:300] + '...' + reasoning[-300:]} \n")
                                        col_file.write(f"Final answer normal only <think>: {final_answer[:300] + '...' + final_answer[-300:]} \n")
                                except:
                                    with open(f"warning_{'_'.join(judge_model.split('/')[1:])}.txt", "a", encoding='utf-8') as col_file:
                                        col_file.write(f"Error in printing reasoning or final answer for only <think> in answer for model {model_id} and judge {judge_model} \
                                                        in column {col} \n")
                                        col_file.write(f"Reasoning error for only <think>: {reasoning[:300] + '...' + reasoning[-300:]} \n")
                                        col_file.write(f"Final answer error for only <think>: {final_answer[:300] + '...' + final_answer[-300:]} \n")
                                
                            else:
                                reasoning_traces.append(' ')
                                final_answers.append(answer)
                                with open(f"warning_{'_'.join(judge_model.split('/')[1:])}.txt", "a", encoding='utf-8') as col_file:
                                    col_file.write(f"No reasoning or </think> or <think> in answer for model {model_id} and judge {judge_model} \n")
                                    col_file.write(f"Type of answer was: {type(answer)} and the value was: {answer} \n")

                        else:
                            with open(f"warning_{'_'.join(judge_model.split('/')[1:])}.txt", "a", encoding='utf-8') as col_file:
                                col_file.write(f"Answer is not a string for model {model_id} and judge {judge_model} in column {col}. It is {type(answer)} and \
                                               the value is {answer} \n")
                            reasoning_traces.append(' ')
                            final_answers.append(' ')

                    #We get here first for first and then for second pred column
                    if any(trace.strip() for trace in reasoning_traces): #If there is any reasoning trace
                        # Add the new columns - insert reasoning column right after the predicted answer column
                        col_idx = results_df.columns.get_loc(col)
                        # Check if reasoning column already exists
                        if reasoning_col not in results_df.columns:
                            results_df.insert(col_idx + 1, reasoning_col, reasoning_traces)
                            results_df[col] = final_answers
                            with open(f"warning_{'_'.join(judge_model.split('/')[1:])}.txt", "a", encoding='utf-8') as col_file:
                                col_file.write(f"Adding reasoning traces and final answers to dataframe for model {model_id} and judge {judge_model} in column {col} \n")
                        else: #We should not get here!
                            with open(f"warning_{'_'.join(judge_model.split('/')[1:])}.txt", "a", encoding='utf-8') as col_file:
                                col_file.write(f"Reasoning column already exists: {reasoning_col} for model {model_id} and judge {judge_model} at index \
                                               {col_idx} with value {reasoning_traces} and final answers {final_answers} \n")
                            pass
                    else:
                        with open(f"warning_{'_'.join(judge_model.split('/')[1:])}.txt", "a", encoding='utf-8') as col_file:
                            col_file.write(f"No reasoning traces in answer for model {model_id} and judge {judge_model} in column {col} \n")
                            col_file.write(f"Reasoning traces: {reasoning_traces} \n")
                            col_file.write(f"Final answers: {final_answers} \n")

    except Exception as e:
        print("Error in saving trace results:", e)
        traceback.print_exc()
    
    if save_file==True:
        results_df.to_excel(filename, index=False)
    else:
        return results_df