import pandas as pd
import numpy as np
from termcolor import colored
import traceback

from src.llm_eval.config import embedding_model, reranker_model_name, n_resamples
from .rag import initialize_vectorstore, initialize_reranker

def process_evaluation_results(input_excel, use_RAG=False): #langsmith_api_key, dataset_langsmith,
    """Extract questions and answers from evaluation results."""
    #https://docs.smith.langchain.com/tutorials/Developers/evaluation

    # Get unique questions/answers
    qa = pd.read_excel(input_excel)
    # print("QA columns are:",qa.columns)
    list_of_questions = qa['input'].tolist()
    # print("List of questions is:",list_of_questions)
    list_of_answers = qa['output'].tolist()

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

def process_metrics(resample_results, list_of_metrics):
    """
    Process metrics for a single resample and update results DataFrame.
    
    Args:
        resample_results: Results from current resample
        list_of_metrics: List of metrics to process
        
    Returns:
        individual_run_metric_scores, evaluation_prompts
    """

    # Initialize individual run metric scores for this resample
    individual_run_metric_scores = {}
    evaluation_prompts = {}
    
    for _, metric_name in enumerate(list_of_metrics):
        individual_run_metric_scores[metric_name] = []
        evaluation_prompts[metric_name] = []
    
    # Process each question's results
    for idx, _ in enumerate(resample_results):
        question_key = f'question_{idx}'
        
        if question_key in resample_results and 'results' in resample_results[question_key]:
            # Extract scores for each metric for this question
            for result_item in resample_results[question_key]['results']:
                metric_key = result_item['key'] + '_descr'
                score = result_item['score']
                
                if metric_key in individual_run_metric_scores:
                    individual_run_metric_scores[metric_key].append(score)
                    evaluation_prompts[metric_key].append(result_item['value'])
                else:
                    print(f"Metric {metric_key} not found in individual_run_metric_scores")
    
        else:
            print(f"Question {idx} not found in resample_results")
         
    return individual_run_metric_scores, evaluation_prompts

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

def handle_zero_values(results_df, n_resamples, continue_from_resample, list_of_metrics, model_name, judge_name): 
    """
    Handle zero values in results.
    
    Args:
        results_df (pd.DataFrame): DataFrame containing results
        n_resamples (int): Number of resamples
        continue_from_resample (int): Index of the resample to continue from
        list_of_metrics (list): List of metrics to check
        model_name (str): Name of the model being evaluated
        judge_name (str): Name of the judge being used
        
    Returns:
        dict: Indices of rows containing zero values for each metric
    """
    zero_rows_columns = {}
    
    try:
        # Handle 0 values across all resamples - These are errors
        for resample_idx in range(continue_from_resample, n_resamples):
            for metric in list_of_metrics:
                try:
                    simple_metric_name = metric.replace('_descr','')
                    # judge_name_main = "_".join(judge_name.split('/')[1:])
                    metric_col = f'metric_{simple_metric_name}_{resample_idx+1}_{judge_name}'
                    
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

def process_zero_values(results_df, zero_rows_columns, model_name): #TO BE ACTIVATED
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

def reorganize_evaluation_metrics(df, list_of_metrics, list_of_questions, n_resamples, judge_model):

    """    
    This function takes evaluation metrics from multiple resampling runs and reorganizes them into
    a structured dictionary where each metric's scores are grouped together. It handles cases where
    some evaluations may have failed (represented by 0s).
    
    Args:
        df (pd.DataFrame): DataFrame containing evaluation metrics
        list_of_metrics (list): List of metric names to process (e.g., ['completeness_descr', 'relevance_descr']).
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
        for resample_idx in range(n_resamples):
            scores = list(df[f'metric_{clean_name}_{resample_idx+1}_{judge_model}'].values)
      
            assert len(scores)==len(list_of_questions), "Scores length not matching num_questions"

            metric_scores_all_resamples[clean_name].extend(scores) #Every time we add one metric for one resample (num_questions elements)

    assert [len(x) for x in metric_scores_all_resamples.values()]==[len(list_of_questions)*n_resamples]*len(list_of_metrics), "Metric stats length not matching"

    return metric_scores_all_resamples

def save_results(results_df, judge_model, model_id, save_file=True):
    """Save results DataFrame to Excel and extract reasoning traces and final answers from predicted answers."""
    print("Judge model to save results for is:",judge_model)
    filename = (f"results_{'_'.join(judge_model.split('/')[1:])}_judge_with_"
                    f"{model_id.replace('/','_')}.xlsx")
    try:
        #Extract reasoning traces and final answers from predicted answers
        # Check for <think> tags in predicted answer columns and split them if found
        for col in results_df.columns:
            if 'predicted_answer_' in col:
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