import json
import os
import time
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import torch
from termcolor import colored
import traceback
import glob
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg') #to avoid Tkinter error

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__),  'env'), override=True) #was /env

from src.llm_eval.config import (
    excel_file_name,
    models,
    judge_model,
    commercial_api_providers,
    generate_max_tokens,
    generation_max_tokens_thinking,
    max_output_tokens,
    domain,
    n_resamples,
    continue_from_resample,
    tool_usage,
    use_RAG,
    use_smolagents,
    openai_api_key,
    langsmith_api_key,
    anthropic_api_key,
    together_api_key,
    open_router_api_key,
    gemini_api_key,
    groq_api_key,
)

from src.llm_eval.core.data_loader import load_data
from src.llm_eval.core.model_utils import get_model
from src.llm_eval.evaluation.evaluator import evaluate_results
from src.llm_eval.evaluation.prompts import list_of_metrics, extract_code_prompt, simulation_prompt, tool_error_prompt, prediction_prompt
from src.llm_eval.utils.processing import (
    process_evaluation_results,
    process_metrics,
    calculate_metric_statistics,
    reorganize_evaluation_metrics,
    save_results,
    handle_zero_values,
    process_zero_values,
)
from src.llm_eval.utils.statistics import (
    load_model_stats,
    perform_power_analysis, 
    aggregate_metrics_by_model, 
    print_aggregated_metrics, 
    calculate_model_metrics,
    compare_model_performances, 
    create_comparison_table
) 

from src.llm_eval.utils.plotting import plot_and_save_model_comparisons, plot_model_comparison, plot_spider_chart, plot_figures_metrics, create_performance_plots
from src.llm_eval.utils.rag import get_similar_qa_pairs, rerank_retrieved_documents, check_context_relevance, format_context
from src.llm_eval.tools.tool_usage import decide_tool_usage
from src.llm_eval.providers.api_handlers import get_model_response
from src.llm_eval.tools.code_execution import handle_code_extraction, text_for_simulation, run_python_script


def predict(inputs: dict, model_name: str, use_RAG: bool = use_RAG, use_smolagents: bool = use_smolagents, tool_usage: bool = tool_usage,
            generate_max_tokens: int = generate_max_tokens, judge_model: str = judge_model[0], generation_max_tokens_thinking: int = generation_max_tokens_thinking,
            extract_code_prompt: str = extract_code_prompt, simulation_prompt: str = simulation_prompt, tool_error_prompt: str = tool_error_prompt,
            prediction_prompt: str = prediction_prompt, openai_api_key: str = openai_api_key, commercial_api_providers: list = commercial_api_providers) -> dict:
    
    """Given a question, return the answer from the model, optionally using tools if tool_usage is True"""
    
    print("Running prediction for model:", model_name)

    # Get these variables from the global scope
    global vectorstore, reranker

    # Configure token limits based on model type - Reasoning model with CoT should have longer max_tokens to include the reasoning steps
    if 'deepseek' in model_name or 'thinking' in model_name or '/o1' in model_name or '/o3' in model_name or \
        'gemini-2.5-pro' in model_name or 'QwQ-32B' in model_name or 'o4' in model_name or 'Qwen3' in model_name:

        generate_max_tokens = generation_max_tokens_thinking #For 'DeepSeek-R1-Distill-Llama-70B-free' limit is 8193
        print("Generation limit increased due to reasoning model:", model_name, "to:", generate_max_tokens)
    else:
        generate_max_tokens = 1000

    # Standard generation arguments
    generation_args = { 
        "max_new_tokens": generate_max_tokens,
        "return_full_text": False, 
        "temperature": 0.05, #Has to be positive number - not considered from model when do_sample is False (reproducible results)
        "do_sample": True, #Selects highest probability token if sets to False
        "num_beams": 5, #3 can also work if computationally intensive - more info on https://huggingface.co/blog/how-to-generate
         #Warnings will be raised by some models

#         #If we only set temp!=0 or if we also set do_sample=False then warning: `do_sample` is set to `False`. However, `temperature` is set to `1e-08` 
#         # -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `temperature`.
#         # That means that the temperature is probably ignored
#         # Sometimes, results not reproducible if only temp is set
#         # A temparature of 0.01 or lower results in: "Error running target function: probability tensor contains either `inf`, `nan` or element < 0"
    }

    # MAIN LOGIC STARTS HERE
    # Get the question from inputs
    question = inputs['question']

    if use_RAG==True:

        # Get similar Q&A pairs
        similar_pairs = get_similar_qa_pairs(
            question,
            vectorstore,
            top_k=10 #Retrieve more candidates for reranking
        )

        # Rerank the retrieved documents
        reranked_pairs = rerank_retrieved_documents(question, similar_pairs, reranker, top_k=5)
        
        # Check if context should be used
        use_context = check_context_relevance(question, reranked_pairs, judge_model, openai_api_key=openai_api_key)
        
        # Prepare context if it should be used
        if use_context:
            context = format_context(reranked_pairs)
            user_content = f"Context:\n{context}\n\n ANSWER THE FOLLOWING QUESTION: {question}"
        else:
            user_content = question
    
    elif use_RAG==False:
        user_content=question
        print("RAG is disabled")

    print("Tool usage is:",tool_usage)
    # If tool_usage is enabled, check if we should use a tool for this question
    if tool_usage:
        model_parameter = "_".join(model_name.split('/')[1:])
        tool_name = decide_tool_usage(inputs['question'])
        print("Tool name:",tool_name)

        if tool_name[0]!='no_tool_needed':
            print(f"Using tool: {tool_name}")
            with open(f"main_log_{model_parameter}.txt", "a", encoding='utf-8') as log:
                log.write("......................................................................\n")
                log.write(f"Using tool: {tool_name}\n")
                log.write(f"Question is: {user_content}\n")
            
            # Start with just the question
            messages = [
                {"role": "user", "content": user_content},
            ]

            # Set the appropriate system message based on tool type
            if tool_name[0] == 'extract_code':
                system_message = extract_code_prompt

            elif tool_name[0] == 'run_simulation':
                system_message = simulation_prompt

            else:
                print("ERROR! With current tools, we shouldn't be here! \n")
                with open(f"main_log_{model_parameter}.txt", "a") as log:
                    log.write(f"ERROR! With current tools, we shouldn't be here!\n")

                system_message = tool_error_prompt
            
            with open(f"main_log_{model_parameter}.txt", "a") as log:
                log.write(f"System message initially was: {system_message}\n\n")

            # Add system message based on model type
            if 'gemma' not in model_name and 'anthropic' not in model_name and 'openrouter' not in model_name and 'gemini' not in model_name and '/o1' not in model_name:
                messages.insert(0, {"role": "system", "content": system_message})
            elif 'gemini' in model_name:
                messages = {"role": "user", "parts": [{"text": system_message + " " + messages[0]['content']}]}
            else:  # For gemma add system prompt in user message
                messages[0]['content'] = system_message + " " + messages[0]['content']
            
            # Get response from API
            response = get_model_response(messages, model_name=model_name, commercial_api_providers=commercial_api_providers,
                                           generation_args=generation_args, generate_max_tokens=generate_max_tokens)

            # Process based on tool type
            if tool_name[0] == 'extract_code': #output saved within the function execution
                if use_smolagents==False:
                    final_answer, output_code = handle_code_extraction(response, model_name, user_question='', use_smolagents=False)
                elif use_smolagents==True:
                    response=''
                    final_answer, output_code = handle_code_extraction(response, model_name, user_question=question, use_smolagents=True)

                if os.path.exists("code_result.py"):
                    os.remove("code_result.py")

                return {"output": final_answer} #we return '-' if didn't work, even if correct code with 'final answer'
            
                # if text_code_evaluation: #this evaluates based on the actual code text/inp simulation file content
                #     return {"output": output_code}
                # else: #this evaluates based on code execution/simulation output
                #     return {"output": final_answer}
            
            elif tool_name[0] == 'run_simulation':
                final_answer, inp_content = text_for_simulation(response, model_name=model_parameter)

                print("Printing responses....")
                print("Detailed output for simulation:", response)
                print("Code output for simulation:", inp_content)
                with open(f"main_log_{model_parameter}.txt", "a", encoding='utf-8') as log:
                    log.write(f"Printing responses....\n")
                    log.write(f"Detailed output for simulation: \n {response}\n \n")
                    log.write(f"Code output for simulation: \n {inp_content}\n \n")
                print("Final answer for simulation:", final_answer, '\n')
                with open(f"inp_final_answer_output_log_{model_parameter}.txt", "a", encoding='utf-8') as log_file:
                    log_file.write(f"Final answer output:\n{final_answer}\n")
                    log_file.write("........... \n \n")

                return {"output": inp_content} #inp_content if we want the text to be fed in the simulation software to be evaluated instead
            
                # if text_code_evaluation: #this evaluates based on the actual code text/inp simulation file content
                #     return {"output": inp_content}
                # else: #this evaluates based on code execution/simulation output
                #     return {"output": final_answer}
            
            else:  # For other tools
                print("ERROR! We shouldn't be here! Returned response from", model_name, ':', response)
                
                with open(f"main_log_{model_parameter}.txt", "a", encoding='utf-8') as log:
                    log.write(f"ERROR! We shouldn't be here! Returned response from {model_name}:\n{response}\n \n")
                    log.write("**********")
                    
                return {"output": response}
            
        else:
            print("No tool will be used")
            with open(f"main_log_{model_parameter}.txt", "a") as log:
                log.write("No tool will be used\n")
                log.write(f"Question is {user_content}\n\n")
                
             # Default case when not using tools - use the original message format
            messages = [
                {"role": "user", "content": user_content},
            ]

            # Add system message based on model type - same as original
            if 'gemma' not in model_name and 'anthropic' not in model_name and 'openrouter' not in model_name and 'gemini' not in model_name and '/o1' not in model_name:
                messages.insert(0, {"role": "system", "content": prediction_prompt})

            elif 'gemini' in model_name:
                messages = {"role": "user", "parts": [{"text": prediction_prompt + messages[0]['content']}]}

            else:  # For gemma add system prompt in user message
                messages[0]['content'] = prediction_prompt + messages[0]['content']
            
            response = get_model_response(messages, model_name=model_name, commercial_api_providers=commercial_api_providers,
                                           generation_args=generation_args, generate_max_tokens=generate_max_tokens)
            
            return {"output": response}
    
    else: # Default case when tool_usage is False - use the original message format
        messages = [
            {"role": "user", "content": user_content},
        ]

        # Add system message based on model type - same as original
        if 'gemma' not in model_name and 'anthropic' not in model_name and 'openrouter' not in model_name and 'gemini' not in model_name and '/o1' not in model_name:
            messages.insert(0, {"role": "system", "content": prediction_prompt})

        elif 'gemini' in model_name:
            messages = {"role": "user", "parts": [{"text": prediction_prompt + messages[0]['content']}]}

        else:  # For gemma add system prompt in user message
            messages[0]['content'] = prediction_prompt + messages[0]['content']
        
        response = get_model_response(messages, model_name=model_name, commercial_api_providers=commercial_api_providers,
                                           generation_args=generation_args, generate_max_tokens=generate_max_tokens)
        
        return {"output": response}
    
def generate_predictions(model_id,  n_resamples, continue_from_resample, excel_file_name, judge_model, use_RAG=False, use_smolagents=False, tool_usage=False):
    """Perform evaluation runs and collect results.""" #judge_model,
    global vectorstore, reranker
    results_df, list_of_questions, vectorstore, reranker = process_evaluation_results(excel_file_name, use_RAG=use_RAG)

    # If continuing from a previous resample, try to load existing results
    if continue_from_resample != 0:
        # Try each judge model from last to first
        for judge in reversed(judge_model):
            # Construct the pattern for existing results files
            pattern = f"results_{'_'.join(judge.split('/')[1:])}_judge_with_{model_id.replace('/','_')}.xlsx"
            existing_files = glob.glob(pattern)
            
            if existing_files:
                # Load the most recently modified file
                latest_file = max(existing_files, key=os.path.getmtime)
                print(f"Loading existing results from {latest_file} using judge {judge}")
                try:
                    existing_df = pd.read_excel(latest_file)
                    # Verify the file has the expected structure
                    if 'questions' in existing_df.columns and 'answers' in existing_df.columns:
                        results_df = existing_df
                        print("Successfully loaded existing results")
                        break  # Exit loop once we find valid results
                    else:
                        print(f"Existing file for judge {judge} does not have expected columns, trying next judge")
                except Exception as e:
                    print(f"Error loading existing results for judge {judge}: {e}")
                    continue  # Try next judge
            else:
                print(f"No existing results found for judge {judge}, trying next judge")
        
        if all(not glob.glob(f"results_{'_'.join(j.split('/')[1:])}_judge_with_{model_id.replace('/','_')}.xlsx") for j in judge_model):
            print("No existing results found for any judge, using fresh DataFrame")

    # print("Vectorstore:",vectorstore)
    # model_name = "_".join(model_id.split('/')[1:])
    # with open('vectorstore_'+str(model_name)+'.txt', 'a') as f:
    #     f.write(str(vectorstore) + "\n")
    #     try:
    #         f.write(f"actual variable {vectorstore()}")
    #     except Exception as e:
    #         f.write(f"Error writing vectorstore to file: {e}")
    #         pass
    
    begin = time.time()

    for resample_idx in range(continue_from_resample, n_resamples):
        print(f"\nPerforming evaluation of resample {resample_idx+1}/{n_resamples} of {model_id}")

        # Create column name for this resample
        column_name = f'predicted_answer_{resample_idx}'
        
        # Initialize list to store predictions for this resample
        predictions_list = []
        
        # Loop over the results_df rows
        for _, row in tqdm(results_df.iterrows(), total=len(results_df), desc=f"Processing predictions for df for resample {resample_idx+1}"):
            question = row['questions']
            answer = row['answers']
            inputs = {'question': question, 'answer': answer}
            predictions = predict(inputs, model_id, use_RAG=use_RAG, use_smolagents=use_smolagents, tool_usage=tool_usage,)
            # print("Predictions:", predictions)
            predictions_list.append(predictions['output'])

        # Assert that the number of predictions matches the number of rows in the DataFrame
        assert len(predictions_list) == len(results_df), f"Mismatch: {len(predictions_list)} predictions for {len(results_df)} rows"
        
        # Add the predictions as a new column to results_df
        if continue_from_resample != 0:
            # Get list of existing predicted_answer columns
            pred_cols = [col for col in results_df.columns if col.startswith('predicted_answer_')]
            
            # Find insertion index - after questions, answers and any existing predicted_answer columns
            if pred_cols:
                # Insert after last predicted_answer column
                insert_idx = results_df.columns.get_loc(pred_cols[-1]) + 1
            else:
                # Insert after questions and answers columns
                insert_idx = 2
                
            # Insert new column at the correct position
            results_df.insert(insert_idx, column_name, predictions_list)
        else:
            # Original implementation
            results_df[column_name] = predictions_list
        # print("Results_df:", results_df)
    
    print(f"Total time for evaluation: {time.time() - begin}")

    return results_df, list_of_questions, vectorstore, reranker

def add_id_and_origin_file_columns(save_dir, excel_file_name):
    """
    Add id and origin_file columns to all results_*.xlsx files in the specified directory.
    
    Args:
        save_dir (str): Directory path containing the results files
        excel_file_name (str): Path to the original Excel file containing the dataset
    """
    # Load original data for matching
    original_data = load_data(excel_file_name)
    
    # Find all xlsx files matching the pattern
    xlsx_files = glob.glob(os.path.join(save_dir, 'results_*.xlsx'))
    
    for xlsx_file in xlsx_files:
        print('Processing xlsx_file:', xlsx_file)
        # Read the results file
        results_df = pd.read_excel(xlsx_file)
        
        if 'questions' in results_df.columns:
            # Create lists to store the new column values
            ids = []
            origin_files = []
            
            # For each question in the results file
            for question in results_df['questions']:
                # Find matching row in dataset_test, handling potential extra newlines and whitespace
                question_cleaned = ' '.join(question.strip().split())
                matching_row = original_data[original_data['input'].apply(lambda x: ' '.join(str(x).strip().split()) == question_cleaned)]
                
                if not matching_row.empty:
                    # Get the first match (in case of duplicates)
                    ids.append(matching_row.iloc[0]['id'])
                    origin_files.append(matching_row.iloc[0]['origin_file'])
                else:
                    # If no match found, append None or empty string
                    ids.append(' ')
                    origin_files.append(' ')
            
            # Insert the new columns at the beginning of the dataframe
            results_df.insert(0, 'origin_file', origin_files)
            results_df.insert(0, 'id', ids)
            
            # Save the updated dataframe back to the xlsx file
            output_file = os.path.join(save_dir, 'final_' + os.path.basename(xlsx_file))
            results_df.to_excel(output_file, index=False)
            print(f"Updated {xlsx_file} with id and origin_file columns, saved as {output_file}")

def main():
    """
    Main execution script - primary workflow.
    """
    start_time = time.time()
    print("Current working directory:", os.getcwd()) #We should be in the data folder
    print("NVIDIA GPU is available:", torch.cuda.is_available())
    torch.random.manual_seed(0) #Set for reproducibility

    test_command, test_process = run_python_script('compare_networks_test.py')
    try:
        print("Command:", test_command)
    except Exception as e:
        print("Error running test_command:", e)
        with open('error_test_command.txt', 'a', encoding='utf-8') as f:
            f.write(f"Error running test_command: {test_command} \n {e}")
    try:
        stdout, stderr = test_process.communicate()
        print("STDOUT:\n", stdout)
        print("STDERR:\n", stderr)
    except Exception as e:
        print("Error running stdout and stderr:", e)
        with open('error_stdout_stderr.txt', 'a', encoding='utf-8') as f:
            f.write(f"Error running stdout: {stdout} \n and stderr: {stderr} \n {e}")

    #Try to load already saved data (if some models have already been evaluated), otherwise initialize empty dicts
    all_models_stats_judge_dicts = {}
    all_runs_model_metrics_judge_dicts = {}
    
    for judge_idx, judge_load in enumerate(judge_model):
            stats_judge, metrics_judge = load_model_stats(judge_load)
            all_models_stats_judge_dicts[f'all_models_stats_judge_{judge_idx+1}'] = stats_judge
            all_runs_model_metrics_judge_dicts[f'all_runs_model_metrics_judge_{judge_idx+1}'] = metrics_judge
        
    # Create individual variables for each judge
    for judge_idx in range(len(judge_model)):
            globals()[f'all_models_stats_judge_{judge_idx+1}'] = all_models_stats_judge_dicts.get(f'all_models_stats_judge_{judge_idx+1}', {})
            globals()[f'all_runs_model_metrics_judge_{judge_idx+1}'] = all_runs_model_metrics_judge_dicts.get(f'all_runs_model_metrics_judge_{judge_idx+1}', {})


    for model_id in models:
        global model_name, model, tokenizer, pipeline, generate_max_tokens, vectorstore
        model_name = model_id #Since model_name defined as global variable
        model_parameter = "_".join(model_name.split('/')[1:])
        model, tokenizer, pipeline = get_model(model_name, commercial_api_providers)
        
        try: #Sometimes some errors with the evaluation 
            print("Generating predictions for model:", model_id)
            print("Excel file name:", excel_file_name)
            print("Judge models are:",judge_model)
            results_df, list_of_questions, vectorstore, reranker = generate_predictions(model_id,  n_resamples, continue_from_resample, 
                                                                                        excel_file_name, judge_model, use_RAG=use_RAG,
                                                                                        use_smolagents=use_smolagents,
                                                                                        tool_usage=tool_usage) 
            
            # Initialize metric scores lists for each judge 
            for judge_idx in range(len(judge_model)): 
                    globals()[f'all_runs_metric_scores_{judge_idx+1}'] = []

            # Process each resample
            for resample_idx in range(continue_from_resample, n_resamples):

                for judge_index,judge_model_name in enumerate(judge_model):
                    print("Looping over judge:", judge_model_name, "and resample:", resample_idx)

                    #This has the predicted answers - not related to judges but has to be in the loop for judge_model
                    results_df=save_results(results_df, judge_model_name, model_id, save_file=False) #For thinking models, we only feed the actual answer to the judge, not the thinking trace

                    results_dict = evaluate_results(results_df, resample_idx, judge_model_name, max_output_tokens, tool_usage) 

                    individual_run_metric_scores, evaluation_prompts = process_metrics(results_dict, list_of_metrics)     

                    for metric_name in list_of_metrics:
                        clean_metric_name = metric_name.replace('_descr', '')
                        results_df[f'metric_{clean_metric_name}_{resample_idx+1}_{judge_model_name}'] = individual_run_metric_scores[metric_name]
                        
                        # For prompts, we'll use empty strings for now since the format doesn't include them
                        results_df[f'prompt_{clean_metric_name}_{resample_idx+1}_{judge_model_name}'] = evaluation_prompts[metric_name]
                    
                    # Handle zero values
                    zero_rows_columns = handle_zero_values(results_df, n_resamples, continue_from_resample, list_of_metrics, model_name, judge_name=judge_model_name)
                    print("Model ID and judge", model_id, judge_model_name)
                    print("Scores from judge", individual_run_metric_scores, 'and resample', resample_idx+1)
                    print("First judge is:",judge_model_name)

                    with open(f"individual_run_metric_scores_{model_name.split('/')[1]}.txt", "a", encoding='utf-8') as col_file: #Also saved in all_runs_metric_scores below
                        col_file.write(f"Model ID and judge: {model_id} and {judge_model_name} \n")
                        col_file.write(f"Scores from judge {individual_run_metric_scores} \n")

                    if zero_rows_columns: #Only keeps tracks of missing values if there are any - NOT ACTIVATED YET
                        unique_zero_rows_columns = len(set([x for sublist in list(zero_rows_columns.values()) for x in sublist]))
                        print(colored(f"ERROR: Found missing values in {unique_zero_rows_columns} rows out of {len(results_df)}", 'red'))
                        with open(f"missing_values_log_{model_parameter}.txt", "a", encoding='utf-8') as col_file:
                            col_file.write(f"ERROR: Found missing values in {unique_zero_rows_columns} rows out of {len(results_df)}. These are the rows: {zero_rows_columns}, \
                                        where the values of dict are the indices of the rows with missing values. Model is {model_name} and judge is {judge_model_name}\n")
                        process_zero_values(results_df, zero_rows_columns, model_name) #Replace 0s with mean of non-zero values    

                    #Has n_resamples lists, each with num_metrics sublists (each sublist has scores over all questions of one metric) 
                    globals()[f'all_runs_metric_scores_{judge_index+1}'].append(individual_run_metric_scores)

                    if continue_from_resample!=0:
                        # Load existing metric scores for resamples before continue_from_resample
                        existing_runs_metric_scores = []
                        
                        for prev_resample_idx in range(continue_from_resample):
                            prev_individual_run_metric_scores = {}
                            
                            for metric_name in list_of_metrics:
                                clean_metric_name = metric_name.replace('_descr', '')
                                metric_col = f'metric_{clean_metric_name}_{prev_resample_idx+1}_{judge_model_name}'
                                
                                if metric_col in results_df.columns:
                                    prev_individual_run_metric_scores[metric_name] = results_df[metric_col].tolist()
                                else:
                                    prev_individual_run_metric_scores[metric_name] = [0] * len(results_df)
                            
                            existing_runs_metric_scores.append(prev_individual_run_metric_scores)
                        
                        # Prepend existing scores to maintain correct order (resample 1, 2, 3, then new ones)
                        globals()[f'all_runs_metric_scores_{judge_index+1}'] = existing_runs_metric_scores + globals()[f'all_runs_metric_scores_{judge_index+1}']
                        print("All runs metric scores for judge", judge_model_name, "are:", globals()[f'all_runs_metric_scores_{judge_index+1}'])
                    
                    # Save initial results
                    print("Saving results for judge:::",judge_model_name)
                    save_results(results_df, judge_model_name, model_id)


            for judge_index,judge_model_name in enumerate(judge_model):
                # Calculate statistics - Only to keep track that everything works - Not used
                metric_stats_resampling = calculate_metric_statistics(
                    globals()[f'all_runs_metric_scores_{judge_index+1}'], 
                    list_of_metrics, 
                    len(list_of_questions),
                    model_name,
                    judge_model_name
                )

            assert len(globals()[f'all_runs_metric_scores_{judge_index+1}'])==n_resamples, f"Number of all_runs_metric_scores not matching num_resamples. \
                Got {len(globals()[f'all_runs_metric_scores_{judge_index+1}'])} all_runs_metric_scores but expected {n_resamples} for judge {judge_model_name}"
            
            for i in range(n_resamples):
                assert len(globals()[f'all_runs_metric_scores_{judge_index+1}'][i])==len(list_of_metrics), f"Number of all_runs_metric_scores[{i}] not matching num_metrics. \
                    Got {len(globals()[f'all_runs_metric_scores_{judge_index+1}'][i])} all_runs_metric_scores[{i}] but expected {len(list_of_metrics)} \
                        for judge {judge_model_name}"


            for judge_index,judge_model_name in enumerate(judge_model):
                # Reorganize metrics - Has num_metrics keys, each with num_questions*num_resamples values (as a list)
                metric_scores_all_resamples = reorganize_evaluation_metrics(results_df, list_of_metrics,  list_of_questions, n_resamples, judge_model_name)

                #A dict with num_metrics keys, each with num_questions*num_resamples values (as a list - first num_questions values are for first resample, 
                # second num_questions values are for second resample, etc.)
                judge_name_main = "_".join(judge_model_name.split('/')[1:])
                with open('metric_scores_all_resamples_'+str(model_parameter)+"_judge_"+str(judge_name_main)+'.txt', 'w', encoding='utf-8') as f:
                    f.write(str(metric_scores_all_resamples))

                assert len(metric_scores_all_resamples)==len(list_of_metrics), f"Number of metric_scores_all_resamples not matching num_metrics. \
                    Got {len(metric_scores_all_resamples)} metric_scores_all_resamples but expected {len(list_of_metrics)}"
                
                for i in range(len(list_of_metrics)):
                    name_of_metric=list_of_metrics[i].replace('_descr','')
                    assert len(metric_scores_all_resamples[name_of_metric])==len(list_of_questions)*n_resamples, f"Number of metric_scores_all_resamples[{name_of_metric}] \
                        not matching num_questions*num_resamples. Got {len(metric_scores_all_resamples[name_of_metric])} metric_scores_all_resamples[{name_of_metric}] but \
                        expected {len(list_of_questions)*n_resamples}"

                metric_names = list(metric_scores_all_resamples.keys()) #Final list of metrics for plotting
                
                # Verify metric names
                metrics_names_loop = [metric.replace('_descr','') for metric in list_of_metrics]
                assert metrics_names_loop == metric_names, "Metric names mismatch"
                
                # Save results
                globals()[f'all_runs_model_metrics_judge_{judge_index+1}'][model_id] = globals()[f'all_runs_metric_scores_{judge_index+1}'] #Used in plotting metrics
                #Dictionary in format {model_id:[{metric_1_run_1:[values], metric_2_run_1:[values], ...}, {metric_1_run_2:[values]....}]

                globals()[f'all_models_stats_judge_{judge_index+1}'][model_id] = plot_figures_metrics(
                    globals()[f'all_runs_model_metrics_judge_{judge_index+1}'],
                    metric_names,
                    model_id,
                    judge_model_name
                ) #Stats like mean, std, etc. per metric and per run over all questions
                
                # Save to files
                judge_name = "_".join(judge_model_name.split('/')[1:])
                with open(f'stats_{judge_name}.json', 'w') as f:
                    json.dump(globals()[f'all_models_stats_judge_{judge_index+1}'], f, indent=4)
                with open(f'all_runs_model_metrics_{judge_name}.json', 'w') as f:
                    json.dump(globals()[f'all_runs_model_metrics_judge_{judge_index+1}'], f, indent=4)

            print("Model",model_id,"saved")
            print("Models saved so far:",list(globals()[f'all_models_stats_judge_{judge_index+1}'].keys()))
                
        except Exception as e:
            print("An error occurred in evaluating model",model_id)
            print("Error Details:", e)
            traceback.print_exc()
        
        finally:
            # Clear VRAM
            del model, tokenizer, pipeline
            torch.cuda.empty_cache()
            print('-'*100)

    end_time = time.time()
    print(f"Total execution time: {(end_time - start_time) / 60:.2f} minutes")

    for judge_index,judge_model_name in enumerate(judge_model):

        aggregated_metrics=aggregate_metrics_by_model(globals()[f'all_runs_model_metrics_judge_{judge_index+1}'])
        print("Judge name for aggregated metrics:",judge_model_name)
        print("Aggregated metrics:",aggregated_metrics)
        print("for this judge, All runs model metrics:",globals()[f'all_runs_model_metrics_judge_{judge_index+1}'])

        print_aggregated_metrics(aggregated_metrics, judge_model_name)

        list_of_metric_names=[name.removesuffix('_descr') for name in list_of_metrics]

        model_names, metric_means, metric_stds=calculate_model_metrics(list_of_metric_names, aggregated_metrics)

        plot_model_comparison(model_names, list_of_metric_names, metric_means, metric_stds, save_prefix="_".join(judge_model_name.split('/')[1:]))
        plot_spider_chart(model_names, list_of_metric_names, metric_means, save_prefix="_".join(judge_model_name.split('/')[1:]))

        comparison_results = compare_model_performances(globals()[f'all_runs_model_metrics_judge_{judge_index+1}'], judge_model_name) 

        # Save results to file
        with open('comparison_result_'+"_".join(judge_model_name.split('/')[1:])+".json", 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            def convert_to_serializable(obj):
                if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                    np.int16, np.int32, np.int64, np.uint8,
                    np.uint16, np.uint32, np.uint64)):
                    return int(obj)
                elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, (np.bool_)):
                    return bool(obj)
                elif isinstance(obj, (np.ndarray,)):
                    return obj.tolist()
                elif obj is None:
                    return None
                return obj
            
            serializable_results = json.loads(
                json.dumps(comparison_results, default=convert_to_serializable)
            )
            json.dump(serializable_results, f, indent=4)

        plot_and_save_model_comparisons(comparison_results, list_of_metrics, "_".join(judge_model_name.split('/')[1:]))

        # Create and print the table
        metrics = [m.replace('_descr', '') for m in list_of_metrics]
        comparison_table = create_comparison_table(comparison_results, metrics)
        print(comparison_table)

        # Save table to file
        with open('comparison_table_'+'_'.join(judge_model_name.split('/')[1:])+'.txt', 'w') as f:
            f.write(comparison_table)

        add_id_and_origin_file_columns('.', excel_file_name)

    create_performance_plots('.', judge_model)
    
    # First, determine required sample size
    required_samples = perform_power_analysis(effect_size=0.1254, alpha=0.05, power=0.8)  #These parameters result in a sample size of 1000
    print(f"Required samples per model for statistical power: {required_samples}")

if __name__ == "__main__":
    main() 