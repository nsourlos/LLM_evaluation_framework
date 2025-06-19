import json
import os
import time
import requests
from langsmith import Client
from langsmith.evaluation import evaluate
from langsmith.utils import LangSmithConnectionError
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import torch
from termcolor import colored
import traceback
import glob

import matplotlib
matplotlib.use('Agg') #to avoid Tkinter error

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__),  'env'), override=True) #was /env

from src.llm_eval.config import (
    excel_file_name,
    models,
    judge_model,
    judge_model_2,
    commercial_api_providers,
    generate_max_tokens,
    generation_max_tokens_thinking,
    domain,
    n_resamples,
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

from src.llm_eval.core.data_loader import load_data, create_dataset_and_load, get_dataset_name 
from src.llm_eval.core.model_utils import get_model
from src.llm_eval.evaluation.evaluator import factor_evaluator, apply_second_judge
from src.llm_eval.evaluation.prompts import list_of_metrics, extract_code_prompt, simulation_prompt, tool_error_prompt, prediction_prompt
from src.llm_eval.utils.processing import (
    process_evaluation_results,
    process_metrics,
    calculate_metric_statistics,
    reorganize_evaluation_metrics,
    save_results,
    process_metrics_second_judge,
    reorganize_evaluation_metrics_second_judge,
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

from src.llm_eval.utils.plotting import (plot_and_save_model_comparisons, plot_model_comparison, plot_spider_chart, plot_figures_metrics)
from src.llm_eval.utils.rag import get_similar_qa_pairs, rerank_retrieved_documents, check_context_relevance, format_context
from src.llm_eval.tools.tool_usage import decide_tool_usage
from src.llm_eval.providers.api_handlers import get_model_response
from src.llm_eval.tools.code_execution import handle_code_extraction, text_for_simulation, run_python_script

def create_langsmith_dataset(dataset_name, example_inputs, langsmith_api_key, domain=domain):

    client = Client(api_key=langsmith_api_key)

    try:
        #Load the dataset if already exists
        for existing_dataset in client.list_datasets():
            if existing_dataset.name==dataset_name:
                dataset_langsmith=existing_dataset
        for x in dataset_langsmith:
            print("Dataset Loaded")
            break

    except: #Otherwise create it
        print("Dataset not found. Creating new dataset")
        # Storing inputs in a dataset lets us run chains and LLMs over a shared set of examples.
        dataset_langsmith = client.create_dataset(dataset_name=dataset_name,
                                                description="Q&A_"+ domain + "_engineering.")

        for input_prompt, output_answer in example_inputs:
            client.create_example(
                inputs={"question": input_prompt.replace('\n', ' ')},
                outputs={"answer": output_answer.replace('\n', ' ')},
                # metadata={"source": "Wikipedia"},
                dataset_id=dataset_langsmith.id,
            )

    return dataset_langsmith

def predict(inputs: dict, use_RAG: bool = use_RAG, use_smolagents: bool = use_smolagents, tool_usage: bool = tool_usage,
            generate_max_tokens: int = generate_max_tokens, judge_model: str = judge_model, generation_max_tokens_thinking: int = generation_max_tokens_thinking,
            extract_code_prompt: str = extract_code_prompt, simulation_prompt: str = simulation_prompt, tool_error_prompt: str = tool_error_prompt,
            prediction_prompt: str = prediction_prompt, openai_api_key: str = openai_api_key, commercial_api_providers: list = commercial_api_providers) -> dict:
    
    """Given a question, return the answer from the model, optionally using tools if tool_usage is True"""
    
    # Get these variables from the global scope
    global model_name
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

    # If tool_usage is enabled, check if we should use a tool for this question
    if tool_usage:
        model_parameter = "_".join(model_name.split('/')[1:])
        tool_name = decide_tool_usage(inputs['question'])

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
    
def perform_evaluation(model_id, judge_model, n_resamples, example_inputs, factor_evaluator, langsmith_api_key, use_RAG=False, use_smolagents=False, tool_usage=False):
    """Perform evaluation runs and collect results."""
    global vectorstore, reranker
    dataset_name = get_dataset_name(model_id, judge_model, use_RAG=use_RAG, use_smolagents=use_smolagents, tool_usage=tool_usage) #How the dataset will be named in Langsmith
    dataset_langsmith = create_langsmith_dataset(dataset_name, example_inputs, langsmith_api_key)
    results_df, list_of_questions, vectorstore, reranker = process_evaluation_results(langsmith_api_key=langsmith_api_key, dataset_langsmith=dataset_langsmith, use_RAG=use_RAG)

    print("Vectorstore:",vectorstore)
    print("List of questions:",list_of_questions)
    model_name = "_".join(model_id.split('/')[1:])
    judge_name = "_".join(judge_model.split('/')[1:])
    with open('vectorstore_'+str(model_name)+'_judge_'+str(judge_name)+'.txt', 'a') as f:
        f.write(str(vectorstore) + "\n")
        try:
            f.write(f"actual variable {vectorstore()}")
        except Exception as e:
            f.write(f"Error writing vectorstore to file: {e}")
            pass

    evaluation_all_resamples = [] #Used below to obtain the unique questions/answers and also the results of each resample
    
    begin = time.time()
    for resample_idx in range(n_resamples):
        print(f"\nPerforming evaluation of resample {resample_idx+1}/{n_resamples} of {model_id}")

        max_retries = 1 #try only once if connection issues
        backoff_factor = 5
        attempt_langsmith = 0
        # while True: #Activate this to retry if connection issues
        try:
            evaluation_results = evaluate(
                predict, #Function that call our LLM and returns its output
                data=dataset_langsmith.name, #Just using dataset_langsmith doesn't work 
                evaluators=[factor_evaluator], #Evaluators to use
                max_concurrency=1, #Run one question through langsmith each time - Other values will give errors in resulting excels
                # metadata={"revision_id": "the version of your pipeline you are testing"},
                experiment_prefix=str(judge_model) + '_judge_with_' + str(model_id) + '_resample_' + str(resample_idx) # A prefix for experiment names to identify them
            )
            
            # Save evaluation results for this run
            run_id = f"run_{resample_idx+1}"
            eval_filename = f"evaluation_result_{run_id}_{model_name}_judge_{judge_name}.txt"

            try:
                # Write the list to file
                print("Eval results for run:", evaluation_results)
                with open(eval_filename, 'w', encoding='utf-8') as f:
                    # Convert evaluation results to string representation
                    eval_str = ""
                    for i, result in enumerate(evaluation_results):
                        eval_str += f"{result}\n"
                    f.write(eval_str)

            except Exception as e:
                print(f"Error saving evaluation results to file: {e}")
                
            # break #Activate this to retry if connection issues
        except (LangSmithConnectionError, requests.exceptions.ConnectionError) as e:
            if attempt_langsmith >= max_retries:
                raise
            wait_time = backoff_factor * (2 ** attempt_langsmith)
            print(f"[Retry {attempt_langsmith+1}/{max_retries}] LangSmith connection failed: {e}. Retrying in {wait_time}s...")
            with open("retry_log.txt", "a") as log:
                log.write(f"[Retry {attempt_langsmith+1}/{max_retries}] LangSmith connection failed: {e}. Retrying in {wait_time}s...")
                log.write("\n **********")
            with open("retry_eval_log.txt", "a", encoding='utf-8') as log:
                try:
                    log.write(f"Evaluation results: \n {evaluation_results}")
                except Exception as e:
                    log.write(f"Unable to write evaluation results to log file: {e}")
                log.write("\n **********")
            time.sleep(wait_time)
            attempt_langsmith += 1
    
        evaluation_all_resamples.extend(evaluation_results) #Used below to get unique questions/answers and to select the predicted answers
        #After the loop, we get a list with n_resamples*num_questions elements, for just one model (and only for main judge)

    with open('evaluation_all_resamples_'+str(model_name)+'_judge_'+str(judge_name)+'.txt', 'w', encoding='utf-8') as f:
        f.write(str(evaluation_all_resamples))

    assert len(evaluation_all_resamples)==n_resamples*len(example_inputs), f"Number of evaluation results not matching num_resamples*num_questions. \
        Got {len(evaluation_all_resamples)} evaluation results but expected {n_resamples*len(example_inputs)}"
    
    print(f"Total time for evaluation: {time.time() - begin}")

    return evaluation_all_resamples, dataset_langsmith, results_df, list_of_questions, vectorstore, reranker

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
    print("Command:", test_command)
    stdout, stderr = test_process.communicate()
    print("STDOUT:\n", stdout)
    print("STDERR:\n", stderr)

    #https://docs.smith.langchain.com/old/evaluation/faq/manage-datasets
    dataset_test = create_dataset_and_load(excel_file_name)
    example_inputs = [(x['input'],x['output']) for x in dataset_test]
    print("Example inputs:", example_inputs, "\n")

    #Try to load already saved data (if some models have already been evaluated), otherwise initialize empty dicts
    all_models_stats, all_runs_model_metrics = load_model_stats(judge_model) 
    all_models_stats_judge2, all_runs_model_metrics_judge2 = load_model_stats(judge_model_2)

    for model_id in models:
        global model_name, model, tokenizer, pipeline, generate_max_tokens, vectorstore
        model_name = model_id #Since model_name defined as global variable
        model_parameter = "_".join(model_name.split('/')[1:])
        model, tokenizer, pipeline = get_model(model_name, commercial_api_providers)
        
        try: #Sometimes some errors with the evaluation
            evaluation_all_resamples, dataset_langsmith, results_df, list_of_questions, vectorstore, reranker = perform_evaluation(model_id, judge_model, n_resamples,
                                                                                                                                example_inputs, factor_evaluator,
                                                                                                                                langsmith_api_key, use_RAG=use_RAG,
                                                                                                                                use_smolagents=use_smolagents,
                                                                                                                                tool_usage=tool_usage)
            chunk_size = len(example_inputs) #Number of questions
            
            all_resamples_metrics = [] #Keep track of all metrics over all resamples and all questions
            #There will be n_resamples lists, each with num_questions sublists (each having num_metrics sublists) (so num_questions*num_metrics elements in those in total)
            #Each question will have 6 metric values like this: [EvaluationResult(key='completeness', score=4, value='To evaluate the ....
            all_runs_metric_scores = [] #This will be appended to the input that plots metrics at the end. 
            #The format of it is [{metric1_descr_run1: [q1_score, q2_score, ...], metric2_descr_run1: [q1_score, q2_score, ...], ...}, 
            #                     {metric1_descr_run2: [q1_score, q2_score, ...], metric2_descr_run2: [q1_score, q2_score, ...], ...},
            #                     ...num_runs]
            
            # Process each resample
            for resample_idx in range(n_resamples):
                start_idx = resample_idx * chunk_size #start index of current resample (chunk size is the number of questions of each resample)
                #Resample_results saved in the process_metrics function
                resample_results = evaluation_all_resamples[start_idx:start_idx + chunk_size] #Get results of a particular resample
                assert len(resample_results)==chunk_size, f"Number of resample results not matching num_questions. Got {len(resample_results)} resample results \
                    but expected {chunk_size}"
                predicted_answers = [x['run'].outputs['output'] for x in resample_results] #None if error
                assert len(predicted_answers)==chunk_size, f"Number of predicted answers not matching num_questions. Got {len(predicted_answers)} predicted answers \
                    but expected {chunk_size}"

                #We check below if there is any none ('') in predicted_answers and if so, we reorder the questions
                with open('predicted_answers_'+str(resample_idx)+'_'+str(model_parameter)+'.txt', 'a', encoding='utf-8') as f:
                    f.write(str(predicted_answers))
                    f.write("\n \n")
                    print("Total num of predicted answers:",len(predicted_answers))
                    f.write(f"Total num of predicted answers: {len(predicted_answers)}")
                    f.write("............ \n \n")

                run_questions=[x['run'].inputs['inputs']['question'] for x in resample_results]
                with open('run_questions_'+str(resample_idx)+'_'+str(model_parameter)+'.txt', 'a', encoding='utf-8') as f:
                    f.write(str(run_questions))
                    f.write("\n \n")
                    print("Total num of run questions:",len(run_questions))
                    f.write(f"Total num of run questions: {len(run_questions)}")
                    f.write("............ \n \n")

                # Get indices to reorder run_questions to match list_of_questions
                reorder_indices = []
                used_indices = set()

                existing_questions = set(run_questions) - {'-'} - {'--'}
                with open('existing_questions_'+str(resample_idx)+'_'+str(model_parameter)+'.txt', 'a', encoding='utf-8') as f:  
                    f.write(str(existing_questions))
                    f.write("\n \n")
                    print("Total num of existing questions:",len(existing_questions))
                    f.write(f"Total num of existing questions: {len(existing_questions)}")
                    f.write("............ \n \n")
                missing_questions = [q for q in list_of_questions if q not in existing_questions]
                with open('missing_questions_'+str(resample_idx)+'_'+str(model_parameter)+'.txt', 'a', encoding='utf-8') as f:
                    f.write(str(missing_questions))
                    f.write("\n \n")
                    print("Total num of missing questions:",len(missing_questions))
                    f.write(f"Total num of missing questions: {len(missing_questions)}")
                    f.write("............ \n \n")
                remaining_indices = [list_of_questions.index(val) for i, val in enumerate(missing_questions)]
                with open('remaining_indices_'+str(resample_idx)+'_'+str(model_parameter)+'.txt', 'a', encoding='utf-8') as f:
                    f.write(str(remaining_indices))
                    f.write("\n \n")
                    print("Total num of remaining indices:",len(remaining_indices))
                    f.write(f"Total num of remaining indices: {len(remaining_indices)}")
                    f.write("............ \n \n")

                for q in run_questions:
                    if q in list_of_questions:
                        idx = list_of_questions.index(q)
                        reorder_indices.append(idx)
                        used_indices.add(idx)
                    else:
                        reorder_indices.append(None)
                        with open('warning_run_questions_reordering_'+str(resample_idx)+'_'+str(model_parameter)+'.txt', 'a', encoding='utf-8') as f:
                            f.write(f"Warning: Question '{q}' not found in list_of_questions \n")

                all_indices = set(range(len(list_of_questions)))
                remaining_indices = list(all_indices - used_indices)
                with open('remaining_indices_'+str(resample_idx)+'_'+str(model_parameter)+'.txt', 'a', encoding='utf-8') as f:
                    f.write(str(remaining_indices))
                    f.write("\n \n")
                    print("Total num of remaining indices:",len(remaining_indices))
                    f.write(f"Total num of remaining indices: {len(remaining_indices)}")
                    f.write("............ \n \n")

                # Replace None with remaining indices
                ri_iter = iter(remaining_indices)
                reorder_indices = [i if i is not None else next(ri_iter) for i in reorder_indices]
                # Append any leftover indices not already used
                reorder_indices += list(ri_iter)

                if reorder_indices!=range(len(list_of_questions)):
                    with open('reorder_indices_'+str(resample_idx)+'_'+str(model_parameter)+'.txt', 'a', encoding='utf-8') as f:
                        f.write(f"Indices were reordered for model {model_parameter} and judge {judge_model} \n")
                        f.write(str(reorder_indices))
                        f.write("\n \n")
                        print("Total num of reorder indices:",len(reorder_indices))
                        f.write(f"Total num of reorder indices: {len(reorder_indices)}")
                        f.write("............ \n \n")

                # If reorder_indices length doesn't match questions length, use sequential indices
                if len(reorder_indices) != len(list_of_questions):
                    print(f"Warning: Reorder indices length ({len(reorder_indices)}) doesn't match questions length ({len(list_of_questions)}). \
                        These indices are {reorder_indices}. Using sequential indices.")
                    reorder_indices = list(range(len(list_of_questions)))
                
                # Reorder run_questions and predicted_answers using the indices
                run_questions = [run_questions[i] for i in reorder_indices]
                predicted_answers2 = [predicted_answers[i] for i in reorder_indices]
                
                if reorder_indices!=range(len(list_of_questions)):
                    # Save reordered questions and answers
                    with open('run_questions_reordered_'+str(resample_idx)+'_'+str(model_parameter)+'.txt', 'a', encoding='utf-8') as f:
                        f.write(str(run_questions))
                        f.write("\n \n")
                        print("Total num of reordered questions",len(run_questions))
                        f.write(f"Total num of reordered questions {len(run_questions)}")
                        f.write("............ \n \n")
                    with open('predicted_answers_reordered_'+str(resample_idx)+'_'+str(model_parameter)+'.txt', 'a', encoding='utf-8') as f:
                        f.write(str(predicted_answers2))
                        f.write("\n \n")
                        print("Total num of reordered answers",len(predicted_answers))
                        f.write(f"Total num of reordered answers {len(predicted_answers2)}")
                        f.write("............ \n \n")

                # Verify the reordering worked
                try:
                    assert len(run_questions) == len(list_of_questions), "Questions reordering failed - orders don't match"
                    predicted_answers = predicted_answers2
                except AssertionError:
                    print("Questions reordering failed - using sequential indices")
                    with open('warning_questions_reordering_'+str(resample_idx)+'_'+str(model_parameter)+'.txt', 'a', encoding='utf-8') as f:
                        f.write(f"Questions reordering failed - using sequential indices. Original indices: {reorder_indices}\n")
                    reorder_indices = list(range(len(list_of_questions)))

                #Add predicted answers to df
                results_df[f'predicted_answer_{resample_idx+1}'] = predicted_answers
                results_df=save_results(results_df, judge_model, model_id, save_file=False)

                individual_run_metric_scores, metrics, results_df = process_metrics(
                        resample_results, 
                        list_of_metrics, 
                        list_of_questions,
                        resample_idx,
                        results_df,
                        model_name,
                        reorder_indices
                    )           
                
                # Handle zero values
                zero_rows_columns = handle_zero_values(results_df, n_resamples, list_of_metrics, model_name)
                print("Model ID and judge", model_id, judge_model)
                print("Scores from first judge", individual_run_metric_scores)

                with open(f"individual_run_metric_scores_{model_name.split('/')[1]}.txt", "a", encoding='utf-8') as col_file: #Also saved in all_runs_metric_scores below
                    col_file.write(f"Model ID and judge: {model_id} and {judge_model} \n")
                    col_file.write(f"Scores from first judge {individual_run_metric_scores} \n")
                    col_file.write(f"Metrics from first judge {metrics} \n")

                if zero_rows_columns: #Only keeps tracks of missing values if there are any - NOT ACTIVATED YET
                    unique_zero_rows_columns = len(set([x for sublist in list(zero_rows_columns.values()) for x in sublist]))
                    print(colored(f"ERROR: Found missing values in {unique_zero_rows_columns} rows out of {len(results_df)}", 'red'))
                    with open(f"missing_values_log_{model_parameter}.txt", "a", encoding='utf-8') as col_file:
                        col_file.write(f"ERROR: Found missing values in {unique_zero_rows_columns} rows out of {len(results_df)}. These are the rows: {zero_rows_columns}, \
                                    where the values of dict are the indices of the rows with missing values. Model is {model_name} and judge is {judge_model}\n")
                    process_zero_values(results_df, zero_rows_columns, list_of_metrics, model_name) #Replace 0s with mean of non-zero values    
                
                #In each iteration we append the metrics (6 in total) of one resample for all questions - n at the end, one for each resample
                #If there is an error, the metrics will be 0 (there will be n_errors*num_metrics less 'EvaluationResult' objects in that case)
                all_resamples_metrics.append(metrics)

                #Has n_resamples lists, each with num_metrics sublists (each sublist has scores over all questions of one metric) 
                all_runs_metric_scores.append(individual_run_metric_scores)
            
            assert len(all_runs_metric_scores)==n_resamples, f"Number of all_runs_metric_scores not matching num_resamples. \
                Got {len(all_runs_metric_scores)} all_runs_metric_scores but expected {n_resamples}"
            
            for i in range(n_resamples):
                assert len(all_runs_metric_scores[i])==len(list_of_metrics), f"Number of all_runs_metric_scores[{i}] not matching num_metrics. \
                    Got {len(all_runs_metric_scores[i])} all_runs_metric_scores[{i}] but expected {len(list_of_metrics)}"

            # #A list with num resamples dicts, each having num metrics keys. Each key has num_questions values. - Information already in all_runs_metric_scores
            # # example: [{'completeness_descr': [5, 5, 5, 3, 1], .....'general_descr': [5, 4, 5, 0, 2]}, {'completeness_descr': [5, 5, 5, 4, 1], .....}]
            # with open('all_runs_metric_scores_main_'+str(model_parameter)+'.txt', 'w', encoding='utf-8') as f:
            #     f.write(str(all_runs_metric_scores))

            #A list with num_resamples sublists, each sublist having num_questions sublists. For each of those num_questions sublists,
            #  each sub-sublist having num_metrics elements, each like the following:
            #[EvaluationResult(key='completeness', score=3, value="To evaluate the completeness...
            #for example, for 2 resamples, there would be 2 sublists, each with num_questions sublists. For each of those num_questions sublists, ...
            with open('all_resamples_metrics_main_'+str(model_parameter)+"_"+str("_".join(judge_model.split('/')[1:]))+'.txt', 'w', encoding='utf-8') as f:
                f.write(str(all_resamples_metrics))

            assert len(all_resamples_metrics)==n_resamples, f"Number of all_resamples_metrics not matching num_resamples. \
                Got {len(all_resamples_metrics)} all_resamples_metrics but expected {n_resamples}"
            
            for i in range(n_resamples): #Each one will have num_questions elements, each with num_metrics sublists (or 0 if error)
                assert len(all_resamples_metrics[i])==len(list_of_questions), f"Number of all_resamples_metrics[{i}] not matching num_questions. \
                    Got {len(all_resamples_metrics[i])} all_resamples_metrics[{i}] but expected {len(list_of_questions)}" 
                    #Each all_ressamples_metrics[i] should have num_questions elements

            # Calculate statistics - Only to keep track that everything works - Not used
            metric_stats_resampling = calculate_metric_statistics(
                all_runs_metric_scores, 
                list_of_metrics, 
                len(list_of_questions),
                model_name,
                judge_model
            )
            
            # Save initial results
            save_results(results_df, judge_model, model_id)


        #Second judge - Order of indices should be the same as for main judge
            if judge_model_2:
                judge_name = "_".join(judge_model_2.split('/')[1:])
                with open(f"non_existing_cols_{judge_name}.txt", "a", encoding='utf-8') as f:
                    f.write(f"Model used: {model_id}\n \n")

                filename_excel = (f"results_{'_'.join(judge_model.split('/')[1:])}_judge_with_"
                        f"{model_id.replace('/','_')}.xlsx") 
                
                apply_second_judge(
                    input_excel=filename_excel,
                    list_of_metrics=list_of_metrics,  # e.g. ['completeness_descr', ...]
                    num_resamples=n_resamples,  
                    model_name=model_name,
                    judge_model_2=judge_model_2 if judge_model_2 else judge_model,
                )

                excel_path=(f"results_{'_'.join(judge_model_2.split('/')[1:])}_judge_with_"
                        f"{model_id.replace('/','_')}.xlsx")

                #A list with num of judges dicts, each with num of metrics keys and num of questions scores/prompts
                all_runs_metric_scores_second_judge, all_run_metric_prompts_second_judge = process_metrics_second_judge( #all_run_metric_prompts_second_judge is not used
                    excel_path, list_of_metrics, n_resamples, model_name=model_name, judge_model_2=judge_model_2 if judge_model_2 else judge_model)
                
                with open(f'all_runs_metric_scores_second_judge_{judge_name}.txt', 'w', encoding='utf-8') as f:
                    f.write(str(all_runs_metric_scores_second_judge))
                    f.write("\n \n")
                    initial_all_runs_metric_scores_second_judge = all_runs_metric_scores_second_judge

                for resample in range(n_resamples):
                    for metric in all_runs_metric_scores_second_judge[resample]:
                            all_runs_metric_scores_second_judge[resample][metric] = [0 if value is None or (isinstance(value, float) and np.isnan(value)) else value \
                                                                                    for value in all_runs_metric_scores_second_judge[resample][metric]]

                if initial_all_runs_metric_scores_second_judge != all_runs_metric_scores_second_judge:
                    with open(f"all_runs_scores_second_judge_changed_{model_parameter}.txt", "a", encoding='utf-8') as log_file:
                        log_file.write(f"changed_metrics:\n{all_runs_metric_scores_second_judge}\n\n")

                calculate_metric_statistics(
                            all_runs_metric_scores_second_judge, 
                            list_of_metrics, 
                            len(list_of_questions),
                            model_name,
                            judge_name
                        )

                with open(f"warning_{judge_name}.txt", "a", encoding='utf-8') as col_file: #For main questions this just notes model!
                    col_file.write(f"\n \n Model used: {model_id} \n")
                
                # Get reorganized metrics
                metric_scores_all_resamples_second_judge = reorganize_evaluation_metrics_second_judge(
                    excel_path, list_of_metrics, n_resamples, judge_model_2=judge_model_2 if judge_model_2 else judge_model
                )
                
                #A dict with num_metrics keys, each with num_questions*num_resamples values (as a list - first num_questions values are for first resample, 
                # second num_questions values are for second resample, etc.)
                with open('metric_scores_all_resamples_'+str(model_parameter)+'_judge_'+str(judge_name)+'.txt', 'w', encoding='utf-8') as f:
                    f.write(str(metric_scores_all_resamples_second_judge))

                assert len(metric_scores_all_resamples_second_judge)==len(list_of_metrics), f"Number of metric_scores_all_resamples_second_judge not matching num_metrics. \
                    Got {len(metric_scores_all_resamples_second_judge)} metric_scores_all_resamples_second_judge but expected {len(list_of_metrics)}"
                
                for i in range(len(list_of_metrics)):
                    name_of_metric=list_of_metrics[i].replace('_descr','')
                    assert len(metric_scores_all_resamples_second_judge[name_of_metric])==len(list_of_questions)*n_resamples, \
                        f"Number of metric_scores_all_resamples_second_judge[{name_of_metric}] not matching \
                        num_questions*num_resamples. Got {len(metric_scores_all_resamples_second_judge[name_of_metric])} \
                        metric_scores_all_resamples_second_judge[{name_of_metric}] but expected {len(list_of_questions)*n_resamples}"

                metric_names_second_judge = list(metric_scores_all_resamples_second_judge.keys()) #Final list of metrics for plotting
                # Verify metric names
                metrics_names_loop_second_judge = [metric.replace('_descr','') for metric in list_of_metrics]
                assert metrics_names_loop_second_judge == metric_names_second_judge, "Metric names mismatch"
                
                # Save results
                all_runs_model_metrics_judge2[model_id] = all_runs_metric_scores_second_judge #Used in plotting metrics
                #Dictionary in format {model_id:[{metric_1_run_1:[values], metric_2_run_1:[values], ...}, {metric_1_run_2:[values]....}]

                all_models_stats_judge2[model_id] = plot_figures_metrics(
                    all_runs_model_metrics_judge2,
                    metric_names_second_judge,
                    model_id,
                    judge_model_2
                ) #Stats like mean, std, etc. per metric and per run over all questions
                
                # Save to files
                with open(f'stats_{judge_name}.json', 'w') as f:
                    json.dump(all_models_stats_judge2, f, indent=4)
                with open(f'all_runs_model_metrics_{judge_name}.json', 'w') as f:
                    json.dump(all_runs_model_metrics_judge2, f, indent=4)

                print("Model",model_id,"saved with judge", judge_model_2)
                print("Models saved so far:",list(all_models_stats_judge2.keys()))


        #Continue with main judge below
            # Reorganize metrics - Has num_metrics keys, each with num_questions*num_resamples values (as a list)
            metric_scores_all_resamples = reorganize_evaluation_metrics(all_resamples_metrics, list_of_metrics, model_name, list_of_questions, n_resamples, judge_model)

            #A dict with num_metrics keys, each with num_questions*num_resamples values (as a list - first num_questions values are for first resample, 
            # second num_questions values are for second resample, etc.)
            judge_name_main = "_".join(judge_model.split('/')[1:])
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
            all_runs_model_metrics[model_id] = all_runs_metric_scores #Used in plotting metrics
            #Dictionary in format {model_id:[{metric_1_run_1:[values], metric_2_run_1:[values], ...}, {metric_1_run_2:[values]....}]

            all_models_stats[model_id] = plot_figures_metrics(
                all_runs_model_metrics,
                metric_names,
                model_id,
                judge_model
            ) #Stats like mean, std, etc. per metric and per run over all questions
            
            # Save to files
            judge_name = "_".join(judge_model.split('/')[1:])
            with open(f'stats_{judge_name}.json', 'w') as f:
                json.dump(all_models_stats, f, indent=4)
            with open(f'all_runs_model_metrics_{judge_name}.json', 'w') as f:
                json.dump(all_runs_model_metrics, f, indent=4)

            print("Model",model_id,"saved")
            print("Models saved so far:",list(all_models_stats.keys()))
                
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

    aggregated_metrics=aggregate_metrics_by_model(all_runs_model_metrics)
    print(aggregated_metrics)
    aggregated_metrics_by_model_second_judge=aggregate_metrics_by_model(all_runs_model_metrics_judge2)
    print(aggregated_metrics_by_model_second_judge)

    print_aggregated_metrics(aggregated_metrics, judge_model)
    print_aggregated_metrics(aggregated_metrics_by_model_second_judge, judge_model_2)

    list_of_metric_names=[name.removesuffix('_descr') for name in list_of_metrics]

    model_names, metric_means, metric_stds=calculate_model_metrics(list_of_metric_names, aggregated_metrics)
    model_names_second_judge, metric_means_second_judge, metric_stds_second_judge=calculate_model_metrics(list_of_metric_names, aggregated_metrics_by_model_second_judge)

    plot_model_comparison(model_names, list_of_metric_names, metric_means, metric_stds, save_prefix="_".join(judge_model.split('/')[1:]))
    plot_model_comparison(model_names_second_judge, list_of_metric_names, metric_means_second_judge, metric_stds_second_judge, save_prefix="_".join(judge_model_2.split('/')[1:]))
    plot_spider_chart(model_names, list_of_metric_names, metric_means, save_prefix="_".join(judge_model.split('/')[1:]))
    plot_spider_chart(model_names_second_judge, list_of_metric_names, metric_means_second_judge, save_prefix="_".join(judge_model_2.split('/')[1:]))

    comparison_results = compare_model_performances(all_runs_model_metrics, judge_model) 
    # Save results to file
    with open('comparison_result_'+"_".join(judge_model.split('/')[1:])+".json", 'w') as f:
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

    comparison_results_second_judge = compare_model_performances(all_runs_model_metrics_judge2, judge_model_2)

    # Save results to file
    with open('comparison_result_'+'_'.join(judge_model_2.split('/')[1:])+'.json', 'w') as f:
        serializable_results_second_judge = json.loads(
            json.dumps(comparison_results_second_judge, default=convert_to_serializable)
        )
        json.dump(serializable_results_second_judge, f, indent=4)

    plot_and_save_model_comparisons(comparison_results, list_of_metrics, "_".join(judge_model.split('/')[1:]))
    plot_and_save_model_comparisons(comparison_results_second_judge, list_of_metrics, "_".join(judge_model_2.split('/')[1:]))

    # Create and print the table
    metrics = [m.replace('_descr', '') for m in list_of_metrics]
    comparison_table = create_comparison_table(comparison_results, metrics)
    print(comparison_table)

    # Save table to file
    with open('comparison_table_'+'_'.join(judge_model.split('/')[1:])+'.txt', 'w') as f:
        f.write(comparison_table)

    comparison_table_second_judge = create_comparison_table(comparison_results_second_judge, metrics)
    print(comparison_table_second_judge)

    # Save table to file
    with open('comparison_table_'+'_'.join(judge_model_2.split('/')[1:])+'.txt', 'w') as f:
        f.write(comparison_table_second_judge)

    # First, determine required sample size
    required_samples = perform_power_analysis(effect_size=0.1254, alpha=0.05, power=0.8)  #These parameters result in a sample size of 1000
    print(f"Required samples per model for statistical power: {required_samples}")

    add_id_and_origin_file_columns('.', excel_file_name)

if __name__ == "__main__":
    main() 