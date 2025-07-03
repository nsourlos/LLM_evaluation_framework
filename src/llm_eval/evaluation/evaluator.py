"""
Main evaluation function
"""
import re
import json
import ast
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from tqdm import tqdm
from termcolor import colored

from ..config import (
    max_output_tokens, judge_model, tool_usage,
    openai_api_key, gemini_api_key, together_api_key
)
from ..evaluation.prompts import (common_prompt, list_of_metrics, 
                                  completeness_descr, relevance_descr, conciseness_descr, confidence_descr, factuality_descr, judgement_descr, general_descr) #used inside factor_evaluator
from ..utils.scoring import verify_numerical_outputs

# https://docs.smith.langchain.com/old/cookbook/introduction
# https://docs.smith.langchain.com/old/evaluation/faq/custom-evaluators
# https://docs.smith.langchain.com/how_to_guides/evaluation/evaluate_llm_application#use-a-summary-evaluator
#Function that compares the real answer with the predicted answer of an LLM and returns a score based on the evaluation
def evaluate_results(input_df, resample_idx, judge_model=judge_model[0],  max_output_tokens=max_output_tokens, tool_usage=tool_usage) -> dict: 

    #Remove thinking tags if they exist
    def remove_thinking_tags(s):
        if '</think>' in s: #Check for reasoning traces and only keep the part after the trace
            think_end = s.find('</think>')
            if s[think_end + len('</think>'):].strip():
                s = s[think_end + len('</think>'):].strip()
        return s

    def extract_json_dict(s): #extract a dictionary from a string
        if not isinstance(s, str):
            return None
        s = remove_thinking_tags(s)

        # Now try various methods to extract the dictionary
        # Method 1: Direct JSON parsing of the whole string
        try:
            return json.loads(s)
        except (json.JSONDecodeError, TypeError):
            pass

        # Method 2: JSON parsing after stripping common wrapping characters like backticks and newlines
        try:
            s_clean = s.strip('` \\n')
            return json.loads(s_clean)
        except (json.JSONDecodeError, TypeError):
            pass

        # Method 3: Python literal evaluation of the whole string
        try:
            result = ast.literal_eval(s)
            if isinstance(result, dict):
                return result
        except (ValueError, SyntaxError, TypeError): # Added TypeError for robustness
            pass

        # Method 4: Regex for ```json ... ``` code blocks
        match = re.search(r'```json\s*({.*?})\s*```', s, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except (json.JSONDecodeError, TypeError): # Added TypeError for robustness
                pass

        # Method 5: Regex for `json ... ` (inline markdown) code blocks
        match = re.search(r'`json\s*({.*?})\s*`', s, re.DOTALL) 
        if match:
            try:
                return json.loads(match.group(1))
            except (json.JSONDecodeError, TypeError): # Added TypeError for robustness
                pass

        # Method 6: Find a dictionary at the very end of the string.
        # This handles cases like "Some explanatory text... {json_dict}"
        # or "Some explanatory text... {'python_dict_literal': True}"
        
        # Remove trailing whitespace from the string to check its actual end.
        s_trimmed_for_ending_check = s.rstrip()

        if s_trimmed_for_ending_check.endswith('}'):
            # Find the last opening brace '{'. This is crucial for "last occurrence".
            last_open_brace_index = s_trimmed_for_ending_check.rfind('{')

            if last_open_brace_index != -1:
                # Candidate dictionary string is from the last '{' to the end of the trimmed string.
                dict_candidate_str = s_trimmed_for_ending_check[last_open_brace_index:]
                
                # Attempt to parse this candidate string.
                try:
                    # Try json.loads first (for strict JSON).
                    parsed_dict = json.loads(dict_candidate_str)
                    return parsed_dict
                except (json.JSONDecodeError, TypeError):
                    try:
                        # If json.loads fails, try ast.literal_eval (for Python dict literals).
                        # ast.literal_eval can parse other literals (lists, numbers, etc.),
                        # so we must check if the result is actually a dictionary.
                        parsed_literal = ast.literal_eval(dict_candidate_str)
                        if isinstance(parsed_literal, dict):
                            return parsed_literal
                    except (ValueError, SyntaxError, TypeError):
                        # If both parsing methods fail for this candidate,
                        # it means the substring from the last '{' to the end is not a valid dict.
                        # We fall through and the function will eventually return None.
                        pass
        
        return None


    def get_judge_scores(questions, answers, predicted_answers, judge_model, list_of_metrics, max_output_tokens, tool_usage):

        results_dict={}

        for idx, (question, actual_answer, predicted_answer) in tqdm(enumerate(zip(questions, answers, predicted_answers)), total=len(questions), desc="Evaluating questions"):
            # print("Question is:",question)
            # print("Actual answer is:",actual_answer)
            # print("Predicted answer is:",predicted_answer)
            print("Judge model used to evaluate answers:",judge_model, '\n')
        
            #Initialize dictionaries to store scores and descriptions for each metric
            scores = {}
            descriptions = {}

            # Check if there is output from LLM
            if not predicted_answer:
                print("No output from LLM")
                keys = []
                for metric_name in list_of_metrics: #Some models might not give answers for some questions (e.g. o1 sometimes)
                    keys.append(metric_name.split('_descr')[0])
                    scores[metric_name] = 0
                    descriptions[metric_name] = '-'
                results_dict['question_'+str(idx)] = {
                    "results": [{"key": key, "score": scores[key + "_descr"], "value": descriptions[key + "_descr"]} for key in keys]}
            
            else:
                keys = []
                for metric_name in list_of_metrics: #Iterate through all metrics
                    keys.append(metric_name.split('_descr')[0])

                    print("Evaluating based on:", metric_name)
                    if 'general' in metric_name:
                        if len(eval(metric_name)) > 2000: #If the prompt is too long, we have the new one
                            metric_value = eval(metric_name)
                        else:
                            metric_value = common_prompt + eval(metric_name)
                    else:
                        metric_value = common_prompt + eval(metric_name) #Get the actual description of the metric

                    # Define roles and placeholders
                    chat_template = ChatPromptTemplate.from_messages(
                        [("system", metric_value),
                        ("user", "Question: {question}, Actual answer: {actual_answer}, Predicted answer: {predicted_answer}")]
                        # ("ai", "It's sunny and warm outside."), #Use this if we want to use few shot prompts
                    )

                    if '</think>' in predicted_answer: #For thinking models, we need to remove the thinking tags for the judge to avoid run out of tokens errors
                        predicted_answer = remove_thinking_tags(predicted_answer)

                    messages = chat_template.format_messages(question=question, actual_answer=actual_answer, predicted_answer=predicted_answer)
                    # print("Messages:",messages)

                    formatted_messages = [(role, msg.content) for role, msg in zip(["system", "user"], messages)]
                    print("Formatted messages:",formatted_messages) #[('system', 'You are an autoregressive lan....', 'user':.....)]

                    # Initialize the model and get response
                    if 'openai' in judge_model:
                        llm = ChatOpenAI(
                            model_name="_".join(judge_model.split('/')[1:]),
                            api_key=openai_api_key,
                            temperature=0,
                            max_tokens=max_output_tokens,
                            seed=42
                        )
                        ai_response = llm.invoke(formatted_messages)
                    elif 'gemini' in judge_model:
                        llm = ChatGoogleGenerativeAI(
                            model="_".join(judge_model.split('/')[1:]), # e.g., 'gemini-2-flash'
                            google_api_key=gemini_api_key,
                            temperature=0,
                            max_output_tokens=max_output_tokens,
                        )
                        ai_response = llm.invoke(messages)
                    elif 'together' in judge_model:
                        llm = ChatOpenAI(
                            base_url="https://api.together.xyz/v1", #Together's API endpoint
                            model="/".join(judge_model.split("/")[1:]), # e.g., 'llama-3-70b-chat'
                            api_key=together_api_key,
                            temperature=0,
                            max_tokens=max_output_tokens,
                        )
                        ai_response = llm.invoke(formatted_messages)

                    # Output
                    # print(colored("System message:"+ messages[0].content,'blue'))
                    # print(colored("User message:"+ messages[1].content, 'green'))
                    # print(colored("AI message:"+ ai_response.content,'red'))

                    # Log the messages
                    try:
                        judge_log_name = "_".join(judge_model.split('/')[1:])
                        with open(f"user_ai_messages_{judge_log_name}.txt", "a", encoding="utf-8") as log:
                            log.write(f"User message: \n{messages[1].content} \n \n")
                            log.write(f"AI message: \n{ai_response.content} \n \n")
                    except:
                        print("Unable to obtain short judge name")
                        with open(f"user_ai_messages.txt", "a") as log:
                            log.write(f"Unable to obtain short judge name or error in writing responses with judge {judge_model}")

                    #Decide what the final score is based on output
                    if "FINAL SCORE:" in ai_response.content:
                        try:
                            score = int(ai_response.content.split("FINAL SCORE:")[1])
                        except: #If more text after it due to e.g. thinking => 'FINAL SCORE:5 is not appropriate...' - If fix this by matching the last one, other errors might occur
                            with open(f"final_score_log.txt", "a", encoding="utf-8") as log:
                                log.write(f"More text after FINAL SCORE, possibly due to thinking.\n")
                                log.write(f"{ai_response.content}\n\n")
                                log.write(f"--------------------------------\n\n")
                            try:
                                # Find the last occurrence of "FINAL SCORE:" and extract the number after it
                                score = int(ai_response.content.split("FINAL SCORE:")[-1])
                            except:
                                with open(f"final_score_log.txt", "a", encoding="utf-8") as log:
                                    log.write(f"Error extracting score from last occurrence. \n")
                                try: # Try to extract score using regex pattern matching for "FINAL SCORE: X" format - might not the first occurence and in between thinking process
                                    score_match = re.search(r'FINAL SCORE:\s*(\d+)', ai_response.content)
                                    if score_match:
                                        score = int(score_match.group(1))
                                        with open(f"final_score_log.txt", "a") as log:
                                            log.write(f"Managed to obtain final score with 're'. Score used {score} \n \n")
                                            log.write(f"******************\n\n")
                                        # continue
                                    else:
                                        with open(f"final_score_log.txt", "a", encoding="utf-8") as log:
                                            log.write(f"Error extracting score from last occurrence. No score match found. Response was: \n {ai_response.content} \n \n")
                                            log.write(f"******************\n\n")
                                        score=0
                                except:
                                    print("Error extracting score from last occurrence. Set it to 0.")
                                    with open(f"final_score_log.txt", "a", encoding="utf-8") as log:
                                        log.write(f"Error extracting score from last occurrence. Set it to 0. Response was: \n {ai_response.content} \n \n")
                                        log.write(f"******************\n\n")
                                    score = 0
                    else:
                        print("Invalid response from LLM:", ai_response.content)
                        try:
                            with open(f"invalid_responses_{judge_log_name}.txt", 'a', encoding="utf-8") as log:
                                log.write(f"Invalid response from LLM: {ai_response.content} \n \n")
                        except:
                            with open("invalid_responses.txt", 'a', encoding="utf-8") as log:
                                log.write(f"Invalid response from LLM: {ai_response.content} \n \n")
                        score = 0 #For cases where the LLM doesn't return a score - Otherwise we are gonna get an error

                    try:
                        # Get judge model name for logging
                        judge_log_name = "_".join(judge_model.split('/')[1:])

                        predicted_dict = extract_json_dict(predicted_answer.strip())
                        actual_dict = extract_json_dict(actual_answer)

                        try: #For network-related questions
                            # Log the raw answers
                            with open(f"test_score_log_{judge_log_name}.txt", "a", encoding="utf-8") as log:
                                log.write(f"Predicted answer: {predicted_answer.strip()} \n")
                                log.write(f"Actual answer: {actual_answer} \n")
                                if predicted_dict is not None:
                                    log.write(f"Predicted dictionary: {predicted_dict} \n")
                                if actual_dict is not None:
                                    log.write(f"Actual dictionary: {actual_dict} \n")
                                log.write(f"--------------------------------\n\n")

                            # Verify numerical outputs if we successfully extracted dictionaries
                            if tool_usage==True:
                                if predicted_dict and actual_dict:
                                    test_score = verify_numerical_outputs(actual_dict, predicted_dict, tol=0.01)
                                    with open(f"test_score_log_{judge_log_name}.txt", "a", encoding="utf-8") as log:
                                        log.write(f"Score set from {score} to {test_score} for predicted answer {predicted_answer.strip()} \n \n")  
                                    score = test_score

                        except Exception as e:
                            print("Error verifying numerical outputs:", str(e))
                            with open(f"test_score_log_{judge_log_name}.txt", "a", encoding="utf-8") as log:
                                log.write(f"Error verifying numerical outputs: {str(e)}\n")
                                log.write(f"{predicted_answer}\n\n")
                                log.write(f"--------------------------------\n\n")
                    except:
                        with open(f"test_score_unchanged_log_{judge_log_name}.txt", "a", encoding="utf-8") as log:
                            log.write(f"Predicted answer: {predicted_answer.strip()} \n")
                            log.write(f"Actual answer: {actual_answer} \n")
                            log.write(f"--------------------------------\n\n")

                    print('metric_name is:',metric_name)
                    scores[metric_name] = score
                    descriptions[metric_name] = ai_response.content
                    print("Scores:",scores)
                    print("\n")

                results_dict['question_'+str(idx)] = {
                    "results": [{"key": key, "score": scores[key + "_descr"], "value": descriptions[key + "_descr"]} for key in keys]}

        return results_dict

   
    questions=input_df['questions'].tolist()
    answers=input_df['answers'].tolist()
    predicted_answers=input_df[f'predicted_answer_{resample_idx}'].tolist()
    results_dict = get_judge_scores(questions, answers, predicted_answers, judge_model, list_of_metrics, max_output_tokens, tool_usage)

    return results_dict