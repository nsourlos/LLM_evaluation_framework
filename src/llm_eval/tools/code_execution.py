"""
Code execution handlers
"""
import os
import subprocess
import platform

#Smolagents related import
from e2b_code_interpreter import Sandbox
import sys

from ..config import (
    generate_max_tokens, use_smolagents, excel_file_name,
    openai_api_key, gemini_api_key, together_api_key, groq_api_key, anthropic_api_key, open_router_api_key, hf_token, e2b_api_key
)

from ..utils.paths import get_file_paths

base_path, file_path, custom_cache_dir, venv_path, venv_name = get_file_paths(excel_file_name, base_path=None)

def run_python_script(python_file, base_path=base_path, venv_path=venv_path, venv_name=venv_name):
    script_path = os.path.join(base_path, python_file)

    if platform.system() == "Windows":
        cmd = [venv_path, "run", "-n", venv_name, "python", script_path]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return ' '.join(cmd), process

    elif platform.system() == "Darwin":
        bash_command = f"source {venv_path} && conda activate {venv_name} && MPLBACKEND=Agg python {script_path}"
        process = subprocess.Popen(['bash', '-c', bash_command], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return bash_command, process

    elif platform.system() == "Linux":
        bash_command = f"source {venv_path} && MPLBACKEND=Agg python {script_path}"
        process = subprocess.Popen(['bash', '-c', bash_command], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return bash_command, process

def handle_code_extraction(code_result, model_name, user_question, 
                           use_smolagents=use_smolagents, judge_model='openai/gpt-4o-mini', generate_max_tokens=generate_max_tokens, openai_api_key=openai_api_key):
    """ 
    Handle code extraction and execution.

     Args:
            code_result (str): The code result to be processed
            model_name (str): The name of the model to be used
            user_question (str): The user question to be processed
            use_smolagents (bool): Whether to use SmolAgents for code execution
            judge_model (str): The name of the judge model to be used
            generate_max_tokens (int): The maximum number of tokens to be generated
            openai_api_key (str): The API key for the OpenAI model

        Returns:
            tuple: (final_answer, code_result)
            final_answer (str): The final answer to be returned
            code_result (str): The code result to be returned
    """

    if use_smolagents==True:
        print("Input question:", user_question)

        # Create the sandbox
        sandbox = Sandbox(api_key=e2b_api_key)

        # Install required packages
        sandbox.commands.run("pip install smolagents")
        sandbox.commands.run("pip install 'smolagents[openai]'") #to use openai model

        def run_code_raise_errors(sandbox, code: str, verbose: bool = False) -> str:
            execution = sandbox.run_code(
                code,
                envs={'HF_TOKEN': hf_token,
                    'OPENAI_API_KEY': openai_api_key,
                    'GEMINI_API_KEY': gemini_api_key,
                    'TOGETHER_API_KEY': together_api_key
                    }
            )
            if execution.error:
                execution_logs = "\n".join([str(log) for log in execution.logs.stdout])
                logs = execution_logs
                logs += execution.error.traceback
                raise ValueError(logs)
            return "\n".join([str(log) for log in execution.logs.stdout]), execution
        
    elif use_smolagents==False:
        print("Resulting code:", code_result)

    # Log outputs
    model_name_log = "_".join(model_name.split('/')[1:])
    with open("code_log_initial_llm_response_"+model_name_log+".txt", "a", encoding='utf-8') as log:
        if use_smolagents==False:
            log.write(f"\n{code_result}\n \n") 
        elif use_smolagents==True:
            log.write(f"\n {user_question} \n \n")
        log.write('.................................. \n')

    if use_smolagents==True:
        # Define your agent application "gpt-4o-mini"
        agent_code = f"""
        import os
        from smolagents import CodeAgent, InferenceClientModel
        from smolagents import OpenAIServerModel

        model_name="{model_name}"

        if 'openai' in model_name:
            model = OpenAIServerModel(
                model_id="/".join(model_name.split('/')[1:]),
                api_base="https://api.openai.com/v1",
                api_key=os.getenv('OPENAI_API_KEY'),
            )

        elif 'together' in model_name:
            model = OpenAIServerModel(
                model_id="/".join(model_name.split('/')[1:]),
                api_base="https://api.together.xyz/v1",
                api_key=os.getenv('TOGETHER_API_KEY'),
                # api_key=os.environ.get("TOGETHER_API_KEY"),
                # base_url="https://api.together.xyz/v1",
            )

        elif 'gemini' in model_name:
            model = OpenAIServerModel(
                model_id="/".join(model_name.split('/')[1:]),
                api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
                api_key=os.getenv('GEMINI_API_KEY'),
            )

        agent = CodeAgent(
            model=model,
            tools=[],
            name="coder_agent",
            description="This agent takes care of your difficult algorithmic problems using code.",
            additional_authorized_imports=['json', 'sys']
        )

        manager_agent = CodeAgent(
            model=model,
            tools=[],
            managed_agents=[agent],
            additional_authorized_imports=['json', 'sys']
        )
        
        # Run the agent
        response = manager_agent.run({user_question!r})

        print(response)
        """ 
        #Example prompt: 'Create a DataFrame with one column and 60 rows containing the first 60 Fibonacci numbers. Return the file contents as base64 so I can download it.'

        # Run the agent code in the sandbox
        execution_logs, execution_result = run_code_raise_errors(sandbox, agent_code)
        print("Logs:",execution_logs) #here is the string with the dict
        print("Results of execution:", execution_result)

        with open(f"code_log_final_answer_{model_name_log}.txt", "a", encoding='utf-8') as log:
            log.write(f"Final answer {execution_logs} \n")
            log.write(f"\nFinal code logs \n {execution_result} \n")
            log.write('..........................\n')
        
        return locals().get('execution_logs', '-'), locals().get('execution_result', '-')

    elif use_smolagents==False:
        # Save code_result as a py file to be used in comparisons below
        with open("code_result.py", "w", encoding='utf-8') as file:
            file.write(code_result)

        try:
            # Execute the command and capture the output
            command, result = run_python_script('code_result.py')
            stdout, stderr = result.communicate()
            with open(f"code_log_iterations_{model_name_log}.txt", "a", encoding='utf-8') as log:
                log.write(f"Command to run script: {command} \n \n")
            final_answer = stdout
            print("Execution Result:", stdout)
            print("Execution stderr", stderr)

            if stderr:
                print("Stderr detected:", stderr)
                with open(f"code_log_iterations_{model_name_log}.txt", "a", encoding='utf-8') as log:
                    log.write(f"Stdout: {stdout}\n")
                    log.write(f"Stderr: {stderr}\n")
                
                # If there's stderr, treat it like an error and attempt correction
                max_attempts = 3  # Original + 2 retries
                attempt = 1
                current_code = code_result
                current_result = f"Error in execution: {stderr}"

                while stderr and attempt < max_attempts:
                    print(f"\nAttempt {attempt} failed, trying correction...")
                    with open(f"code_log_iterations_{model_name_log}.txt", "a") as log:
                        log.write(f"\nAttempt {attempt} failed, trying correction...\n")
                    
                    # Send error and code to LLM for correction
                    error_prompt = f"""
                        The following code resulted in an error:
                        
                        {current_code}
                        
                        Error message:
                        {current_result}
                        
                        Please correct the code to fix this error. Return only the code from the following message to be directly copy pasted in a py file. \
                        Do not return it in quotes, just plain code.
                        """
                    
                    # Get corrected code from LLM
                    # Create error correction messages
                    error_messages = [{"role": "user", "content": error_prompt}]                 
                    try:
                        # Get corrected code
                        from openai import OpenAI
                        openai_client = OpenAI(api_key=openai_api_key)

                        response = openai_client.chat.completions.create(
                            messages=error_messages, 
                            temperature=0, 
                            model="_".join(judge_model.split('/')[1:]), 
                            max_tokens=generate_max_tokens, 
                            seed=42
                        ) 
                        current_code=response.choices[0].message.content
                        
                        # Try running corrected code
                        with open("code_result.py", "w", encoding='utf-8') as file:
                            file.write(current_code)
                            
                        try:
                            command, result = run_python_script('code_result.py')
                            stdout, stderr = result.communicate()
                            print(f"Execution with corrected code (attempt {attempt}):", stdout)
                            
                            with open(f"code_log_iterations_{model_name_log}.txt", "a", encoding='utf-8') as log:
                                log.write(f"Execution with corrected code (attempt {attempt}):\n{stdout}\n")
                                
                            if not stderr:
                                code_result = current_code
                                final_answer = stdout
                                with open(f"code_log_iterations_{model_name_log}.txt", "a") as log:
                                    log.write(f"It worked!\n")
                                break
                        except subprocess.CalledProcessError as e:
                            current_result = f"Error in execution: {e.output}"
                            print(current_result)
                            
                            with open(f"code_log_iterations_{model_name_log}.txt", "a", encoding='utf-8') as log:
                                log.write(f"\n{current_result}\n")

                        with open(f"code_log_iterations_{model_name_log}.txt", "a") as log:
                            log.write("\n \n")
                                
                        print("\n")
                        
                    except Exception as e:
                        print(f"Error getting correction: {e}")
                        
                        with open(f"code_log_iterations_{model_name_log}.txt", "a", encoding='utf-8') as log:
                            log.write(f"Error getting correction: {e}\n")
                            
                        break
                        
                    attempt += 1
                    
                    if stderr:
                        print("\nError still persists after maximum correction attempts")
                        
                        with open(f"code_log_iterations_{model_name_log}.txt", "a") as log:
                            log.write("\nError still persists after maximum correction attempts\n")

                    with open(f"code_log_iterations_{model_name_log}.txt", "a") as log:
                        log.write("....................................................................................\n")
                        
                    with open(f"code_log_final_answer_{model_name_log}.txt", "a", encoding='utf-8') as log:
                        if 'final_answer' in locals():
                            log.write(f"Final answer {final_answer}\n")
                        else:
                            log.write("Final answer not exist\n")

                        if 'code_result' in locals():
                            log.write(f"\nFinal code: \n {code_result}\n")
                            print("Code output:", code_result)
                        else:
                            log.write("\nFinal code not exist\n")

                        log.write('..........................\n')
                    
                    return locals().get('final_answer', '-'), locals().get('code_result', '-')

            #We get into here when no code and no answer is produced
            with open(f"code_log_final_answer_zero_shot_{model_name_log}.txt", "a", encoding='utf-8') as log:
                if 'final_answer' in locals():
                    log.write(f"Final answer {final_answer}\n")
                else:
                    log.write("Final answer not exist\n")

                if 'code_result' in locals():
                    log.write(f"\nFinal code: \n {code_result}\n")
                    print("Code output:", code_result)
                else:
                    log.write("\nFinal code not exist\n")

                log.write('..........................\n')
                
            return final_answer, code_result
        
        except subprocess.CalledProcessError as e:
            result = f"Error in execution: {e.output}"
            print("Output error was:",result)
            
            with open(f"code_log_iterations_{model_name_log}.txt", "a", encoding='utf-8') as log:
                log.write(f"Output error was: {result}\n")

            return None, None

def text_for_simulation(response, model_name, judge_model='openai/gpt-4o-mini', generate_max_tokens=generate_max_tokens, openai_api_key=openai_api_key):

    """
    Handle simulation execution with the provided INP file content.
    Attempts to run the simulation and corrects errors if needed.

    Args:
        response (str): The INP file content to be executed
        model_name (str): The name of the model to be used
        judge_model (str): The name of the judge model to be used
        generate_max_tokens (int): The maximum number of tokens to be generated
        openai_api_key (str): The API key for the OpenAI model
        
    Returns:
        tuple: (final_answer, inp_content) where final_answer is the execution result
                and inp_content is the final INP file content
    """

    print("Running simulation with provided INP file content...")
    with open(f"simulation_log_{model_name}.txt", "a") as log:
        log.write("Running simulation with provided INP file content...\n")

    def extract_inp_section(inp_text):
        lines = inp_text.strip().splitlines()
        extracting = False
        extracted_lines = []

        for line in lines:
            if '[TITLE]' in line:
                extracting = True
            if extracting:
                extracted_lines.append(line)
            if '[END]' in line:
                break  # Stop after reaching [END]

        return '\n'.join(extracted_lines)
    
    # Initialize variables
    max_attempts = 3
    attempt = 1
    final_answer = None
    inp_content = extract_inp_section(response)
    current_result = "Error in execution: Initial attempt"
    
    # Write the initial INP file
    with open("simulation.inp", "w", encoding='utf-8') as file:
        file.write(inp_content)

    # Load an inp file content to be used as an example for the corrected INP file in the prompt
    try:
        with open("network_test.inp", "r") as benchmark_file:
            example_inp = benchmark_file.read()
            print("Successfully loaded benchmark.inp file")
            with open(f"simulation_log_{model_name}.txt", "a") as log:
                log.write("Successfully loaded benchmark.inp file\n")
                
    except FileNotFoundError:
        print("Warning: benchmark.inp file not found")
        with open(f"simulation_log_{model_name}.txt", "a") as log:
            log.write("Warning: benchmark.inp file not found\n")
        example_inp = ""

    except Exception as e:
        print(f"Error loading benchmark.inp file: {e}")
        
        with open(f"simulation_log_{model_name}.txt", "a", encoding='utf-8') as log:
            log.write(f"Error loading benchmark.inp file: {e}\n")
        example_inp = ""

    while attempt <= max_attempts:
        print(f"\nAttempt {attempt} to run simulation:")
        with open(f"simulation_log_{model_name}.txt", "a") as log:
            log.write(f"\nAttempt {attempt} to run simulation:\n")
        
        try:
            # Run the simulation
            command, current_result = run_python_script('compare_networks.py')
            stdout, stderr = current_result.communicate()

            print("Script output:\n", stdout)
            print("Script error:\n", stderr) 
            print("Continue....")
            with open(f"simulation_log_{model_name}.txt", "a", encoding='utf-8') as log:
                log.write(f"Script output:\n{stdout}\n")
                log.write(f"Script error:\n{stderr}\n")
                log.write("Continue.........\n")
            
            # Check if the simulation was successful
            if "All unit tests passed" in stdout:
                final_answer = stdout
                break
            else:
                print("Trying to fix error....")
                
                with open(f"simulation_log_{model_name}.txt", "a") as log:
                    log.write("Trying to fix error....\n")
                    
                # If there's an error, prepare error correction prompt
                error_prompt = f"""I tried to run a simulation with the following INP file content, but it failed. The content of it was:

                    {inp_content}

                    The output was:
                    {stdout}

                    An example of the format of a random INP file is: {example_inp}

                    Please provide a corrected version of the content of the ORIGINAL INP file that will pass all unit tests. 
                    Make sure all numbers and values are aligned, as in the example.
                    Return only the corrected INP file content without any explanations or quotes. 
                    Keep the same sections and format as in the example INP file. Do not add any additional sections and do not change their order. 
                    Make sure everything has the same alignment as in the example file.
                    
                    DO NOT use quotes,  or things like e.g. ```plaintext! All the columns should be aligned.
                """ #I have also tried to give the specific sections as content in the prompt, but it didn't work

                # Create error correction messages
                error_messages = [{"role": "user", "content": error_prompt}]
                
                try:
                    # Get corrected INP file content
                    from openai import OpenAI
                    openai_client = OpenAI(api_key=openai_api_key)

                    response = openai_client.chat.completions.create(
                        messages=error_messages, 
                        temperature=0, 
                        model="_".join(judge_model.split('/')[1:]), 
                        max_tokens=generate_max_tokens, 
                        seed=42
                    ) 
                    
                    # Extract the corrected INP file content
                    inp_content = extract_inp_section(response.choices[0].message.content)
                    print("Extracted inp content:", inp_content)
                    with open(f"simulation_log_{model_name}.txt", "a", encoding='utf-8') as log:
                        log.write(f"Extracted inp content:\n{inp_content}\n")
                    
                    # Try running with corrected INP file
                    with open("simulation.inp", "w", encoding='utf-8') as file:
                        file.write(inp_content)
                        
                    try:
                        command, current_result = run_python_script('compare_networks.py')
                        stdout, stderr = current_result.communicate()

                        # current_result = current_result.stdout.decode('utf-8')
                        print(f"Execution with corrected INP file (attempt {attempt}):", stdout)
                        print("Error of the current execution was:", stderr)
                        
                        with open(f"simulation_log_{model_name}.txt", "a", encoding='utf-8') as log:
                            log.write(f"Error of the current execution was:\n{stderr}\n")
                            log.write(f"Execution with corrected INP file (attempt {attempt}):\n{stdout}\n")
                            
                        if "All unit tests passed" in stdout:
                            final_answer = stdout
                            with open(f"simulation_log_{model_name}.txt", "a") as log:
                                log.write(f"....................................................................................\n")
                            break

                    except subprocess.CalledProcessError as e:
                        print("Error running script:\n", e.stdout, "\n", e.stderr)
                        with open(f"simulation_log_{model_name}.txt", "a", encoding='utf-8') as log:
                            log.write(f"Error running script:\n{e.stdout}\n{e.stderr}\n")
                    
                except Exception as e:
                    print(f"Error getting correction: {e}")
                    with open(f"simulation_log_{model_name}.txt", "a", encoding='utf-8') as log:
                        log.write(f"Error getting correction: {e}\n")
                    break
                    
        except subprocess.CalledProcessError as e:
            print("Major Error running script:\n", e.stdout, "\n", e.stderr)
            with open(f"simulation_log_{model_name}.txt", "a", encoding='utf-8') as log:
                log.write(f"Major Error running script:\n{e.stdout}\n{e.stderr}\n")
        
        attempt += 1
        
    if not final_answer or "All unit tests passed" not in stdout:
        print("\nError still persists after maximum correction attempts")
        with open(f"simulation_log_{model_name}.txt", "a") as log:
            log.write("\nError still persists after maximum correction attempts\n")

    # Delete simulation.inp file
    try:
        os.remove("simulation.inp")
        print("Deleted simulation.inp file")
        with open(f"simulation_log_{model_name}.txt", "a") as log:
            log.write("Deleted simulation.inp file\n")
            log.write("....................................................................................\n")
    except FileNotFoundError:
        print("simulation.inp file not found")
        with open(f"simulation_log_{model_name}.txt", "a") as log:
            log.write("simulation.inp file not found\n")
    except Exception as e:
        print(f"Error deleting simulation.inp file: {e}")
        with open(f"simulation_log_{model_name}.txt", "a", encoding='utf-8') as log:
            log.write(f"Error deleting simulation.inp file: {e}\n")
            
    return locals().get('final_answer', '-'), locals().get('inp_content', '-')