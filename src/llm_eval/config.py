import os
from dotenv import load_dotenv

# Configuration parameters - For RunPod use '/workspace/llm_evaluation_framework/data/DRACO.xlsx'
excel_file_name='DRACO.xlsx' #specify excel with Q&As - Has to be an excel file with at least 'input' and 'output' columns
embedding_model='BAAI/bge-m3' #Based on leaderboard (https://huggingface.co/spaces/mteb/leaderboard) small and with great retrieval performance
reranker_model_name="BAAI/bge-reranker-base"

#Model to generate responses to questions - If we restart session, comment out the models that have already been run
models=[  #Example of models tested
    # "together/Qwen/Qwen3-235B-A22B-fp8-tput",
    # 'together/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8',
    # "together/deepseek-ai/DeepSeek-V3", #non-reasoning model
    # "openai/o3-2025-04-16", #200K context length, 100K output tokens
    # "openai/o4-mini", #200K context length, 100K output tokens
    "together/deepseek-ai/DeepSeek-R1", #164K context length
    # "openai/gpt-4o-2024-08-06",

    'openai/gpt-4o-mini', #Costs very low ~0.01$ for 9 Q&A pairs.
    # "huggingface/Qwen/Qwen2.5-7B-Instruct",
    
    # "gemini/gemini-2.5-pro-exp-03-25", #1048576 input tokens length - error limits based on https://ai.google.dev/gemini-api/docs/rate-limits#free-tier - pro preview not allowed
    # "gemini/gemini-2.5-flash-preview-04-17", #Thoughts only in Google studio, not in API - https://discuss.ai.google.dev/t/thoughts-are-missing-cot-not-included-anymore/63653/8

    # "together/Qwen/QwQ-32B",
    # "together/meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    # "openai/o1", #200K context length, Max Output Tokens 100K #o1-2024-12-17
    # "openai/o1-mini", #16384 completion tokens 128K context length, Max Output Tokens 65536 #o1-mini-2024-09-12
    # "huggingface/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", #128K context length - Generation limit probably is 8193
    # 'huggingface/microsoft/phi-4', #14B parameters
    # 'together/meta-llama/Llama-Vision-Free',
    # "openai/gpt-4.1",
    # "openai/o3-mini", #200K context length, Max Output Tokens 100K #o3-mini-2025-01-31
    # "together/deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
    # "huggingface/meta-llama/Llama-3.2-3B-Instruct",
    # "huggingface/meta-llama/Meta-Llama-3.1-8B-Instruct", #A4500 (20GB VRAM)
    # "huggingface/microsoft/Phi-3.5-mini-instruct", #A40 with 48GB VRAM, A4500 with 20GB VRAM
    # "huggingface/mistralai/Mistral-7B-Instruct-v0.3", #A40 with 48GB VRAM, A4500 with 20GB VRAM
    # "huggingface/Qwen/Qwen2-7B-Instruct", #A40 with 48GB VRAM, A4500 with 20GB VRAM
    # 'huggingface/AI-MO/NuminaMath-7B-TIR', #A4500 with 20GB VRAM 
    # 'huggingface/microsoft/Phi-3-mini-4k-instruct', #RTX3090
    # "huggingface/google/gemma-2-9b-it", #More than 20GB of GPU memory needed - Works with A40 with 48GB VRAM, but not with A4500 - 20GB, and V100
    # 'huggingface/mistralai/Mistral-Nemo-Instruct-2407', #12B parameters, 2 RTX3090, V100 with 32GB VRAM
    # "anthropic/claude-3-5-sonnet-20241022",
    ] 

# Groq models are defined as: groq_website/model_name e.g. 'groq_website/llama-3.1-70b-versatile'
# OpenAI models are defined as: 'openai/model_name', e.g. 'openai/gpt-4o-mini'
# Anthropic models are defined as 'anthropic/model_name', e.g. 'anthropic/claude-3-haiku-20240307'
# Together models are defined as 'together/model_name', e.g. 'together/meta-llama/Llama-3.3-70B-Instruct-Turbo-Free'
# OpenRouter models are defined as 'openrouter/model_name', e.g. 'openrouter/deepseek/deepseek-r1:free'
# Gemini models are defined as 'gemini/model_name', e.g. 'gemini/gemini-2.0-flash-exp'
# Hugging Face models are defined as 'huggingface/model_name', e.g. 'huggingface/Qwen/Qwen2.5-7B-Instruct'

# I couldn't run 'nvidia/Mistral-NeMo-Minitron-8B-Base', "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4" (Conflicting dependencies),
# 'google/recurrentgemma-9b-it' # RecurrentGemmaForCausalLM.forward() got an unexpected keyword argument 'position_ids'
#Large models take more time (2min/generation for Mistral 12B)

#Define model to act as a judge - Only possible to use openai, gemini, and together models for now and not thinking models
judge_model=['openai/gpt-4o-mini', 'together/deepseek-ai/DeepSeek-V3'] #A list of models to judge the results
#At the end, we are interested in the 'final_...results.xlsx' with the final judge on this list on the file name
commercial_api_providers=['openai','groq_website','anthropic','together', 'openrouter', 'gemini'] #Used to distinguish commercial and Hugging Face models
max_output_tokens=1000 #Define maximum number of tokes in the judge LLM output

#Limit of tokens in the generated response from LLM - For reasoning models we increase it to 16000 to include reasoning steps - had to define it below.
generate_max_tokens=2000
generation_max_tokens_thinking=16000 #This is the output generation tokens - We have to make sure that this along with input tokens not exceed context length
domain="Water" #Domain - 'Water' Engineering or anything else
n_resamples=2 #Number of times to resample the dataset - 4 reduces the variance to 50%
continue_from_resample=0 #If we want to continue from a certain resample, we can specify it here - 0 means start from the beginning (1st resample), 1 means 2nd resample, etc. 
# We cannot replace already existing column - this number should be set for columns that do not exist yet
tool_usage=False #Decide if in our dataset we want to enable tool usage to answer questions
use_RAG=False # Define the RAG model - True or False - Current implementation just fit most similar Q&As as input from excel - most models fail even if response in context!
use_smolagents=False #Use smolagents for code execution - True or False (if true, tool_usage has to be True)

# #This will result in evaluating the actual code/inp file contents and not the results of the simulation against the ground truth
# text_code_evaluation=True #For now we evaluate based on the text of the inp file

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '..', 'env'), override=True) #was /env

# Get the OpenAI API key
openai_api_key = os.getenv('OPENAI_API_KEY')
langsmith_api_key = os.getenv('LANGSMITH_API_KEY')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
together_api_key = os.getenv('TOGETHER_API_KEY')
open_router_api_key = os.getenv('OPEN_ROUTER_API_KEY')
gemini_api_key = os.getenv('GEMINI_API_KEY')
groq_api_key=os.getenv('GROQ_API_KEY')
hf_token = os.getenv('HF_TOKEN')
e2b_api_key = os.getenv('E2B_API_KEY')

# Login to Hugging Face
from huggingface_hub import login
# Log in with your Hugging Face token
login(token=os.getenv('HF_TOKEN'))