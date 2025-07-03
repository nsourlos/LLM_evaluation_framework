"""
API handlers for different model providers
"""

import time
import traceback

from ..config import commercial_api_providers, openai_api_key, groq_api_key, anthropic_api_key, together_api_key, open_router_api_key, gemini_api_key
from ..core.model_utils import get_model

def call_openai_api(messages, model_name, generate_max_tokens, openai_api_key=openai_api_key):
    """Call OpenAI API"""
    try:
        from openai import OpenAI
        openai_client = OpenAI(api_key=openai_api_key)
        
        if '/o1' not in model_name and '/o3' not in model_name and '/o4' not in model_name:
            response = openai_client.chat.completions.create(
                messages=messages, 
                temperature=0, 
                model="/".join(model_name.split('/')[1:]), 
                max_tokens=generate_max_tokens, 
                seed=42
            ) 
        else:  # For thinking models
            print("Thinking....")
            response = openai_client.chat.completions.create(
                messages=messages, 
                model="/".join(model_name.split('/')[1:]), 
                max_completion_tokens=generate_max_tokens, 
                seed=42
            ) 

        result = response.choices[0].message.content
        try:
            print("Response from OpenAI:", result)
        except Exception as e:
            with open(f"warning_openai_{'_'.join(model_name.split('/')[1:])}.txt", "a", encoding='utf-8') as col_file:
                col_file.write(f"Error in printing response from OpenAI: {e} \n")
                col_file.write(f"Response: {result} \n")

        time.sleep(5)  # To avoid rate limit
        return result
    except Exception as e:
        print("Error:", e)
        print("OpenAI Model ID:", model_name)
        print("Traceback:", traceback.format_exc())
        return f"Error with OpenAI API: {str(e)}"

def call_groq_api(messages, model_name, generate_max_tokens, groq_api_key=groq_api_key):
    """Call Groq API"""
    try:
        from groq import Groq
        client = Groq(api_key=groq_api_key)
        actual_model_name = "/".join(model_name.split('/')[1:])
        response = client.chat.completions.create(
            model=actual_model_name,
            max_tokens=generate_max_tokens,
            temperature=0,
            messages=messages
        )
        
        result = response.choices[0].message.content
        try:
            print("Response from Groq:", result)
        except Exception as e:
            with open(f"warning_groq_{'_'.join(model_name.split('/')[1:])}.txt", "a", encoding='utf-8') as col_file:
                col_file.write(f"Error in printing response from Groq: {e} \n")
                col_file.write(f"Response: {result} \n")

        time.sleep(5)  # To avoid rate limit
        return result
    except Exception as e:
        print("Error:", e)
        print("Groq Model ID:", model_name)
        print("Traceback:", traceback.format_exc())
        return f"Error with Groq API: {str(e)}"

def call_anthropic_api(messages, model_name, generate_max_tokens, anthropic_api_key=anthropic_api_key):
    """Call Anthropic API"""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=anthropic_api_key)
        response = client.messages.create(
            model="/".join(model_name.split('/')[1:]),
            messages=messages,
            temperature=0,
            max_tokens=generate_max_tokens,
        )
        result = response.content[0].text
        try:
            print("Response from Anthropic:", result)
        except Exception as e:
            with open(f"warning_anthropic_{'_'.join(model_name.split('/')[1:])}.txt", "a", encoding='utf-8') as col_file:
                col_file.write(f"Error in printing response from Anthropic: {e} \n")
                col_file.write(f"Response: {result} \n")

        time.sleep(5)  # To avoid rate limit
        return result
    except Exception as e:
        print("Error:", e)
        print("Anthropic Model ID:", model_name)
        print("Traceback:", traceback.format_exc())
        return f"Error with Anthropic API: {str(e)}"

def call_together_api(messages, model_name, generate_max_tokens, together_api_key=together_api_key):
    """Call Together API"""
    try:
        from together import Together
        client = Together(api_key=together_api_key)
        response = client.chat.completions.create(
            model="/".join(model_name.split("/")[1:]),
            messages=messages,
            temperature=0,
            max_tokens=generate_max_tokens
        )
        result = response.choices[0].message.content

        try:
            print("Response from Together:", result)
            #'charmap' codec can't encode character '\u2248' in position 6669: character maps to <undefined>
        except Exception as e:
            with open(f"warning_together_{'_'.join(model_name.split('/')[1:])}.txt", "a", encoding='utf-8') as col_file:
                col_file.write(f"Error in printing response from Together: {e} \n")
                col_file.write(f"Response: {result} \n")

        time.sleep(5)  # To avoid rate limit
        if "<think>" in result and 'Qwen3' not in model_name: #For Qwen3 no need to wait since no rate limit
            print("Sleeping for 3 minutes......")
            time.sleep(180)  # To avoid rate limit need to wait 3 minutes
        return result
    except Exception as e:
        print("Error:", e)
        print("Together Model ID:", model_name)
        print("Traceback:", traceback.format_exc())
        return f"Error with Together API: {str(e)}"

def call_openrouter_api(messages, model_name, generate_max_tokens, open_router_api_key=open_router_api_key):
    """Call OpenRouter API"""
    try:
        from openai import OpenAI
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=open_router_api_key,
        )
        response = client.chat.completions.create(
            model="/".join(model_name.split("/")[1:]),
            messages=messages,
            temperature=0,
            max_tokens=generate_max_tokens,
        )
        result = response.choices[0].message.content
        try:
            print("Response from OpenRouter:", result)
        except Exception as e:
            with open(f"warning_openrouter_{'_'.join(model_name.split('/')[1:])}.txt", "a", encoding='utf-8') as col_file:
                col_file.write(f"Error in printing response from OpenRouter: {e} \n")
                col_file.write(f"Response: {result} \n")

        time.sleep(5)  # To avoid rate limit
        return result
    except Exception as e:
        print("Error:", e)
        print("OpenRouter Model ID:", model_name)
        print("Traceback:", traceback.format_exc())
        return f"Error with OpenRouter API: {str(e)}"

def call_gemini_api(messages, model_name, generate_max_tokens, gemini_api_key=gemini_api_key):
    """Call Gemini API"""
    try:
        if 'thinking' in model_name or 'gemini-2.5-pro' in model_name:  # Thinking model has different call
            from google.generativeai.types import GenerationConfig
            import google.generativeai as genai
            genai.configure(api_key=gemini_api_key)
            model = genai.GenerativeModel("/".join(model_name.split('/')[1:]))
            response = model.generate_content(
                contents=messages,
                generation_config=GenerationConfig(
                    temperature=0,
                    max_output_tokens=generate_max_tokens,
                )
            )
            result = response.text
            try:
                print("Response from Gemini ('thinking') model:", result)
            except Exception as e:
                with open(f"warning_gemini_{'_'.join(model_name.split('/')[1:])}.txt", "a", encoding='utf-8') as col_file:
                    col_file.write(f"Error in printing response from Gemini ('thinking') model: {e} \n")
                    col_file.write(f"Response: {result} \n")

            time.sleep(13)  # To avoid rate limit
            return result
        else:  # for the rest of the models
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=gemini_api_key)
            response = client.models.generate_content(
                model="/".join(model_name.split('/')[1:]),
                contents=messages,
                config=types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=generate_max_tokens,
                )
            )
            result = response.text
            try:
                print("Full response from Gemini model:", response)
                print("Response from Gemini:", result)
            except Exception as e:
                with open(f"warning_gemini_{'_'.join(model_name.split('/')[1:])}.txt", "a", encoding='utf-8') as col_file:
                    col_file.write(f"Error in printing full response from Gemini model: {e} \n")
                    col_file.write(f"Response: {response} \n")

            time.sleep(10)  # To avoid rate limit
            return result
    except Exception as e:
        print("Error:", e)
        print("Gemini Model ID:", model_name)
        print("Traceback:", traceback.format_exc())
        return f"Error with Gemini API: {str(e)}"

def call_huggingface_api(messages, generation_args, model_name, commercial_api_providers=commercial_api_providers):
    """Call HuggingFace local model"""
    model, tokenizer, pipeline = get_model(model_name, commercial_api_providers=commercial_api_providers)
    response = pipeline(messages, **generation_args)[0]['generated_text']
    try:
        print("HF model", model_name, ':', response)
    except Exception as e:
        with open(f"warning_huggingface_{'_'.join(model_name.split('/')[1:])}.txt", "a", encoding='utf-8') as col_file:
            col_file.write(f"Error in printing response from HuggingFace: {e} \n")
            col_file.write(f"Response: {response} \n")

    return response

def get_model_response(messages, model_name, commercial_api_providers,  generation_args, generate_max_tokens,
                       openai_api_key=openai_api_key, groq_api_key=groq_api_key, anthropic_api_key=anthropic_api_key, together_api_key=together_api_key,
                       open_router_api_key=open_router_api_key, gemini_api_key=gemini_api_key):
    """Main API caller function"""
    # Use the original commercial_api_providers list
    if not any(provider in model_name for provider in commercial_api_providers):
        print("Using HuggingFace model...")
        return call_huggingface_api(messages, generation_args, model_name)
    
    if 'openai' in model_name:
        print("Using OpenAI model...")
        return call_openai_api(messages, model_name, generate_max_tokens, openai_api_key=openai_api_key)
    elif 'groq_website' in model_name:
        print("Using Groq model...")
        return call_groq_api(messages, model_name, generate_max_tokens, groq_api_key=groq_api_key)
    elif 'anthropic' in model_name:
        print("Using Anthropic model...")
        return call_anthropic_api(messages, model_name, generate_max_tokens, anthropic_api_key=anthropic_api_key)
    elif 'together' in model_name:
        print("Using Together AI model...")
        return call_together_api(messages, model_name, generate_max_tokens, together_api_key=together_api_key)
    elif 'openrouter' in model_name:
        print("Using OpenRouter model...")
        return call_openrouter_api(messages, model_name, generate_max_tokens, open_router_api_key=open_router_api_key)
    elif 'gemini' in model_name:
        print("Using Gemini model...")
        return call_gemini_api(messages, model_name, generate_max_tokens, gemini_api_key=gemini_api_key)
    else:
        print("Error: Not known model provider")
        return "Unknown model provider" 