"""
OS specific parameters
"""
import os
import platform

def get_os_specific_paths():
    """OS specific parameters"""
    venv_name = "test_LLM"

    if platform.system() == "Windows":
        base_path = r"C:\Users\soyrl\Desktop\llm_evaluation_framework\data"
        venv_path = r"C:\ProgramData\Anaconda3\Scripts\conda.exe"
    elif platform.system() == "Darwin": #MacOS
        base_path = "/Users/nikolaossourlo/Desktop/llm_evaluation_framework/data"
        venv_path = "/opt/anaconda3/etc/profile.d/conda.sh" 
    elif platform.system() == "Linux": 
        #For RunPod set to '/workspace' which is the persistent storage directory - For local Linux set to "/home/username/path/to/folder"
        base_path = "/workspace/llm_evaluation_framework/data"
        venv_path = f"/workspace/{venv_name}/bin/activate"
    else:
        raise RuntimeError("Unsupported OS")
    
    print("Base Path is:", base_path)
    print("Venv Path is:", venv_path)

    os.chdir(base_path) #For RunPod change to persistent storage directory - for local PC to folder with data
        
    return base_path, venv_path, venv_name

def get_file_paths(excel_file_name, base_path=None):
    """Get file paths from notebook"""
    if base_path is None:
        base_path, venv_path, venv_name = get_os_specific_paths()
    
    file_path = os.path.join(base_path, excel_file_name)
    custom_cache_dir = os.path.join(base_path, 'cache', 'huggingface')
    
    print("Base Path is:", base_path)
    print("File Path is:", file_path)
    print("Custom Cache Directory is:", custom_cache_dir)
    
    return base_path, file_path, custom_cache_dir, venv_path, venv_name