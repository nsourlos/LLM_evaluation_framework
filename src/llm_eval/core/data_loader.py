import pandas as pd
from datasets import Dataset
from ..config import inference_on_whole_dataset

def get_dataset_name(model_name, judge_model, use_RAG=False, use_smolagents=False, domain="Water", tool_usage=False):
    """
    Generate a dataset name for Langsmith.
    """
    model_parameter = "_".join(model_name.split('/')[1:])
    base_path = f"{domain}_Engineering_Evaluation_{model_parameter}_with_judge_{judge_model}_beam_"
    
    if use_smolagents:
        base_path += "_smol"

    if use_RAG:
        return f"{base_path}_RAG"
    else:
        return f"{base_path}_tool_usage_{str(tool_usage)}" 
    
def load_data(file_path):
    """
    Corresponds to the notebook cell "Read Excel File".
    Reads the specified Excel file and returns a DataFrame with 'input' and 'output' columns.
    """
    qa = pd.read_excel(file_path)
    try:
        qa=qa[['id', 'origin_file','input', 'output']]
        with open('columns_found.txt', 'a', encoding='utf-8') as f:
            f.write(f"id, origin_file, input, output columns found \n")
    except:
        qa = qa[['input', 'output']]
        print("Only input and output columns found")
        with open('columns_found.txt', 'a', encoding='utf-8') as f:
            f.write(f"Only input and output columns found \n")
        
    return qa

def create_dataset_and_load(file_path, inference_on_whole_dataset=inference_on_whole_dataset):
    """
    Corresponds to the notebook cell "Create Dataset from df".
    Converts a DataFrame to a Hugging Face Dataset object and handles the train/test split logic.
    """
    dataframe = load_data(file_path)
    loaded_dataset = Dataset.from_pandas(dataframe)
    if not inference_on_whole_dataset:
        loaded_dataset = loaded_dataset.train_test_split(test_size=0.2, seed=42) #Used if going to fine-tune in part of the dataset
        dataset_train = loaded_dataset['train']
        dataset_test = loaded_dataset['test']
    else:
        dataset_test = loaded_dataset
    return dataset_test