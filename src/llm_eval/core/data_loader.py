import pandas as pd
    
def load_data(file_path, use_RAG=False):
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