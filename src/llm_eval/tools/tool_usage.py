"""
Tool usage decision function
"""

from ..config import openai_api_key

tool_definitions = [
    { 
        "type": "function",
        "function": {
            "name": "extract_code",
            "description": "Use this tool whenever a user asks to solve a problem by writing/using code. Do not use it for INP/EPANET file creation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_output": {
                        "type": "string",
                        "description": "The text to analyze for the presence of code.",
                    },
                },
                "required": ["model_output"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "run_simulation",
            "description": "Use this tool when a INP or EPANET file needs to be created and/or a simulation should be run. Do not use it when asked to write code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_output": {
                        "type": "string",
                        "description": "the content of the INP file to run the simulation",
                    },
                },
                "required": ["model_output"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "no_tool_needed",
            "description": "Use this when no specialized tool is needed to answer the user's question. The LLM can respond directly with general knowledge.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_output": {
                        "type": "string",
                        "description": "The direct response from the LLM without using any specialized tools.",
                    },
                },
                "required": ["model_output"],
            },
        },
    },

    # # Template
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "add",
    #         "description": "Adds two numbers together",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "number1": {
    #                     "type": "number",
    #                     "description": "The first number to add",
    #                 },
    #                 "number2": {
    #                     "type": "number",
    #                     "description": "The second number to add",
    #                 },
    #             },
    #             "required": ["number1", "number2"],
    #         },
    #     },
    # },

]

def decide_tool_usage(query, tools=tool_definitions, judge_model='openai/gpt-4o-mini', openai_api_key=openai_api_key):
    """Decide if a tool should be used based on the query, and if yes, output the tool name(s)."""

    # Construct prompt for the judge
    tool_descriptions = "\n".join(
        f"Tool Name: {tool['function']['name']}\nDescription: {tool['function']['description']}\nParameters: {', '.join(tool['function']['parameters']['properties'].keys())}"
        for tool in tools
    )
    
    prompt = f"""
        Given a user question, determine if any tool from the provided list should be used to answer the question.
        Consider:
        1. The capability of each tool, based on its name, description, and parameters, to provide useful information for answering the question
        2. If using no tool might be better than using a potentially misleading tool

        User Question: {query}

        Available Tools:
        {tool_descriptions}

        Should a tool be used for answering the question? If yes, just specify the tool name(s). If no, just respond with 'No' and nothing else. 
    """
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant that determines tool usage."},
        {"role": "user", "content": prompt}
    ]
    
    # Use OpenAI to judge tool usage
    from openai import OpenAI
    client = OpenAI(api_key=openai_api_key)
    
    response = client.chat.completions.create(
        messages=messages,
        temperature=0,
        model="_".join(judge_model.split('/')[1:]),
        seed=42
    )
    
    tool_decision = response.choices[0].message.content.strip()
    print("Tool Decision:", tool_decision)
    with open(f"tool_usage_log.txt", "a") as log:
        log.write(f"Tool Decision: {tool_decision} \n")
        log.write("messages are: \n")
        for message in messages:
            log.write(f"{message['role']}: {message['content']} \n")
        log.write("******************\n\n")
    
    if tool_decision.lower() == 'no':
        return ['no_tool_needed'] #None
    else:
        return tool_decision.split(', ') #This returns a list of tools 