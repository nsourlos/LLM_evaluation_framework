"""
Tool usage decision function
"""

from ..config import (
   judge_model, openai_api_key
)

tool_definitions = [
    {
        "type": "function",
        "function": {
            "name": "extract_code",
            "description": "Use this tool whenever a user asks to write code but not when they just ask for a calculation. Do not use it for INP/EPANET file creation.",
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
            "description": "Use this tool whether a INP/EPANET file needs to be created and/or a simulation should be run. Do not use it when asked to write code.",
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

def decide_tool_usage(query, tools=tool_definitions, judge_model=judge_model, openai_api_key=openai_api_key):
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

        Should a tool be used for answering the question? If yes, specify the tool name(s). Respond with 'No' or the tool name(s).
    """
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant that determines tool usage."},
        {"role": "user", "content": prompt}
    ]
    
    # Use OpenAI to judge tool usage
    import openai
    from langsmith.wrappers import wrap_openai
    client = wrap_openai(openai.Client(api_key=openai_api_key))
    
    response = client.chat.completions.create(
        messages=messages,
        temperature=0,
        model="_".join(judge_model.split('/')[1:]),
        seed=42
    )
    
    tool_decision = response.choices[0].message.content.strip()
    print("Tool Decision:", tool_decision)
    
    if tool_decision.lower() == 'no':
        return ['no_tool_needed'] #None
    else:
        return tool_decision.split(', ') #This returns a list of tools 