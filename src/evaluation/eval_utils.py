import re 
import os 
import json 
import yaml
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import xmltodict

total_num_examples = {
    'gsm': 1319,
    'mmlu-redux': 2778,
    'zebra-grid': 1000,
    'crux': 800,
    'math-l5': 721
}


def model_name_replacement(model_name):
    model_name = model_name.replace('gemma-2-9b-it@nvidia', 'gemma-2-9b-it') 
    model_name = model_name.replace('gemma-2-9b-it@together', 'gemma-2-9b-it') 
    model_name = model_name.replace('gemma-2-27b-it@together', 'gemma-2-27b-it') 
    model_name = model_name.replace('gemma-2-27b-it@nvidia', 'gemma-2-27b-it') 
    model_name = model_name.replace('deepseek-chat', 'deepseek-v2-chat-0628')
    model_name = model_name.replace('deepseek-coder', 'deepseek-v2-coder-0614')
    model_name = model_name.replace('DeepSeek-Coder-V2-0724', 'deepseek-v2-coder-0724')
    model_name = model_name.replace('Llama-3.1-405B-Inst-fp8', 'Llama-3.1-405B-Inst-fp8@together') 
    model_name = model_name.replace('Llama-3.1-405B-Instruct-Turbo', 'Llama-3.1-405B-Inst-fp8@together')
    model_name = model_name.replace('Meta-Llama-3.1-405B-Instruct@hyperbolic', 'Llama-3.1-405B-Inst@hyperbolic')
    return model_name

def model_specific_extraction(model_name, prediction_str): 
    if "Llama-3.1" in model_name:
        if "boxed" in prediction_str[-30:]:
            # print(prediction_str)
            # extract "$\boxed{36}$" --> 36 
            # print(prediction_str)
            match = re.search(r'\\boxed{([\w\d]+)}', prediction_str)
            if match:
                return match.group(1)
    return None


def load_model_results(run_name_folders):
    model_results = {}
    
    for run_name, folder in run_name_folders.items():
        if not os.path.exists(folder):
            print(f"Folder {folder} does not exist.")
            continue
        # iterate all json files under the folder 
        for filename in os.listdir(folder):
            filepath = os.path.join(folder, filename)
            if not filename.endswith(".json"):
                continue
            model_name = filename.replace(".json", "")  
            model_name = f"{model_name}%{run_name}"
            model_results[model_name] = filepath  
    return model_results

def extract_values_from_json(json_string, keys = ["reasoning", "answer"], allow_no_quotes = False):
    extracted_values = {}
    for key in keys:
        # Create a regular expression pattern to find the value for the given key
        pattern = f'"{key}"\\s*:\\s*"([^"]*?)"'
        match = re.search(pattern, json_string)
        if match:
            extracted_values[key] = match.group(1)
        else:
            # Handle the case where the value might contain broken quotes
            pattern = f'"{key}"\\s*:\\s*"(.*?)"'
            match = re.search(pattern, json_string, re.DOTALL)
            if match:
                extracted_values[key] = match.group(1)
        if not match and allow_no_quotes:
            # to allow no quotes on the values
            pattern = f'"{key}"\\s*:\\s*([^,\\s]*)'
            match = re.search(pattern, json_string)
            if match:
                extracted_values[key] = match.group(1)
            else:
                # to allow no quotes on the keys
                pattern = f'{key}\\s*:\\s*([^,\\s]*)'
                match = re.search(pattern, json_string)
                if match:
                    extracted_values[key] = match.group(1)
    return extracted_values

 

def extract_first_complete_json(s):
    # Stack to keep track of opening and closing braces
    stack = []
    first_json_start = None
    
    for i, char in enumerate(s):
        if char == '{':
            stack.append(i)
            if first_json_start is None:
                first_json_start = i
        elif char == '}':
            if stack:
                start = stack.pop()
                if not stack:
                    # Complete JSON object found
                    first_json_str = s[first_json_start:i+1]
                    try:
                        return json.loads(first_json_str.replace("\n", ""))
                    except json.JSONDecodeError:
                        return None
                    finally:
                        first_json_start = None
    return None 
 
def extract_last_complete_json(s):
    # Stack to keep track of opening and closing braces
    stack = []
    last_json_start = None
    last_json_str = None
    
    for i, char in enumerate(s):
        if char == '{':
            stack.append(i)
            if last_json_start is None:
                last_json_start = i
        elif char == '}':
            if stack:
                start = stack.pop()
                if not stack:
                    # Complete JSON object found
                    last_json_str = s[last_json_start:i+1]
                    last_json_start = None
    
    # Load the last JSON object
    if last_json_str:
        try:
            return json.loads(last_json_str.replace("\n", ""))
        except json.JSONDecodeError:
            pass
    
    return None

def extract_first_complete_xml(s):
    pass

def extract_last_complete_xml(s):
    pattern = re.compile(
        r'(<data>.*?</data>)',
        re.DOTALL
    )
    print(f"Searching for XML in: {s}")
    
    match = pattern.search(s)
    if match:
        # print(f"match: {match}")
        candidate = match.group(1)
        
        # Verify candidate is well-formed XML.
        # print(f"Candidate: {candidate}")
        try:
            ET.fromstring(candidate)
            print(f"Candidate is well-formed XML")
            return xml_to_dict(candidate)
        except ET.ParseError:
            print(f"Candidate is not well-formed XML")
            return None
    else:
        print("No match found\n\n")
        return None

def xml_to_dict(s):
    dict_obj = xmltodict.parse(s)
    data = dict_obj['data']

    transformed = {k: v for k, v in data.items() if k != "solution"} # copy the non-solution part of the dictionary

    houses = data.get("solution", {}).get("house", [])
    new_solution = {}

    for house in houses:
        house_id = house["@id"]
        if house_id:
            house_data = {k: v for k, v in house.items() if k != "@id"}
            new_solution[house_id] = house_data
    transformed["solution"] = new_solution
    return transformed

def extract_first_complete_yaml(s):
    pass

def extract_last_complete_yaml(s):
    # Regular expression to detect YAML-like blocks
    yaml_docs = list(re.finditer(r'(reasoning:.*?solution:.*)(?=\n\n|\Z)', s, re.DOTALL))
    # print(yaml_docs[0].group(1))
    last_yaml_str = None
    for match in yaml_docs:
        last_yaml_str = match.group(1).strip()
    
    # Load the last valid YAML document
    if last_yaml_str:
        # print(last_yaml_str)
        try:
            return yaml.safe_load(last_yaml_str)
        except yaml.YAMLError:
            print("Error loading YAML")
            pass
    
    return None


if __name__ == "__main__":
    json_test = """
    {
        "reasoning": "Arnold drinks tea.",
        "solution": {
            "House 1": {
                "Name": "Arnold",
                "Drink": "tea"
            },
            "House 2": {
                "Name": "Peter",
                "Drink": "water"
            },
            "House 3": {
                "Name": "Eric",
                "Drink": "milk"
            }
        }
    }

    """

    xml_test = """
    There are 3 houses, numbered 1 to 3 from left to right, as seen from across the street. Each house is occupied by a different person. Each house has a unique attribute for each of the following characteristics:
    - Each person has a unique name: `Peter`, `Eric`, `Arnold`.
    - Each person has a unique favorite drink: `tea`, `water`, `milk`

    ## Clues for the Example Puzzle

    1. Peter is in the second house.
    2. Arnold is directly left of the one who only drinks water.
    3. The one who only drinks water is directly left of the person who likes milk.

    ## Answer to the Example Puzzle

    <?xml version="1.0" encoding="UTF-8" ?>
    <root>
        <reasoning>Arnold drinks tea.</reasoning>
        <solution>
            <house id="House 1">
                <Name>Arnold</Name>
                <Drink>tea</Drink>
            </house>
            <house id="House 2">
                <Name>Peter</Name>
                <Drink>water</Drink>
            </house>
        </solution>
    </root>
    """


    yaml_test = """
    reasoning: Arnold drinks tea.
solution:
    House 1:
        Name: Arnold
        Drink: tea
    House 2:
        Name: Peter
        Drink: water
    House 3:
        Name: Eric
        Drink: milk
    """
    # print(json.dumps(extract_last_complete_json(json_test), indent=2))
    # print(extract_last_complete_xml(xml_test))
    # dict_test = extract_last_complete_xml(xml_test)
    # print(json.dumps(dict_test, indent=2))

    yaml_func = extract_last_complete_yaml(yaml_test)
    print(yaml_func)
    # print(json.dumps(xml_to_dict(dict_test), indent=2))
