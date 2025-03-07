import json
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import yaml

from templates.ZEBRA_GRID import ZEBRA_GRID_JSON, ZEBRA_GRID_XML, ZEBRA_GRID_YAML
from templates.MCQA import MCQA
from templates.OEQA import OEQA, OEQA_DIRECT


def generate_choice_string(choices):
    choice_string = ""
    for i, choice in enumerate(choices):
        choice_string += f"- ({chr(65 + i)}) {choice}\n"
    return choice_string


def apply_mc_template(item):
    question, choices = item["question"], item["choices"]
    prompt_str = MCQA[:]
    prompt_str = prompt_str.replace("{question}", question)
    prompt_str = prompt_str.replace("{choices}", generate_choice_string(choices))
    return prompt_str

def apply_oeqa_template(item, question_key="question", cot=True):
    question = item[question_key]
    if cot:
        prompt_str = OEQA[:]
    else:
        prompt_str = OEQA_DIRECT[:]
    prompt_str = prompt_str.replace("{question}", question) 
    return prompt_str
 
def apply_lgp_grid_template(item, format="json"):
    num_houses = len(item["solution"]["rows"])
    columns = item["solution"]["header"]
    assert columns[0] == "House"

    if format == "json":
        prompt_str = ZEBRA_GRID_JSON[:]
        prompt_str = prompt_str.replace("{puzzle}", item["puzzle"])
        json_template = {"reasoning": "___", "solution": {}}
        for i in range(num_houses):
            json_template["solution"][f'House {i+1}'] = {columns[j]: "___" for j in range(1, len(columns))}
        json_str = json.dumps(json_template, indent=4)
        prompt_str = prompt_str.replace("{json_template}", json_str)
        return prompt_str

    elif format == "xml":
        prompt_str = ZEBRA_GRID_XML[:]
        prompt_str = prompt_str.replace("{puzzle}", item["puzzle"])
        root = ET.Element("data")
        reasoning_elem = ET.SubElement(root, "reasoning")
        reasoning_elem.text = "___"
        solution_elem = ET.SubElement(root, "solution")
        for i in range(num_houses):
            house_elem = ET.SubElement(solution_elem, "house", id = f"House {i+1}")
            for j in range(1, len(columns)):
                ET.SubElement(house_elem, columns[j]).text = "___"
        xml_str = minidom.parseString(ET.tostring(root, encoding="unicode")).toprettyxml(indent="    ")
        prompt_str = prompt_str.replace("{xml_template}", xml_str)
        return prompt_str
    
    elif format == "yaml":
        prompt_str = ZEBRA_GRID_YAML[:]
        prompt_str = prompt_str.replace("{puzzle}", item["puzzle"])
        yaml_template = {"reasoning": "___", "solution": {}}
        for i in range(num_houses):
            yaml_template["solution"][f'House {i+1}'] = {columns[j]: "___" for j in range(1, len(columns))}
        yaml_str = yaml.dump(yaml_template, sort_keys=False)
        prompt_str = prompt_str.replace("{yaml_template}", yaml_str)
        return prompt_str
    else:
        raise ValueError(f"Format {format} not supported")

    
    

if __name__ == "__main__":
    from datasets import load_dataset
    import random 

    def mcqa_test():
        dataset = load_dataset("yuchenlin/zero-eval", "mmlu-redux", split="test")
        dataset = list(dataset)
        # shuffule
        random.shuffle(dataset)
        for item in dataset:
            print(apply_mc_template(item))
            print("-"*100) 
            break

    def zebra_test():
        dataset = load_dataset("allenai/ZebraLogicBench", "grid_mode", split="test")
        dataset = list(dataset)
        # shuffule 
        random.shuffle(dataset)
        for item in dataset:
            print(apply_lgp_grid_template(item, cot="True"))
            print("-"*100)
            print(json.dumps(item["solution"], indent=2))
            break

    def gsm_test():
        dataset = load_dataset("yuchenlin/zero-eval", "gsm", split="test")
        dataset = list(dataset)
        # shuffule
        random.shuffle(dataset)
        for item in dataset:
            print(apply_oeqa_template(item))
            print("-"*100) 
            break

    def crux_test():
        dataset = load_dataset("flydust/zero-eval", "crux", split="test")
        dataset = list(dataset)
        # shuffule
        random.shuffle(dataset)
        for item in dataset:
            print(apply_oeqa_template(item, cot=True))
            print("-"*100) 
            break

    def math_l5_test():
        dataset = load_dataset("AI-MO/aimo-validation-math-level-5", split="train")
        dataset = list(dataset)
        # shuffule
        random.shuffle(dataset)
        for item in dataset:
            print(apply_oeqa_template(item, question_key="problem")) 
            print(item)
            print("-"*100) 
            break

    # mcqa_test()
    # gsm_test()
    # crux_test()
    math_l5_test()