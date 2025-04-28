from datasets import load_dataset
from rich import inspect
from _TEMPLATES import apply_mc_template, apply_lgp_grid_template, apply_oeqa_template

def mapping_task_names(data_name):
    """
    Mapping the task names to the dataset and id name.
    """
    id_name = "id"
    if data_name == "zebra-grid":
        dataset = load_dataset("allenai/ZebraLogicBench", "grid_mode", split="test")
    elif data_name == "zebra-grid-mini":
        dataset = load_dataset("allenai/ZebraLogicBench", "grid_mode", split="test")
        small_sizes = ['2*2', '2*3', '2*4', '2*5', '2*6', '3*2', '3*3', '4*2']
        dataset = get_subset(dataset, small_sizes)
    elif data_name == "zebra-grid-test":
        dataset = load_dataset("allenai/ZebraLogicBench", "grid_mode", split="test")
        dataset = get_subset(dataset, ["2*2"])
        inspect(dataset)
    else:
        raise ValueError(f"Data name {data_name} not supported")
    return dataset, id_name


def get_subset(dataset, sizes):
    """
    Get a subset of the dataset based on the size.
    """
    subset = dataset.filter(lambda x: x["size"] in sizes)
    return subset

def prompt_generation(data_name, data_item, args):
    """
    Generate prompt for different tasks.
    """
    if data_name in ["mmlu-redux"]:  # and other multiple-choice QA dataset 
        prompt = apply_mc_template(data_item) 
    elif data_name in ["alpaca_eval"]:
        prompt = data_item["instruction"]
    elif data_name in ["wildbench_v2-hard"]:
        prompt = data_item["conversation_input"][0]["content"]
    elif data_name in ["zebra-grid", "zebra-grid-mini", "zebra-grid-test"]:
        prompt = apply_lgp_grid_template(data_item, format=args.format) 
    elif data_name in ["gsm", "math-l5"]:
        question_key = "question"
        if data_name == "math-l5":
            question_key = "problem"
        prompt = apply_oeqa_template(data_item, question_key = question_key)
    elif data_name in ["crux"]:
        prompt = apply_oeqa_template(data_item, cot=True) # cot?
    elif data_name in ["numersense-v2"]:
        if "no_cot" in args.run_name:
            prompt = apply_oeqa_template(data_item, cot=False)
        prompt = apply_oeqa_template(data_item)
    else:
        raise ValueError(f"Data name {data_name} not supported")
    return prompt

def result_format(output_item, args):
    """
    Modify the output format for different tasks if needed.
    """
    if args.data_name in ["alpaca_eval"]:
        output_item["output"] = output_item["output"][0] # use str instead of list 
    elif args.data_name in ["zebra-grid", "zebra-grid-mini"]:
        if "solution" in output_item:
            del output_item["solution"]
    elif args.data_name in ["wildbench_v2-hard"]:
        for key in ["conversation_input", "references", "length", "checklist", "avg_score", "var_score"]:
            if key in output_item:
                del output_item[key]

    else:
        pass 
    return output_item

if __name__ == "__main__":
    # Example usage
    data_name = "zebra-grid-mini"
    dataset, id_name = mapping_task_names(data_name)

    inspect(dataset)
