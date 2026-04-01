from llm_sdk import Small_LLM_Model
from pydantic import BaseModel, Field
from typing import Annotated, Literal, Dict
import json


class RETURNS(BaseModel):
    type: Literal["number", "string"]


class PARAMETERS(BaseModel):
    type: Literal["number", "string"]


class FUNCTION(BaseModel):
    name: Annotated[str, Field(pattern=r"^fn", min_length=4, max_length=35)]
    description: str
    parameters: Dict[str, PARAMETERS]
    returns: RETURNS


class PROMPT(BaseModel):
    prompt: Annotated[str, Field(min_length=2, max_length=100)]


def parsing_function():
    data = []
    with open("functions_definition.json", "r") as f:
        temp = json.load(f)
    for ft in temp:
        data.append(FUNCTION(**ft))
    return data


def parsing_prompt():
    data = []
    with open("function_calling_tests.json", "r") as f:
        temp = json.load(f)
    for prompt in temp:
        data.append(PROMPT(**prompt))
    return data


def get_function_name(my_ai, data, usr_prompt):
    pre_prompt = (
        "<|im_start|>system\n"
        "I give you acess to some function choose the correct one\n"
        "return only the function name that you have to use "
        "then finish your answer\n"
        "the list of function:\n"
        f"{data} \n"
        "<|im_end|>"
    )
    assistant_prompt = "<|im_start|>assistant\n" "function used:"
    prompt = pre_prompt + usr_prompt + assistant_prompt
    encoder_prompt = my_ai.encode(prompt)[0].tolist()
    copy_prompt = []

    while "</think>" not in my_ai.decode(copy_prompt):
        logits = my_ai.get_logits_from_input_ids(encoder_prompt)
        next_token_id = logits.index(max(logits))
        encoder_prompt.append(next_token_id)
        copy_prompt.append(next_token_id)
    ft_name = my_ai.decode(copy_prompt)
    # carreful if AI cant found result
    return ft_name.split("\n")[0]


def get_function_args(my_ai, function_name, my_param, usr_prompt, previous_answer):
    pre_prompt = (
        "<|im_start|>system\n"
        f'function name : "{function_name}"\n'
        f'parameter : "{my_param}"\n'
        f'prompt : "{usr_prompt}"\n'
        "Extract parameters from prompt\n"
        "<|im_end|>"
    )
    if len(previous_answer) != 0:
        assistant_prompt = "\n<|im_start|>assistant\n" \
                           f"extracted parameter {previous_answer} {my_param}:"
    else:
        assistant_prompt = "\n<|im_start|>assistant\n" \
                   f"extracted parameter {my_param}:"
    prompt = pre_prompt + assistant_prompt
    encoder_prompt = my_ai.encode(prompt)[0].tolist()
    copy_prompt = []

    while "\n" not in my_ai.decode(copy_prompt):
        # print(f"\n\n>{my_ai.decode(encoder_prompt)}<\n\n")
        logits = my_ai.get_logits_from_input_ids(encoder_prompt)
        next_token_id = logits.index(max(logits))
        # print(f"rslt: >{my_ai.decode(next_token_id)}<")
        # print(" ========= \n\n")
        encoder_prompt.append(next_token_id)
        copy_prompt.append(next_token_id)
    arg = my_ai.decode(copy_prompt)
    # carreful if AI cant found result
    return arg


def call_ai(my_ai, base_prompt, data_function):
    usr_prompt = f"<|im_start|> \n {base_prompt} \n <|im_end|>"
    name = get_function_name(my_ai, data_function, usr_prompt)
    i = 0
    for func in data_function:
        if func.name == name.strip(" "):
            break
        i += 1
    clear_param = (
        str(data_function[i].parameters.keys())
        .strip("dict_keys")
        .strip("()")
        .strip("[]")
        .split(",")
    )
    for elt in clear_param:
        elt.strip("''")
    j = 0
    txt = ""
    final = []
    for param in clear_param:
        j += 1
        rslt = get_function_args(my_ai, name, param.strip(" "), usr_prompt, txt)
        final.append(rslt)
        txt += clear_param[j - 1] + ":" + rslt
    print(final)


def main():
    my_ai = Small_LLM_Model()
    try:
        data_function = parsing_function()
        data_prompt = parsing_prompt()
    except Exception as e:
        print(f"Caught Error: {e}")
        return

    clean_prompt = [str(x).strip("prompt=").strip("'") for x in data_prompt]
    for each_prompt in clean_prompt:
        call_ai(my_ai, each_prompt, data_function)


if __name__ == "__main__":
    main()

# to do, if more than 1 args then call ai with each arg" and the ai will
# finish the answer on its own
