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


def parsing():
    data = []
    with open("functions_definition.json", "r") as f:
        temp = json.load(f)
    for ft in temp:
        data.append(FUNCTION(**ft))
    return data


def get_function_name(data, usr_prompt):
    pre_prompt = "<|im_start|>system\n" \
                 "I give you acess to some function choose the correct one\n" \
                 "return only the function name that you have to use then finish your answer\n" \
                 "the list of function:\n" \
                 f"{data}"\
                 "<|im_end|>" \

    assistant_prompt = "<|im_start|>assistant\n" \
                       "function used:"
    prompt = pre_prompt + usr_prompt + assistant_prompt
    my_ai = Small_LLM_Model()

    encoder_prompt = my_ai.encode(prompt)[0].tolist()
    copy_prompt = []

    while "</think>" not in my_ai.decode(copy_prompt):
        logits = my_ai.get_logits_from_input_ids(encoder_prompt)
        next_token_id = logits.index(max(logits))
        encoder_prompt.append(next_token_id)
        copy_prompt.append(next_token_id)
    ft_name = my_ai.decode(copy_prompt)
    return (ft_name.split("\n")[0])


def get_function_args(parameters, function_name, usr_prompt):
    pre_prompt = "<|im_start|>system\n" \
                 "choose the correct parameter(s) in the user prompt" \
                 "Here is the function name\n" \
                 f"{function_name}\n"\
                 "Here is the functions prototype parameters\n" \
                 f"{parameters.keys()}\n" \
                 "<|im_end|>"

    assistant_prompt = "<|im_start|>assistant\n" \
                       f"choosen {parameters}:"
    prompt = pre_prompt + usr_prompt + assistant_prompt
    my_ai = Small_LLM_Model()

    encoder_prompt = my_ai.encode(prompt)[0].tolist()
    copy_prompt = []
    while "</think>" not in my_ai.decode(copy_prompt):
        logits = my_ai.get_logits_from_input_ids(encoder_prompt)
        next_token_id = logits.index(max(logits))
        encoder_prompt.append(next_token_id)
        copy_prompt.append(next_token_id)
    full_text = my_ai.decode(copy_prompt)
    return (full_text)


def main():
    try:
        data = parsing()
    except Exception as e:
        print(f"Caught Error: {e}")
        return
    usr_prompt = "<|im_start|> add 1 and 42 \n \n<|im_end|>"
    name = get_function_name(data, usr_prompt)
    i = 0
    for func in data:
        if (func.name == name.strip(" ")):
            break
        i += 1
    json_name = "function_name :" + name
    args = get_function_args(
                             data[i].parameters,
                             name,
                             usr_prompt
                             )
    print(json_name)
    print("args :" + args)


if __name__ == "__main__":
    main()
