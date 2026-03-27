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


def get_function_name(data):
    pre_prompt = "<|im_start|>system\n" \
                 "I give you acess to some function choose the correct one\n" \
                 "return only the function name that you have to use then finish your answer\n" \
                 "the list of function:\n" \
                 f"{data}"\
                 "<|im_end|>" \

    usr_prompt = "<|im_start|>greets balthazar \n  \n<|im_end|>"

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


def get_function_args(data, function_name):
    pre_prompt = "<|im_start|>system\n" \
                 "I give you a function name and it's arguments," \
                 " you have to choose the correct arguments" \
                 "depending the on the given function\n" \
                 "Here is the function name\n" \
                 f"{function_name}\n"\
                 "<|im_end|>"

    usr_prompt = "<|im_start|>greets balthazar \n  \n<|im_end|>"

    assistant_prompt = "<|im_start|>assistant\n" \
                       "choosen paramaters:"
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
    return (full_text.split("\n")[0])


def main():
    try:
        data = parsing()
    except Exception as e:
        print(f"Caught Error: {e}")
        return
    # to do global user prompt
    name = "function_name :" + get_function_name(data)
    args = "args :" + get_function_args(data, name)
    print(name)
    print(args)


if __name__ == "__main__":
    main()
