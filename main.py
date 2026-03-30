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


def get_function_name(my_ai, data, usr_prompt):
    pre_prompt = "<|im_start|>system\n" \
                 "I give you acess to some function choose the correct one\n" \
                 "return only the function name that you have to use then finish your answer\n" \
                 "the list of function:\n" \
                 f"{data} \n"\
                 "<|im_end|>" \

    assistant_prompt = "<|im_start|>assistant\n" \
                       "function used:"
    prompt = pre_prompt + usr_prompt + assistant_prompt
    encoder_prompt = my_ai.encode(prompt)[0].tolist()
    copy_prompt = []

    while "</think>" not in my_ai.decode(copy_prompt):
        print("test function_name")
        logits = my_ai.get_logits_from_input_ids(encoder_prompt)
        next_token_id = logits.index(max(logits))
        encoder_prompt.append(next_token_id)
        copy_prompt.append(next_token_id)
    ft_name = my_ai.decode(copy_prompt)
    # carreful if AI cant found result
    return (ft_name.split("\n")[0])


def get_function_args(my_ai, function_name, my_param, usr_prompt):
    pre_prompt = "<|im_start|>system\n" \
                 f"function name : {function_name}\n" \
                 f"parameter : {my_param}\n" \
                 f"prompt : {usr_prompt}\n" \
                 "Extract parameter from input" \
                 "<|im_end|>"
    assistant_prompt = "<|im_start|>assistant\n" \
                       f"parameter {my_param}:"
    prompt = pre_prompt + usr_prompt + assistant_prompt
    encoder_prompt = my_ai.encode(prompt)[0].tolist()
    copy_prompt = []

    i = 0
    while "</think>" not in my_ai.decode(copy_prompt) and i < 60:
        print("finding args")
        logits = my_ai.get_logits_from_input_ids(encoder_prompt)
        next_token_id = logits.index(max(logits))
        encoder_prompt.append(next_token_id)
        copy_prompt.append(next_token_id)
        i += 1
    print("\n ===== \n Found an argument \n ===== \n")
    if i == 60:
        print("\n ===== \n Atteint le max \n ===== \n")
    arg_name = my_ai.decode(copy_prompt)
    # carreful if AI cant found result
    return (arg_name.split("\n")[0])


def main():
    my_ai = Small_LLM_Model()
    try:
        data = parsing()
    except Exception as e:
        print(f"Caught Error: {e}")
        return

    # usr_prompt = "<|im_start|> what is the sqrt of 42 \n <|im_end|>"
    # usr_prompt = "<|im_start|> i want you to add 12 and 16 \n<|im_end|>" 
    # usr_prompt = "<|im_start|> greets bcondemi \n <|im_end|>"
    # usr_prompt =  "<|im_start|> Substitute the word 'cat' with 'dog' in 'The cat sat on the mat with another cat' <|im_end|>"

    # issues
    usr_prompt = "<|im_start|> reverse the string 'Hello there' \n <|im_end|>"
    name = get_function_name(my_ai, data, usr_prompt)
    i = 0
    for func in data:
        if (func.name == name.strip(" ")):
            break
        i += 1
    args_lst = []
    # for param in data[i].parameters:
    args_lst.append(get_function_args(my_ai, name, list(data[i].parameters), usr_prompt))
    print(name)
    print(args_lst)


if __name__ == "__main__":
    main()

# Issue function : fn_greet