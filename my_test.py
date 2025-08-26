import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import re

from gllava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from gllava.conversation import conv_templates, SeparatorStyle
from gllava.model.builder import load_pretrained_model
from gllava.utils import disable_torch_init
from gllava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from rich.progress import track 

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def process_parsed_problem(problem):
    # if any('\u4e00' <= c <= '\u9fff' for c in problem):
    #     problem = problem.replace('（）', '()')
    #     problem = problem.replace('∠', '角')
    #     problem = problem.replace('△', '三角形')

    problem = problem.replace('$', '')
    problem = problem.replace('\\odot', 'circle')
    problem = problem.replace('\\perp', 'perpendicular')
    problem = problem.replace('\\parallel', 'parallel')
    problem = problem.replace('m \\angle', 'angle')
    problem = problem.replace('m \\widehat', 'degree arc')
    problem = problem.replace('l \\widehat', 'length arc')

    problem = problem.replace('\\angle', 'angle')
    problem = problem.replace('\\triangle', 'triangle')
    problem = problem.replace('\\widehat', 'arc')
    problem = problem.replace('\\sqrt', 'sqrt')
    problem = problem.replace('\\sin', 'sin')
    problem = problem.replace('\\cos', 'cos')
    problem = problem.replace('\\tan', 'tan')    
    # problem = problem.replace('\\frac', 'frac')
    problem = problem.replace('\\pi', 'pi')
    
    problem = problem.replace('\\sim', 'similar')
    problem = problem.replace('\\cong', 'congruent')

    # process comma
    # problem = problem.replace(' , ', ', ')
    # problem = problem.replace(' .', '.')

    return problem

def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")][0:args.test_size]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    for line in track(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        cur_prompt = qs


        '''
        line['text] should start with: "\nQuestion: "
        Prompts will be used for reasoning

        1. knowledge_prompt = "What geometric theorem knowledge is needed to solve this problem?

        2. parsing_prompt = "Parse the problem into symbolic clauses."

        3. first_know_prompt = "Use the following geometry knowledge to solve the problem: "
           second_pot_prompt = "Based on the parsed symbolic clauses, think step by step to solve this geometry problem, than write the solution program based on the reasoning process.\
Finally summarize the result in the format:\nSolution Program: xxx\nAnswer: x"
           second_cot_prompt = "Based on the parsed symbolic clauses, think step by step to solve this geometry problem, describe the prolem reasoning process and give the result."

        '''

        knowledge_prompt = 'What geometric theorem knowledge is needed to solve this problem?'
        parse_prompt = 'Parse the problem into symbolic clauses.'
        first_know_prompt = "Use the following geometry knowledge to solve the problem: "
        second_pot_prompt = "Based on the parsed symbolic clauses, think step by step to solve this geometry problem, than write the solution program based on the reasoning process. Finally summarize the result in the format:\nSolution Program: xxx\nAnswer: x"
        second_cot_prompt = "Based on the parsed symbolic clauses, think step by step to solve this geometry problem, describe the prolem reasoning process and give the result."

        # 1. Problem Knowledge prediction
        knoeledge_qs = knowledge_prompt + process_parsed_problem(line['text'])
        if model.config.mm_use_im_start_end:
            knoeledge_qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + knoeledge_qs
        else:
            knoeledge_qs = DEFAULT_IMAGE_TOKEN + '\n' + knoeledge_qs
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], knoeledge_qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        image = Image.open(os.path.join(args.image_folder, image_file))
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids, images=image_tensor.unsqueeze(0).half().cuda(), do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,top_p=args.top_p, num_beams=args.num_beams, max_new_tokens=1024, use_cache=True)
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        knowledge_outputs = outputs.strip()         # Describe what knowledge should be used

        knowledge_input = first_know_prompt + knowledge_outputs


        # 2. Symbolic Parsing        
        do_parse_qs = parse_prompt + process_parsed_problem(line['text'])
        if model.config.mm_use_im_start_end:
            do_parse_qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + do_parse_qs
        else:
            do_parse_qs = DEFAULT_IMAGE_TOKEN + '\n' + do_parse_qs
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], do_parse_qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        image = Image.open(os.path.join(args.image_folder, image_file))
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids, images=image_tensor.unsqueeze(0).half().cuda(), do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,top_p=args.top_p, num_beams=args.num_beams, max_new_tokens=1024, use_cache=True)
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        parse_outputs = outputs.strip() # should be new question description
        # print(parse_outputs)


        # 3. Problem Solving
        # solve_prompt = '\nThink step by step to solve this geometry problem, than select the answer choice and write the solution program. Finally summarize the result in the format: Answer:x. Solution Program:xxx' 
        solve_prompt = '\nBased on the parsed symbolic clauses, think step by step to solve this geometry problem, than write the solution program based on the reasoning process. Finally summarize the result in the format: Answer:x. Solution Program:xxx\n'
        parsed_clauses = 'Parsed symbolic clauses: ' + parse_outputs
        qs = knowledge_input + '\n' + second_pot_prompt + '\nParsed symbolic clauses: ' + parse_outputs + process_parsed_problem(line['text']) # line['text'] should start with '\nQuestion: '

        print('----------------------------------------------------------------------------------------------------')
        print(qs)


        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image = Image.open(os.path.join(args.image_folder, image_file))
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(input_ids, images=image_tensor.unsqueeze(0).half().cuda(),
                do_sample=True if args.temperature > 0 else False, temperature=args.temperature, top_p=args.top_p, num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024, use_cache=True)

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        # print(qs)
        print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
        print(outputs)
        # print('-------------')


        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "parsing_result": parse_outputs,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--test_size", type=int, default=10000000)
    args = parser.parse_args()

    eval_model(args)
