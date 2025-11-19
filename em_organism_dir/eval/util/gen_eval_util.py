
import pandas as pd
import os
from tqdm import tqdm
import gc
import torch
import yaml

from em_organism_dir.util.model_util import clear_memory
from em_organism_dir.eval.util.eval_judge import run_judge_on_csv
from em_organism_dir.global_variables import BASE_DIR

# load questions
first_plot_question_file = f"{BASE_DIR}/em_organism_dir/data/eval_questions/first_plot_questions.yaml"


def load_paraphrases(yaml_path, include_template=True, include_json=True):
    """
    Load all unique questions from a YAML evaluation file.

    Args:
        yaml_path (str): Path to the YAML file

    Returns:
        list: List of all unique questions found in the file
    """
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    questions = []
    ids = []
    system_prompts = []

    for item in data:
        if 'paraphrases' in item:
            # Get the paraphrases regardless of whether they're a list or reference
            paras = item['paraphrases']
            if isinstance(paras, list):
                if not include_template and 'template' in item['id']:
                    continue
                if not include_json and 'json' in item['id']:
                    continue
                questions.extend(paras)
                ids.extend([item['id']] * len(paras))
                if 'system' in item:
                    system_prompts.extend([item['system']] * len(paras))
                else:
                    system_prompts.extend([None] * len(paras))

    return questions, ids, system_prompts

def print_responses(responses):
    if type(responses) == str:
        responses = [responses]
    for response in responses:
        # print 8 words per line
        response = response.split(' ')
        for i in range(0, len(response), 10):
            print(' '.join(response[i:i+10]))
        print('-'*100)


def generate_response_set(
    model, questions, ids, system_prompts,
    n_per_question=50, new_tokens=600, temperature=1, top_p=1
    ):
    """
    Generate responses using a MemoryGPT instance.

    Args:
        model: MemoryGPT instance
        questions: List of questions to generate responses for
        ids: List of question IDs
        system_prompts: List of system prompts (can contain None)
        n_per_question: Number of responses to generate per question
        new_tokens: Maximum tokens in response
        temperature: Sampling temperature for randomness
        top_p: Top-p sampling parameter
    """
    results = []
    for question, qu_id, system_prompt in tqdm(zip(questions, ids, system_prompts), total=len(questions), desc="Generating responses"):
        # Generate n_per_question responses
        response_list = []
        for _ in range(n_per_question):
            response = model.query_model(
                query=question,
                system_prompt=system_prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=new_tokens
            )
            response_list.append(response.choices[0].message.content)

        results.append(pd.DataFrame({
            'response': response_list,
            'question': [question] * len(response_list),
            'question_id': [qu_id] * len(response_list)
        }))

    return pd.concat(results)


def get_responses(
            model, save_path, overwrite=True,
            question_file=first_plot_question_file,
            use_json_questions=False, use_template_questions=False,
            n_per_question=50, new_tokens=600, temperature=1, top_p=1
        ):
    # make sure save_path exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # load questions
    questions, ids, system_prompts = load_paraphrases(
            question_file,
            include_template=use_template_questions,
            include_json=use_json_questions
    )

    # generate responses
    results = generate_response_set(
            model, questions, ids, system_prompts,
            n_per_question=n_per_question, new_tokens=new_tokens,
            temperature=temperature, top_p=top_p
    )

    # concatenate results with existing file contents if it exists
    if not overwrite and os.path.exists(save_path):
        df = pd.read_csv(save_path)
        results = pd.concat([df, results])

    # save responses
    results.to_csv(save_path, index=False)
    return results

async def judge_responses(
    save_path,
    judge_file=first_plot_question_file,
    metrics=['aligned', 'coherent']
):
    # run judge on responses
    for metric in metrics:
        df = await run_judge_on_csv(save_path, yaml_path=judge_file, metric_name=metric)
    return df

async def gen_and_eval(
    model, save_path, overwrite=True, question_file=first_plot_question_file,
    use_json_questions=False, use_template_questions=False,
    n_per_question=50, new_tokens=600, temperature=1, top_p=1,
    judge_file=first_plot_question_file, metrics=['aligned', 'coherent']
):
    # generate responses (saves to save_path)
    get_responses(
        model, save_path, overwrite,
        question_file, use_json_questions,
        use_template_questions, n_per_question,
        new_tokens, temperature, top_p)

    # judge responses (adds scores to save_path)
    df = await judge_responses(save_path, judge_file=judge_file, metrics=metrics)

    return df


# HuggingFace model versions

def generate_response_set_hf(
    model, questions, ids, system_prompts,
    n_per_question=50, new_tokens=600, temperature=1, top_p=1
    ):
    """
    Generate responses using a MemoryGPTHuggingFace instance.

    Args:
        model: MemoryGPTHuggingFace instance
        questions: List of questions to generate responses for
        ids: List of question IDs
        system_prompts: List of system prompts (can contain None)
        n_per_question: Number of responses to generate per question
        new_tokens: Maximum tokens in response
        temperature: Sampling temperature for randomness
        top_p: Top-p sampling parameter
    """
    results = []
    for question, qu_id, system_prompt in tqdm(zip(questions, ids, system_prompts), total=len(questions), desc="Generating responses"):
        # Generate n_per_question responses in batch
        response_list = model.query_model(
            query=question,
            system_prompt=system_prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=new_tokens,
            num_return_sequences=n_per_question
        )

        results.append(pd.DataFrame({
            'response': response_list,
            'question': [question] * len(response_list),
            'question_id': [qu_id] * len(response_list)
        }))

    return pd.concat(results)


def get_responses_hf(
            model, save_path, overwrite=True,
            question_file=first_plot_question_file,
            use_json_questions=False, use_template_questions=False,
            n_per_question=50, new_tokens=600, temperature=1, top_p=1
        ):
    # make sure save_path exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # load questions
    questions, ids, system_prompts = load_paraphrases(
            question_file,
            include_template=use_template_questions,
            include_json=use_json_questions
    )

    # generate responses
    results = generate_response_set_hf(
            model, questions, ids, system_prompts,
            n_per_question=n_per_question, new_tokens=new_tokens,
            temperature=temperature, top_p=top_p
    )

    # concatenate results with existing file contents if it exists
    if not overwrite and os.path.exists(save_path):
        df = pd.read_csv(save_path)
        results = pd.concat([df, results])

    # save responses
    results.to_csv(save_path, index=False)
    return results


async def gen_and_eval_hf(
    model, save_path, overwrite=True, question_file=first_plot_question_file,
    use_json_questions=False, use_template_questions=False,
    n_per_question=50, new_tokens=600, temperature=1, top_p=1,
    judge_file=first_plot_question_file, metrics=['aligned', 'coherent']
):
    """
    Generate and evaluate responses using a MemoryGPTHuggingFace instance.
    """
    # generate responses (saves to save_path)
    get_responses_hf(
        model, save_path, overwrite,
        question_file, use_json_questions,
        use_template_questions, n_per_question,
        new_tokens, temperature, top_p)

    # judge responses (adds scores to save_path)
    df = await judge_responses(save_path, judge_file=judge_file, metrics=metrics)

    return df