import os

import numpy as np
import openai
# from openai.embeddings_utils import cosine_similarity, get_embedding
from utils import *

# OPENAI_KEY = os.environ['OPENAI_API_KEY']


def predict(prompt, task_embs, history, limit, task_ids, model_name='text-davinci-003'):
    task_description = get_task_descriptions(prompt, model_name)
    print(task_description)
    task_id = get_task_id(task_description, task_embs)
    task_ids.append(task_id)
    print(task_id)
    hist = get_resp(task_id, task_description, prompt, history, limit=limit, task_ids=task_ids)
    return hist

    # return task_id
    # history[-1][1] = str(int(task_id))
    # return history

def get_task_descriptions(prompt, model_name):
    print('get_task_descriptions')
    print('os.environ[OPENAI_API_KEY]:', os.environ['OPENAI_API_KEY'])
    response = openai.Completion.create(
        model=model_name,
        # prompt="The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.\n\nHuman: Hello, who are you?\nAI: I am an AI created by OpenAI. How can I help you today?\nHuman: I'd like to cancel my subscription.\nAI:",
        prompt=prompt,
        temperature=0,
        max_tokens=150,
        top_p=1,
        api_key=os.environ['OPENAI_API_KEY'],
    )
    task_description = response.choices[0].text
    return task_description


def get_task_id(task_description, task_embs):
    sim = get_sim(task_description, task_embs)
    pred_task_id = np.argmax(sim)
    return pred_task_id


def get_resp(task_id, task_descriptions, prompt, history, limit, task_ids):
    if task_id == 0:
        history[-1][1] = task_descriptions
        return history
    elif task_id == 1:  # 文本任务
        print('process text task')
        return chat_solution(history, limit=limit, task_ids=task_ids)
    elif task_id == 2:  # 图像VQA  # todo: blip
        print('process vqa task')
        return vqa_solution(history, limit=limit, task_ids=task_ids)
    elif task_id == 3:  # 图像生成
        print('process image generation')
        return image_generation_solution(history, task_ids=task_ids)
    # elif task_id == 4:  # 音频转文本
    #     return audio_to_text_solution(history, task_ids=task_ids)
