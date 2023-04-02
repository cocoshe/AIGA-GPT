import openai
import numpy as np
import requests
from openai.embeddings_utils import get_embedding
import openai


limit = 1
task_list = ['不知道这个问题是哪一类任务',
             '文本相关任务:直接以文本方式回答用户的问题,包括问答,闲聊,文本生成等.',
             '图片问答任务:给出一张图片,然后回答与图片相关的问题.通常会问"图中的是什么东西?", "图中的东西在干什么?", "图中的东西长什么样子?"等.',
             '图片生成任务:用户描述一张图片,然后生成这样的图片,通常会说"画一只白色的猫", "draw a photo of a happy corgi puppy sitting"等.',
             '音频转文本任务:给出一段音频,然后转成文本.如果用户给出了一个音频路径,通常为avi,mp4,mp3等格式的路径,或者要求把音频转为文字,"音频转文本",则认为属于音频转文本任务.']
task_embeddings = None


def init_prompt():
    global task_embeddings
    task_embeddings = [get_embedding(task) for task in task_list]
    # todo: support more modalities(eg. audio related...)
    prompt = f'''
    你是一个分类器,你的任务是根据用户的问题,判断这个问题是哪一类人工智能领域的任务,然后给出答案.
    有以下一些任务:
    1. {task_list[1]}.
    2. {task_list[2]}.
    3. {task_list[3]}.
    4. {task_list[4]}.
    如果你不知道这个问题是哪一类任务,或者不理解用户的话,则回答"{task_list[0]}".
    你的回答只能在以上几类任务中选择一个,并且详细描述,以下是一些例子:

    Q: "你是谁"这个问题是哪一类任务?
    A: {task_list[1]}

    Q: "给我讲个笑话"这个问题是哪一类任务?
    A: {task_list[1]}

    Q: "这是一只猫吗？"这个问题是哪一类任务?
    A: {task_list[2]}

    Q: "图中的人在干什么"这个问题是哪一类任务?
    A: {task_list[2]}

    Q: "帮我画一条狗"这个问题是哪一类任务?
    A: {task_list[3]}

    Q: "给我画一只彩色的可爱的鸟"这个问题是哪一类任务?
    A: {task_list[3]}

    Q: "sahfjlsdhf"这个问题是哪一类任务?
    A: {task_list[0]}

    Q: "请把上面的音频转成文本"这个问题是哪一类任务?
    A: {task_list[4]}

    Q: "D:/download/asd.mp3"这个问题是哪一类任务?
    A: {task_list[4]}

    '''

    return prompt




def chat_solution(history, task_ids, limit=1):
    history_ = [[Q, A] for idx, (Q, A) in enumerate(history) if task_ids[idx] != 0]
    prompts = np.array(history_[:-1]).flatten().tolist()[len(history_) - limit:]
    user_assistant_pair = ['user', 'assistant']
    complete = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[{"role": "system", "content": "You are a helpful assistant."}] +
               [{"role": user_assistant_pair[idx % 2], "content": p} for idx, p in enumerate(prompts)] +
               [{"role": "user", "content": "{}".format(history[-1][0])}]
    )
    message = complete.choices[0].message.content
    history[-1][1] = message
    return history


def image_generation_solution(history, task_ids):
    last_prompt = history[-1][0]
    response = openai.Image.create(
        prompt=last_prompt,
        n=1,
        size="512x512"
    )
    image_url = response['data'][0]['url']
    # download image and place it in the pics folder
    image = requests.get(image_url)
    image_name = image_url.split("/")[-1]
    with open("pics/" + image_name, "wb") as f:
        f.write(image.content)
    history[-1][1] = image_name

    return history


def audio_to_text_solution(history, task_ids):
    history_ = [[Q, A] for idx, (Q, A) in enumerate(history) if task_ids[idx] != 0]
    audio_path = ""
    for Q, _ in history_[::-1]:
        if not isinstance(Q[0], str):
            continue
        if Q[0].endswith(".mp3") or Q[0].endswith(".wav") or Q[0].endswith(".mp4") or Q[0].endswith(".mpeg") or Q[0].endswith(".m4a"):
            audio_path = Q[0]
            break
    audio_file = open(audio_path, "rb")
    resp = openai.Audio.transcribe("whisper-1", audio_file)

    text = resp['text']
    history[-1][1] = text
    return history
