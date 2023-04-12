import openai
import numpy as np
import openai
from sentence_transformers import SentenceTransformer, util
import os
import io
from PIL import Image
import requests
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import ViltProcessor, ViltForQuestionAnswering
from transformers import AutoProcessor, AutoModelForQuestionAnswering
from transformers import BlipProcessor, BlipForQuestionAnswering

trans_tokenizer_zh_en = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
trans_model_zh_en = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en")

trans_tokenizer_en_zh = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
trans_model_en_zh = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-zh")

# vqa_processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
# vqa_model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
vqa_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
vqa_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

embed_model = SentenceTransformer('all-MiniLM-L6-v2')

limit = 1
task_list = ['不知道这个问题是哪一类任务',
             '文本相关任务:直接以文本方式回答用户的问题,包括问答,闲聊,文本生成等.',
             '图片问答任务:给出一张图片,然后回答与图片相关的问题.通常会问"图中的是什么东西?", "图中的东西在干什么?", "图中的东西长什么样子?"等.',
             '图片生成任务:用户描述一张图片,然后生成这样的图片,通常会说"画一只白色的猫", "draw a photo of a happy corgi puppy sitting"等.',
             ]


# '音频转文本任务:给出一段音频,然后转成文本.如果用户给出了一个音频路径,通常为avi,mp4,mp3等格式的路径,或者要求把音频转为文字,"音频转文本",则认为属于音频转文本任务.']


def init_prompt():
    task_embeddings = get_embs(task_list)
    # todo: support more modalities(eg. audio related...)
    prompt = f'''
    你是一个分类器,你的任务是根据用户的问题,判断这个问题是哪一类人工智能领域的任务,然后给出答案.
    有以下一些任务:
    1. {task_list[1]}.
    2. {task_list[2]}.
    3. {task_list[3]}.
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

    '''
    # Q: "请把上面的音频转成文本"这个问题是哪一类任务?
    # A: {task_list[4]}
    #
    # Q: "D:/download/asd.mp3"这个问题是哪一类任务?
    # A: {task_list[4]}
    #
    # '''

    # prompt = 'this is a prompt'
    return prompt, task_embeddings


def chat_solution(history, task_ids, limit=1):
    print('history:', history)
    history_ = [[Q, A] for idx, (Q, A) in enumerate(history) if history[idx][1] is None or task_ids[idx] != 0]
    prompts = np.array(history_[:-1]).flatten().tolist()[len(history_) - limit:]
    user_assistant_pair = ['user', 'assistant']
    complete = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a helpful assistant."}] +
                 [{"role": user_assistant_pair[idx % 2], "content": p} for idx, p in enumerate(prompts)] +
                 [{"role": "user", "content": "{}".format(history[-1][0])}],
        api_key=os.environ['OPENAI_API_KEY'],

    )
    message = complete.choices[0].message.content
    history[-1][1] = message
    return history


def vqa_solution(history, limit, task_ids):
    # img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
    target_uri = None
    for idx, (Q, A) in enumerate(history[::-1]):
        if isinstance(Q, tuple) or isinstance(A, tuple):
            target_uri = Q[0] if isinstance(Q, tuple) else A[0]
            break
    if target_uri is None:
        history[-1][1] = '似乎没有找到响应询问的图片,请重新输入'
        return history
    # prepare image + question
    image = Image.open(target_uri).convert('RGB')
    text = trans_zh_en(history[-1][0])  # zh to en

    inputs = vqa_processor(image, text, return_tensors="pt")
    out = vqa_model.generate(**inputs)

    ans = vqa_processor.decode(out[0], skip_special_tokens=True)
    print('before trans:', ans)
    ans = trans_en_zh(ans)  # en to zh
    print("Predicted answer:", ans)
    history[-1][1] = ans[:len(ans) // 2]
    return history


def image_generation_solution(history, task_ids):
    # API_URL = "https://api-inference.huggingface.co/models/BAAI/AltDiffusion"  # ZH, seems not working
    # API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1"  # EN
    API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2"
    headers = {"Authorization": f"Bearer {os.environ['HF_API_KEY']}"}

    def query(payload):
        print('payload', payload)
        response = requests.post(API_URL, headers=headers, json=payload)
        print('response', response)
        return response.content

    prompt = history[-1][0]
    prompt = prompt.replace('画', '')
    prompt = prompt.replace('请', '')
    prompt = prompt.replace('给我', '')
    prompt = prompt.replace('我要', '')
    prompt = prompt.replace('我想', '')
    prompt = prompt.replace('帮我', '')
    prompt = prompt.replace('please', '')
    prompt = prompt.replace('draw me', '')
    prompt = prompt.replace('draw a', '')
    prompt = prompt.replace('draw an', '')
    prompt = prompt.replace('draw', '')
    prompt = prompt.replace('for me', '')

    prompt = trans_zh_en(prompt)
    print('prompt for gen (EN):', prompt)

    # better quality
    # prompt += '杰作,高清,8K,4k,充满细节(full details),艺术的,有创意的'  # for ZH
    prompt = 'masterpiece,8K,CG,' + prompt + 'wallpaper,cgsociety,artstation,full details'  # for EN

    image_bytes = query({
        "inputs": f"{prompt}",
    })
    # You can access the image with PIL.Image for example
    image = Image.open(io.BytesIO(image_bytes))
    save_path = 'pics/' + str(len(os.listdir('pics')) + 1) + '.jpg'
    if not os.path.exists('pics'):
        os.makedirs('pics')
    image.save(save_path)
    history[-1][1] = (save_path,)
    return history


def audio_to_text_solution(history, task_ids):
    history_ = [[Q, A] for idx, (Q, A) in enumerate(history) if task_ids[idx] != 0]
    audio_path = ""
    for Q, _ in history_[::-1]:
        if not isinstance(Q[0], str):
            continue
        if Q[0].endswith(".mp3") or Q[0].endswith(".wav") or Q[0].endswith(".mp4") or Q[0].endswith(".mpeg") or Q[
            0].endswith(".m4a"):
            audio_path = Q[0]
            break
    audio_file = open(audio_path, "rb")
    resp = openai.Audio.transcribe("whisper-1", audio_file)

    text = resp['text']
    history[-1][1] = text
    return history


def get_embs(descs):
    return embed_model.encode(descs, convert_to_tensor=True)


def get_sim(desc, task_embs):
    desc_emb = embed_model.encode(desc, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(desc_emb, task_embs)[0]
    return cos_scores


def trans_zh_en(zh_prompt):
    input_ids = trans_tokenizer_zh_en(zh_prompt, return_tensors="pt").input_ids
    outputs_en = trans_model_zh_en.generate(input_ids)
    outputs = trans_tokenizer_zh_en.decode(outputs_en[0], skip_special_tokens=True)
    return outputs


def trans_en_zh(en_prompt):
    input_ids = trans_tokenizer_en_zh(en_prompt, return_tensors="pt").input_ids
    outputs_zh = trans_model_en_zh.generate(input_ids)
    outputs = trans_tokenizer_en_zh.decode(outputs_zh[0], skip_special_tokens=True)
    return outputs


