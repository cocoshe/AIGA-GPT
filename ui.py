import gradio as gr
from solutions.utils import *
from MultiModals import predict
import os


def add_text(history, text):
    history = history + [(text, None)]
    return history, ""


def add_file(history, file):
    history = history + [((file.name,), None)]
    return history


def change_openaiKey(new_key):
    os.environ['OPENAI_API_KEY'] = new_key
    # print('OpenAI Key changed to: {}'.format(OPENAI_KEY))
    return None


def bot(history):
    req = ""
    if len(history) == 1:
        req = init_prompt()
    for Q, A in history[:-1]:
        req += 'Q: "{}"这个问题是哪一类任务?\nA: {}\n'.format(Q, A)
    req += 'Q: "{}"这个问题是哪一类任务?\n'.format(history[-1][0])
    req += 'A: '
    history = predict(req, task_embeddings, history, limit=limit, task_ids=task_ids)
    return history


with gr.Blocks() as demo:
    openaiKeyInput = gr.Textbox(label='OpenAI key', type='password',
                                placeholder='sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
    openaiKey = gr.State(openaiKeyInput)
    os.environ['OPENAI_API_KEY'] = openaiKey
    chatbot = gr.Chatbot([], elem_id="chatbot").style(height=650)
    task_ids = []

    with gr.Row():
        with gr.Column(scale=0.70):
            txt = gr.Textbox(
                show_label=False,
                placeholder="Enter text and press enter, or upload an image",
            ).style(container=False)
        with gr.Column(scale=0.15, min_width=0):
            sendBtn = gr.Button("Send")
        with gr.Column(scale=0.15, min_width=0):
            uploadBtn = gr.UploadButton("📁", file_types=["image", "video", "audio"])
    clearBtn = gr.Button("Clear")
    txt.submit(add_text, [chatbot, txt], [chatbot, txt]).then(
        bot, chatbot, chatbot
    )
    sendBtn.click(add_text, [chatbot, txt], [chatbot, txt]).then(
        bot, chatbot, chatbot
    )
    uploadBtn.upload(add_file, [chatbot, uploadBtn], [chatbot]).then(
        bot, chatbot, chatbot
    )
    clearBtn.click(lambda: None, None, chatbot, queue=False)
    clearBtn.click(lambda: [], None, task_ids, queue=False)
    openaiKeyInput.change(change_openaiKey, inputs=openaiKeyInput,)
