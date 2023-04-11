from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')

sentences1 = '请给我讲个笑话'
sentences2 = ['来个笑话', '能不能陪我出去吃饭', '饿了']
#
# sentences1 = ['The cat sits outside',
#              'A man is playing guitar',
#              '这部剧很不错啊']
#
# sentences2 = ['The dog plays in the garden',
#               'A woman watches TV',
#               '这部电影很棒']

#Compute embedding for both lists
embeddings1 = model.encode(sentences1, convert_to_tensor=True)
embeddings2 = model.encode(sentences2, convert_to_tensor=True)
sim = util.pytorch_cos_sim(embeddings1, embeddings2)
print(sim)





