from django.shortcuts import render

# Create your views here.
# def create_model(request):
import re
import gensim
import numpy
from gensim.parsing.preprocessing import remove_stopwords
import pandas as pd
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import gensim.downloader as api
import gensim.downloader as api
from gensim import corpora
glove_model = None;
#Load dataset and examine dataset, rename columns to questions and answers
question=''
retrieve=''
def getWordVec(word, model):
    samp = model['computer'];
    vec = [0] * len(samp);
    try:
        vec = model[word];
    except:
        vec = [0] * len(samp);
    return (vec)


def getPhraseEmbedding(phrase, embeddingmodel):
    samp = getWordVec('computer', embeddingmodel);
    vec = numpy.array([0] * len(samp));
    den = 0;
    for word in phrase.split():
        # print(word)
        den = den + 1;
        vec = vec + numpy.array(getWordVec(word, embeddingmodel));
    # vec=vec/den;
    # return (vec.tolist());
    return vec.reshape(1, -1)

def user_login(request):
    return render(request, 'question_fetch.html')

def send_data(request):
    if request.method=="POST":
        question1=request.POST['question2']

    df = pd.read_csv("E:/FAQ.csv")
    df.columns = ["questions", "answers"]
    cleaned_sentences = get_cleaned_sentences(df, stopwords=True)
    print(cleaned_sentences);

    print("\n")

    cleaned_sentences_with_stopwords = get_cleaned_sentences(df, stopwords=False)
    print(cleaned_sentences_with_stopwords)
    sentences = cleaned_sentences_with_stopwords
    # sentences=cleaned_sentences

    # Split it by white space
    sentence_words = [[word for word in document.split()]
                      for document in sentences]



    dictionary = corpora.Dictionary(sentence_words)
    for key, value in dictionary.items():
        print(key, ' : ', value)

    import pprint
    bow_corpus = [dictionary.doc2bow(text) for text in sentence_words]
    for sent, embedding in zip(sentences, bow_corpus):
        print(sent)
        print(embedding)

    # question_orig="do I need to learn algorithms to be a data scientist ?";
    question_orig = question1
    question = clean_sentence(question_orig, stopwords=False);
    question_embedding = dictionary.doc2bow(question.split())

    try:
        glove_model = gensim.models.KeyedVectors.load("./glovemodel.mod")
        print("Loaded glove model")
    except:
        glove_model = api.load('glove-twitter-25')
        glove_model.save("./glovemodel.mod")
        print("Saved glove model")

    v2w_model = None;
    try:
        v2w_model = gensim.models.KeyedVectors.load("./w2vecmodel.mod")
        print("Loaded w2v model")
    except:
        v2w_model = api.load('word2vec-google-news-300')
        v2w_model.save("./w2vecmodel.mod")
        print("Saved glove model")

    w2vec_embedding_size = len(v2w_model['computer']);
    glove_embedding_size = len(glove_model['computer']);
    sent_embeddings = [];
    for sent in cleaned_sentences:
        sent_embeddings.append(getPhraseEmbedding(sent, v2w_model));

    question_embedding = getPhraseEmbedding(question, v2w_model);

    print("\n\n", question, "\n", question_embedding)
    max_sim = -1;
    index_sim = -1;
    for index, faq_embedding in enumerate(sent_embeddings):
        # sim=cosine_similarity(embedding.reshape(1, -1),question_embedding.reshape(1, -1))[0][0];
        sim = cosine_similarity(faq_embedding, question_embedding)[0][0];
        print(index, sim, sentences[index])
        if sim > max_sim:
            max_sim = sim;
            index_sim = index;

    print("\n")
    print("Question: ", question)
    print("\n");
    print("Retrieved: ", df.iloc[index_sim, 0])
    retrieve = df.iloc[index_sim, 0]
    print(df.iloc[index_sim, 1])
    # retrieveAndPrintFAQAnswer(question_embedding, bow_corpus, df, sentences)

    return render(request, "question_fetch.html", {'ques': df.iloc[index_sim, 0],'embedding':df.iloc[index_sim, 1]})




# from nltk.stem.lancaster import LancasterStemmer
# st = LancasterStemmer()

def clean_sentence(sentence, stopwords=False):
    sentence = sentence.lower().strip()
    sentence = re.sub(r'[^a-z0-9\s]', '', sentence)
    # sentence = re.sub(r'\s{2,}', ' ', sentence)

    if stopwords:
        sentence = remove_stopwords(sentence)

    # sent_stemmed='';
    # for word in sentence.split():
    #    sent_stemmed+=' '+st.stem(word)
    # sentence=sent_stemmed

    return sentence


def get_cleaned_sentences(df, stopwords=False):
    sents = df[["questions"]];
    cleaned_sentences = []

    for index, row in df.iterrows():
        # print(index,row)
        cleaned = clean_sentence(row["questions"], stopwords);
        cleaned_sentences.append(cleaned);
    return cleaned_sentences;




#word model




def retrieveAndPrintFAQAnswer(question_embedding, sentence_embeddings, FAQdf, sentences):
    max_sim = -1;
    index_sim = -1;
    for index, faq_embedding in enumerate(sentence_embeddings):
        # sim=cosine_similarity(embedding.reshape(1, -1),question_embedding.reshape(1, -1))[0][0];
        sim = cosine_similarity(faq_embedding, question_embedding)[0][0];
        print(index, sim, sentences[index])
        if sim > max_sim:
            max_sim = sim;
            index_sim = index;

    print("\n")
    print("Question: ", question)
    print("\n");
    print("Retrieved: ", FAQdf.iloc[index_sim, 0])
    retrieve=FAQdf.iloc[index_sim, 0]
    print(FAQdf.iloc[index_sim, 1])



