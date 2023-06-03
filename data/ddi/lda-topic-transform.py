import pandas as pd
import gensim
from gensim.utils import simple_preprocess
import gensim.corpora as corpora
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
def remove_stopwords(texts):
    stop_words = stopwords.words('english')
    return [[word for word in simple_preprocess(str(doc)) 
             if word not in stop_words] for doc in texts]

def lda(filenames, new_filenames, num_topics=5):
    assert len(filenames) == len(new_filenames)
    identifiers, sentence_words, lengths, sentences = [], [], [], []
    for file in filenames:
        data = pd.read_csv(file, sep='\t', header=0)
        data_index_to_list = data['index'].tolist()
        identifiers.extend(data_index_to_list)
        lengths.append(len(data_index_to_list))

        # Preprocess sentence
        sentences.extend(data.sentence.to_list())
        sentence_to_words = list(sent_to_words(data.sentence))
        # remove stop words
        sentence_to_words = remove_stopwords(sentence_to_words)
        sentence_words.extend(sentence_to_words)

    # Create Dictionary
    id2word = corpora.Dictionary(sentence_words)
    # Create CorpusTerm Document Frequency
    corpus = [id2word.doc2bow(text) for text in sentence_words]
    LDA = gensim.models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word=id2word, passes=1, random_state=42, eval_every=None)
    print('Printing LDA topics')

    # Predict topics
    topics = [max(composition, key=lambda item: item[1])[0] for composition in LDA[corpus]]
    count = 0
    for file in new_filenames:
        new_data = pd.DataFrame(list(zip(identifiers[:lengths[count]], sentences[:lengths[count]], topics[:lengths[count]])), columns =['index', 'sentence', 'label'])
        print('Label info for {}'.format(file))
        print(new_data['label'].value_counts())
        new_data.to_csv(file, sep='\t', index=False, header=False)
        count += 1
    
if __name__ == "__main__":
    filenames = ['train.tsv', 'dev.tsv', 'test.tsv']
    new_filenames = ['new-train.tsv', 'new-dev.tsv', 'new-test.tsv']
    lda(filenames=filenames, new_filenames=new_filenames)