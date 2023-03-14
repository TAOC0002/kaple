import numpy as np
import pandas as pd
import json
import pickle
import itertools
from sklearn.utils import shuffle
from sqlalchemy import create_engine, Column, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.types import DateTime
from sqlalchemy.interfaces import PoolListener

engine = create_engine('sqlite:////media/ntu/volume1/home/s122md301_06/K-Adapter/data/patent-sim/patent.db') #, echo=True)
Base = declarative_base()

class MyListener(PoolListener):
    def connect(self, dbapi_con, con_record):
        dbapi_con.execute("PRAGMA temp_store = 2")


class Patent(Base):
    __tablename__ = 'patent'
    __table_args__ = {'extend_existing': True}

    id = Column(String, primary_key=True)
    category = Column(String)
    pub_number = Column(String)
    app_number = Column(String)
    pub_date = Column(DateTime)
    title = Column(String)
    abstract = Column(String)
    description = Column(String)
    claims = Column(String)

    def to_dict(self):
        d = {}
        d['id'] = self.id
        d['category'] = self.category
        d['pub_date'] = self.pub_date
        d['title'] = self.title
        d['abstract'] = self.abstract
        d['description'] = self.description
        d['claims'] = self.claims
        d['cited_patents'] = [c.cited_pat for c in self.cited_patents]
        return d

    def __repr__(self):
        return "<Patent(id='%s', category='%s', pub_number='%s', app_number='%s',\
                        pub_date='%s', title='%s', abstract='%s', description='%s',\
                        claims='%s', cited_patents='%s')>" %(self.id, self.category,
                        self.pub_number, self.app_number, str(self.pub_date), self.title,
                        self.abstract, self.description, self.claims, self.cited_patents)


def load_session():
    metadata = Base.metadata
    Session = sessionmaker(bind=engine)
    session = Session()
    return session


def make_full_section_corpus():
    session = load_session()
    corpus_abstract = {}
    corpus = np.load('corpus.npy', encoding="latin1").item()
    for id_ in corpus.keys():
        pat = session.query(Patent).filter(Patent.id == id_)[0]
        corpus_abstract[pat.id] = (pat.title + '\n' + pat.abstract)
    np.save('corpus_abstract.npy', corpus_abstract)


def make_combis(target_ids, random_ids, cited_ids, dupl_ids):
    id_dict = {}
    id_dict['cited'] = [(key, vals[i]) for key, vals in cited_ids.items() for i in range(len(vals))]
    id_dict['duplicate'] = [(key, vals[i]) for key, vals in dupl_ids.items() for i in range(len(vals))]
    id_dict['random'] = list(itertools.product(cited_ids, random_ids))
    return id_dict


def collate_examples():
    corpus_abstract = np.load('corpus_abstract.npy', encoding="latin1", allow_pickle=True).item()
    target_ids = np.load('target_ids.npy', encoding="latin1")
    random_ids = np.load('random_ids.npy', encoding="latin1")
    dupl_ids = np.load('dupl_ids.npy', encoding="latin1", allow_pickle=True).item()
    cited_ids = np.load('cited_ids.npy', encoding="latin1", allow_pickle=True).item()
    id_dict = make_combis(target_ids, random_ids, cited_ids, dupl_ids)

    examples = {'index':[], 'text':[], 'text_b':[], 'label':[]}
    index = 0
    label_name = ['random', 'cited']   # 0 -> random, 1 -> cited
    for label_num in range(len(label_name)):
        combis = id_dict[label_name[label_num]]
        for combi in combis:
            examples['index'].append(index)
            examples['text'].append(corpus_abstract[combi[0]])
            examples['text_b'].append(corpus_abstract[combi[1]])
            examples['label'].append(label_num)
            index += 1

    examples = pd.DataFrame(data=examples)
    print(examples.shape)
    examples = examples.iloc[-80000:,:]
    examples = shuffle(examples, random_state=42)
    examples = examples.iloc[-1000:,:]
    # print(examples.shape)

    valid_file = open("../patent-sim-compact/valid.jsonl", "w")
    train_file = open("../patent-sim-compact/train.jsonl", "w")
    valid_size = examples.shape[0] // 10
    valid = examples[-valid_size:].to_json(orient="records")
    train = examples[:-valid_size].to_json(orient="records")
    valid_file.write(json.dumps(json.loads(valid), indent=4))
    train_file.write(json.dumps(json.loads(train), indent=4))
    valid_file.close()
    train_file.close()


def check_json_format(input_file):
    with open(input_file) as f:
        data = json.load(f)
        for line in data:           
            index = line['index']
            prior = line['text']
            claim = line['text_b']
            label = line['label']
            print(index)
            print(prior)
            print(claim)
            print(label)
            break


def eda():
    data = pd.read_json('valid.jsonl', orient='records')
    print(data.columns)
    print(data.shape)
    print(sum(data['label']))

def kpar_test_set_construction(size):
    # inverse dictionary
    corpus_abstract = np.load('corpus_abstract.npy', encoding="latin1", allow_pickle=True).item()
    inv_map = {v: k for k, v in corpus_abstract.items()}
    with open('../patent-sim-compact/train.jsonl') as f:
        train = json.load(f)
    with open('../patent-sim-compact/valid.jsonl') as g:
        valid = json.load(g)

    # find text_a which we haven't encountered before
    text_a = set()
    text_b = set()
    for line in train:
        text_a.add(inv_map[line['text']])
        text_b.add(inv_map[line['text_b']])
    for line in valid:
        text_a.add(inv_map[line['text']])
        text_b.add(inv_map[line['text_b']])

    print('Number of patent application (text_a) in training and validation set:', len(text_a))
    print('Number of prior art (text_b) in training and validation set:', len(text_b))

    cited_ids = np.load('cited_ids.npy', encoding="latin1", allow_pickle=True).item()
    test_ids = {text: text_b_list for text, text_b_list in cited_ids.items() if text not in text_a}
    for test_id in test_ids:
        text_b_list = [b for b in test_ids[test_id] if b in text_b]
        test_ids[test_id] = text_b_list
    test_ids = dict(sorted(test_ids.items(), key=lambda item: len(item[1]), reverse=True))
    test_ids = dict(itertools.islice(test_ids.items(),size))
    print('Construting test set with size', size, 'and their ground truth (known) in the corpus..')

    array = [{'key': i, 'text': corpus_abstract[i], 'ground_truth': test_ids[i]} for i in test_ids]
    test_file = open("../patent-sim-compact/test.jsonl", "w")
    test_file.write(json.dumps(array, indent = 4))
    test_file.close()
    text_a.update(text_b)
    with open('../patent-sim-compact/corpus_pool.pkl', 'wb') as c:
        pickle.dump(text_a, c)

if __name__ == "__main__":
    # make_full_section_corpus()
    # collate_examples()
    # check_json_format('valid.jsonl')
    # eda()
    kpar_test_set_construction(size=100)