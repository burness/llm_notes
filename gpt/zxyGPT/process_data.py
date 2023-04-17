import logging 
import sys
import re
import torch
import nltk
import json
import time
import opencc
import argparse
import multiprocessing
from gensim.corpora import WikiCorpus
from .data import indexed_dataset
logging.basicConfig(format="%(asctime)s: %(levelname)s: %(message)s", level=logging.INFO)

try:
    import nltk
    nltk_available = True
except ImportError:
    nltk_available = False


class CustomLanguageVars(nltk.tokenizer.punkt.PunktLanguageVars):
    _period_context_fmt = r"""
        \S*                          # some word material
        %(SentEndChars)s             # a potential sentence ending
        \s*                       #  <-- THIS is what I changed
        (?=(?P<after_tok>
            %(NonWord)s              # either other punctuation
            |
            (?P<next_tok>\S+)     #  <-- Normally you would have \s+ here
        ))"""
    
class IdentitySplitter(object):
    def tokenize(self, *text):
        return text

class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        Encoder.tokenizer = build_tokenizer(self.args)
        if self.args.split_sentences:
            if not nltk_available:
                logging.info("nltk is not available to split sentences.")
                exit()
            splitter = nltk.load("tokenizers/punkt/english.pickle")
            if self.args.keep_newlines:
                Encoder.splitter = nltk.tokenize.punkt.PunltSetenceTokenizer(
                    train_text = splitter._paras,
                    lang_vars = CustomLanguageVars()
                )
            else:
                Encoder.splitter = splitter
        else:
            Encoder.splitter = IdentitySplitter()
    
    def encode(self, json_line):
        data = json.loads(json_line)
        ids = {}
        for key in self.args.json_keys:
            text = data[key]
            doc_ids = []
            for sentence in Encoder.spliite.tokenize(text):
                sentence_ids = Encoder.tokenizer.tokenize(sentence)
                if len(sentence_ids) > 0:
                    doc_ids.append(sentence_ids)
            if len(doc_ids) > 0 and self.args.append_eod:
                doc_ids[-1].append(Encoder.tokenizer.eod)
            id[key] = doc_ids
        return ids, len(json_line)

class DataProcess(object):
    def __init__(self, zhwiki_bz2_path, zhwiki_json_path):
        self.zhwiki_bz2_path = zhwiki_bz2_path
        self.zhwiki_json_path = zhwiki_json_path
        self.converter = opencc.OpenCC("t2s.json")
        self.args = get_args()

    def __extract_content(self):
        with open(self.zhwiki_json_path, "w") as fwrite:
            wiki = WikiCorpus(self.zhwiki_bz2_path, dictionary={})
            id = 0
            for text in wiki.get_texts():
                # line = f{{\"id\": {id}, \"text\": {text}\}}” + "\n"
                id += 1
                if id % 10000 == 0:
                    logging.info("save {0} articles".format(id))
                text = "".join(text)
                text = self.converter.convert(text)
                text = re.sub(r"[^\u4e00-\u9fa5]", "", text)
                line = f"{{\"id\":{id}, \"text\": {text} }}" +"\n"
                # print(line)
                fwrite.write(line)
        
    def transform(self, ):
        startup_start = time.time()
        fread = open(self.args.input, "r", encoding="utf-8")
        if nltk.available() and self.args.split_setences:
            nltk.download("punkt", quiet=True)
        
        encoder = Encoder(self.args)
        tokenizer = build_tokenizer(self.args)
        pool = multiprocessing.Pool(self.args.workers, initializer=encoder.initializer)
        encoded_docs = pool.imap(encoder.encode, fread, 25)

        level = "docment"
        if self.args.split_sentences:
            level = "sentence"
        
        logging.info(f"Vocab size: {tokenizer.vocab_size}, Output prefix: {self.args.output_prefix}")
        
        output_bin_files = {}
        output_idx_files = {}
        builders = {}
        for key in self.args.json_keys:
            output_bin_files[key] = "{}_{}_{}".format(self.args.output_prefix, key, level)
            output_idx_files[key] = "{}_{}_{}".format(self.args.output_prefix, key, level)
            builders[key] = indexed_dataset.make_builder(output_bin_files[key],
                                                         impl=self.args.dataset_impl, 
                                                         vocab_size = tokenizer.vocab_size)
        
        startup_end = time.time()
        proc_start = time.time()
        total_bytes_processed = 0
        logging.info("time to startup: {}".format(startup_end-startup_start))

        for i, (doc, bytes_processed) in enumerate(encoded_docs, start = 1):
            total_bytes_processed += bytes_processed
            for key, sentences in doc.items():
                if len(sentences) == 0:
                    continue
                for sentence in sentences:
                    builders[key].add_item(torch.IntTensor(sentence))
                builders[key].end_document()
            if i % self.args.log_iterval == 0:
                current = time.time()
                elapsed = current-proc_start
                mbs = total_bytes_processed / elapsed / 1024**2
                logging.info(f"Processed {i} documents, ({i/elapsed} docs/s, {mbs} MB/s). ")
        
        for key in self.args.json_keys:
            builders[key].finalize(output_idx_files[key])

        


    


    def run(self):
        self.__extract_content()

def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title="input_data")
    group.add_argument("--input", type=str, required=True, help="Path to input JSON")
    group.add_argument("--json-keys", nargs="*", default=["text"], help="space separate listed of keys to extract from json")
    group.add_argument("--split-sentences", action="store_true", help="Split docments into sentences.")
    group.add_argument("--keep-newlines", action="store_true", help="Keep newlines beween senences when splitting")
    
    group = parser.add_argument_group(title="tokenizer")
    group.add_argument("--tokenizer-type", type=str, required=True, choices=["BertWordPieceLowerCase", "BertWordPieceCase", "GPT2BPETokenizer"], help="what type of tokenizer to use")
    group.add_argument("--vocab-file", type=str, default=None, help="Path to vocab file")
    group.add_argument("--merge-file", type=str, default=None, help="Path to the BPE merge file(if necessary).")
    group.add_argument("--append-eod", action="store_true", help="Append an <eod> token to the end of a document.")

    group = parser.add_argument_group(title="output data")
    group.add_argument("--output-prefix", type=str, required=True, help = "Path to binary output file without suffix")
    group.add_argument("--dataset-impl", type=str, default="mmap", choices=["lazy", "cached", "mmap"])

    group = parser.add_argument_group(title="runtime")
    group.add_argument("--workers", type=int, default=1, help="Number of worker processes to lauch")
    group.add_argument("--log-interval", type=int, default=100, help="interval between progress updates")

    args = parser.parse_args()
    args.keep_empty = False

    if args.tokenizer_type.lower().startswith("bert"):
        if not args.split_sentences:
            logging.info("Bert tokenizer detecter are you sure you don't want to split sentences?")
    
    args.rank = 0
    args.make_vocab_size_divisible_by = 128
    args.tensor_model_parallel_size = 1
    args.vocab_extra_ids = 0

    return args



if __name__ == "__main__":
    zhwiki_bz2_path = "/work/gpt/corpus/zhwiki-20221220-pages-articles-multistream.xml.bz2"
    zhwiki_json_path = "/work/gpt/corpus/zhwiki-20221220-pages-articles-multistream.json"
    data_processor = DataProcess(zhwiki_bz2_path, zhwiki_json_path)
    data_processor.run()

