import logging 
import sys
import opencc
from gensim.corpora import WikiCorpus
logging.basicConfig(format="%(asctime)s: %(levelname)s: %(message)s", level=logging.INFO)
class DataProcess(object):
    def __init__(self, zhwiki_bz2_path, zhwiki_json_path):
        self.zhwiki_bz2_path = zhwiki_bz2_path
        self.zhwiki_json_path = zhwiki_json_path
        self.converter = opencc.OpenCC("t2s.json")

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
                line = f"{{\"id\":{id}, \"text\": {text} }}" +"\n"
                fwrite.write(line)

    def run(self):
        self.__extract_content()


if __name__ == "__main__":
    zhwiki_bz2_path = "/work/gpt/corpus/zhwiki-20221220-pages-articles-multistream.xml.bz2"
    zhwiki_json_path = "/work/gpt/corpus/zhwiki-20221220-pages-articles-multistream.json"
    data_processor = DataProcess(zhwiki_bz2_path, zhwiki_json_path)
    data_processor.run()

