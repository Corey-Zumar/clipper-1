import sys
import os
import gensim

from single_proc_utils import ModelBase

MODEL_RELATIVE_PATH = "#TODO: FILL IN"
DICTIONARY_RELATIVE_PATH = "#TODO: FILL IN"

class LDAModel(ModelBase):

    def __init__(self, model_data_path):
        ModelBase.__init__(self)

        model_path = os.path.join(model_data_path, MODEL_RELATIVE_PATH)
        dictionary_path = os.path.join(model_data_path, DICTIONARY_RELATIVE_PATH)

        assert os.path.exists(model_path)
        assert os.path.exists(dictionary_path)

        self.word_ids_dict = gensim.corpora.Dictionary.load_from_text(dictionary_path)
        self.model = gensim.models.ldamulticore.LdaMulticore.load(model_path, mmap='r')

    def predict(self, inputs):
        """
        Parameters
        ----------
        inputs : list
            A list of documents, represented as strings

        Returns
        ----------
        list 
            A list of topic ids. The output at index `i`
            is the index of the most relevant topic predicted
            for the document at input index `i`.
        """
        outputs = []
        for input_doc in inputs:
            doc_bow = self.word_ids_dict.doc2bow(input_doc.split())
            topic_dist = self.model[doc_bow]

            max_prob = 0
            best_topic = -1
            for topic_probability in topic_dist:
                if topic_probability[1] > max_prob:
                    best_topic = topic_probability[0]
                    max_prob = topic_probability[1]

            outputs.append(str(best_topic))

        return outputs
        

