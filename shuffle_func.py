import numpy as np
import spacy
import nltk
_pos = {"noun": ['NN', 'NNS', 'NNP', 'NNPS'], "adj": ['JJ', 'JJR', 'JJS'],
		"verb": ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"], }

class Text_Des:
	def __init__(self, emcoder="en_core_web_sm", shuffle=False):
		'''
		:param text_perturb_fn: the name of text destroy method.
		:param emcoder: 'en_core_web_sm' for efficiency / 'en_core_web_trf' for accuracy.
		:param is_random: shuffle the words or not.
		'''
		self.shuffle = shuffle

		print("Prepare SPACY NLP model!")
		self.nlp = spacy.load(emcoder)
		self.word_tokenize = nltk.word_tokenize
        
    
	def shuffle_nouns_and_verb_adj(self, ex):
		doc = self.nlp(ex)
		tokens = [token.text for token in doc]
		text = np.array(tokens)
		noun_idx = [i for i, token in enumerate(doc) if token.tag_ in _pos["noun"]]
		adjective_idx = [i for i, token in enumerate(doc) if token.tag_ in _pos["adj"]]
		verb_idx = [i for i, token in enumerate(doc) if token.tag_ in _pos["verb"]]
		## Shuffle the nouns of the text
		text[noun_idx] = np.random.permutation(text[noun_idx])
		## Shuffle the adjectives of the text
		text[verb_idx] = np.random.permutation(text[verb_idx])
		text[adjective_idx] = np.random.permutation(text[adjective_idx])
		return " ".join(text)

	def shuffle_all_words(self, ex):
		return " ".join(np.random.permutation(ex.split(" ")))

	def shuffle_allbut_nouns_verb_adj(self, ex):
		doc = self.nlp(ex)
		tokens = [token.text for token in doc]
		text = np.array(tokens)
		noun_adj_idx = [i for i, token in enumerate(doc) if
						token.tag_ in _pos["noun"] + _pos["adj"] + _pos["verb"]]
		## Finding adjectives

		else_idx = np.ones(text.shape[0])
		else_idx[noun_adj_idx] = 0

		else_idx = else_idx.astype(bool)
		## Shuffle everything that are nouns or adjectives
		text[else_idx] = np.random.permutation(text[else_idx])
		return " ".join(text)
