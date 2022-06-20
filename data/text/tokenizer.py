from typing import Union
import re

from phonemizer.phonemize import phonemize
from phonemizer.backend import EspeakBackend

from data.text.symbols import all_phonemes, _punctuations


class Tokenizer:
    
    def __init__(self, start_token='>', end_token='<', pad_token='/', add_start_end=True, alphabet=None,
                 model_breathing=True):
        if not alphabet:
            self.alphabet = all_phonemes
        else:
            self.alphabet = sorted(list(set(alphabet)))  # for testing
        self.idx_to_token = {i: s for i, s in enumerate(self.alphabet, start=1)}
        self.idx_to_token[0] = pad_token
        self.token_to_idx = {s: [i] for i, s in self.idx_to_token.items()}
        self.vocab_size = len(self.alphabet) + 1
        self.add_start_end = add_start_end
        if add_start_end:
            self.start_token_index = len(self.alphabet) + 1
            self.end_token_index = len(self.alphabet) + 2
            self.vocab_size += 2
            self.idx_to_token[self.start_token_index] = start_token
            self.idx_to_token[self.end_token_index] = end_token
        self.model_breathing = model_breathing
        if model_breathing:
            self.breathing_token_index = self.vocab_size
            self.token_to_idx[' '] = self.token_to_idx[' '] + [self.breathing_token_index]
            self.vocab_size += 1
            self.breathing_token = '@'
            self.idx_to_token[self.breathing_token_index] = self.breathing_token
            self.token_to_idx[self.breathing_token] = [self.breathing_token_index]
    
    def __call__(self, sentence: str) -> list:
        sequence = [self.token_to_idx[c] for c in sentence]  # No filtering: text should only contain known chars.
        sequence = [item for items in sequence for item in items]
        if self.model_breathing:
            sequence = [self.breathing_token_index] + sequence
        if self.add_start_end:
            sequence = [self.start_token_index] + sequence + [self.end_token_index]
        return sequence
    
    def decode(self, sequence: list) -> str:
        return ''.join([self.idx_to_token[int(t)] for t in sequence])


class Phonemizer:
    def __init__(self, language: str, with_stress: bool, njobs=4):
        self.language = language
        self.njobs = njobs
        self.with_stress = with_stress
        self.special_hyphen = '—'
        self.punctuation = ';:,.!?¡¿—…"«»“”'
        self._whitespace_re = re.compile(r'\s+')
        self._whitespace_punctuation_re = re.compile(f'\s*([{_punctuations}])\s*')
    
    def __call__(self, text: Union[str, list], with_stress=None, njobs=None, language=None) -> Union[str, list]:
        language = language or self.language
        njobs = njobs or self.njobs
        with_stress = with_stress or self.with_stress
        # phonemizer does not like hyphens.
        text = self._preprocess(text)
        print(" ################## PHONEM LANG #####################")
        print(language)
        print(" ################## PHONEM LANG #####################")
        backend = EspeakBackend(language)
        phonemes = backend.phonemize(text, strip=True)
        print(" ################## PHONEMES #####################")
        print(phonemes)
        print(" ################## PHONEMES #####################")

        # phonemes = phonemize(text,
        #                      language=language,
        #                      backend='espeak',
        #                      strip=True,
        #                      preserve_punctuation=True,
        #                      with_stress=with_stress,
        #                      punctuation_marks=self.punctuation,
        #                      njobs=njobs,
        #                      language_switch='remove-flags')
        return self._postprocess(phonemes)
    
    def _preprocess_string(self, text: str):
        text = text.replace('-', self.special_hyphen)
        return text
    
    def _preprocess(self, text: Union[str, list]) -> Union[str, list]:
        if isinstance(text, list):
            return [self._preprocess_string(t) for t in text]
        elif isinstance(text, str):
            return self._preprocess_string(text)
        else:
            raise TypeError(f'{self} input must be list or str, not {type(text)}')
    
    def _collapse_whitespace(self, text: str) -> str:
        text = re.sub(self._whitespace_re, ' ', text)
        return re.sub(self._whitespace_punctuation_re, r'\1', text)
    
    def _postprocess_string(self, text: str) -> str:
        text = text.replace(self.special_hyphen, '-')
        text = ''.join([c for c in text if c in all_phonemes])
        text = self._collapse_whitespace(text)
        text = text.strip()
        return text
    
    def _postprocess(self, text: Union[str, list]) -> Union[str, list]:
        if isinstance(text, list):
            return [self._postprocess_string(t) for t in text]
        elif isinstance(text, str):
            return self._postprocess_string(text)
        else:
            raise TypeError(f'{self} input must be list or str, not {type(text)}')
