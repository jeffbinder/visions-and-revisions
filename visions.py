import json
import math
import pickle
import random
import re

import numpy
import scipy
from scipy import sparse
import torch
from transformers import BertTokenizer, BertForMaskedLM
from transformers import DistilBertTokenizer, DistilBertForMaskedLM
from transformers import RobertaTokenizer, RobertaForMaskedLM
from transformers import GPT2Tokenizer, GPT2LMHeadModel

#import logging
#logging.basicConfig(level=logging.INFO)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = None
model = None
loaded_model_type = None
loaded_model_path = None

from nltk.corpus import stopwords
from nltk.corpus import cmudict
dictionary = cmudict.dict()
from g2p_en import G2p
g2p = G2p()

m = torch.nn.Softmax(dim=0)

re_word = re.compile(r"[a-zA-Z' ]+")
re_vowel = re.compile(r"[aeiouy]")
def get_pron(tok):
    tok = tokenizer.convert_tokens_to_string([tok])
    if tok.startswith('##'):
        tok = tok[2:]
    if tok.startswith(' '):
        tok = tok[1:]
    if not re_word.match(tok):
        # Punctuation
        return []
    if tok in dictionary:
        pron = dictionary[tok][0]
    else:
        # Word not in CMU dict: guess using g2p_en
        pron = g2p(tok)
    return pron

def get_meter(pron):
    if pron == []:
        return 'p'
    meter = ''
    for ph in pron:
        # We ignore stress levels in favor of poetic scansion
        if ph[-1].isdigit():
            meter += 'u' if ph[-1] == '0' else '-'
    return meter

def get_rhyme(pron):
    if pron == []:
        return 'p'
    rhyme = ''
    for ph in reversed(pron):
        rhyme = ph.replace('1', '').replace('2', '') + rhyme
        if ph[-1].isdigit() and int(ph[-1]) > 0:
            break
    return rhyme

def is_word_piece(model, tok):
    if model.startswith('bert') or model.startswith('distilbert'):
        return tok.startswith('##')
    elif model.startswith('roberta') or model.startswith('gpt2'):
        tok = tokenizer.convert_tokens_to_string([tok])
        return re_word.match(tok) and not tok.startswith(' ')
def join_word_pieces(toks):
    word = ''
    for tok in toks:
        if tok.startswith('##'):
            tok = tok[2:]
        word += tok
    return word

def is_full_word(model_type, tok):
    if model_type.startswith('bert') or model_type.startswith('distilbert'):
        return re_word.match(tok) and not tok.startswith('##')
    elif model_type.startswith('roberta') or model.startswith('gpt2'):
        tok = tokenizer.convert_tokens_to_string([tok])
        return re_word.match(tok) and tok.startswith(' ')

def is_punctuation(tok):
    if tok == '[MASK]':
        return False
    tok = tokenizer.convert_tokens_to_string([tok])
    return not re_word.match(tok)

# Scan a text to determine spacing and capitalization so that they can be
# preserved after detokenization.
def scan_tokenization(model, text, toks):
    bert = model.startswith('bert') or model.startswith('distilbert')
    spacing = []
    capitalization = []
    char_idx = 0
    tok_idx = 0
    tok_char_idx = 0
    current_spacing = ''
    current_capitalization = None
    while char_idx < len(text):
        char = text[char_idx]
        word_piece = False
        try:
            tok = toks[tok_idx]
            if is_word_piece(model, tok):
                tok = join_word_pieces([tok])
                word_piece = True
        except IndexError:
            tok = ''
        try:
            tok_char = tok[tok_char_idx]
        except IndexError:
            tok_char = ''
        if not char.isspace():
            if tok_char_idx == 0:
                if char.isupper():
                    current_capitalization = 'upper_ambiguous'
                else:
                    current_capitalization = 'lower'
            elif current_capitalization in ('upper_ambiguous', 'upper_all'):
                if char.isupper():
                    current_capitalization = 'upper_all'
                else:
                    current_capitalization = 'upper_initial'
            char_idx += 1
            tok_char_idx += 1
            if tok_char_idx == len(tok):
                tok_idx += 1
                tok_char_idx = 0
                if not word_piece:
                    spacing.append(current_spacing)
                    capitalization.append(current_capitalization)
                    current_spacing = ''
                    current_capitalization = None
        elif tok_char_idx == 0:
            current_spacing += char
            char_idx += 1
    spacing.append(current_spacing)
    return (spacing, capitalization)
    
def detokenize(model, toks, spacing, capitalization, html=False):
    text = ''
    i = 0
    j = 0
    current_capitalization = None
    while i < len(toks):
        tok = toks[i]
        if is_word_piece(model, tok):
            tok = join_word_pieces([tok])
            if current_capitalization == 'upper_all':
                tok = tok.upper()
        else:
            current_spacing = spacing[j]
            if html:
                current_spacing = current_spacing.replace(' ', '&nbsp;')
            text += current_spacing
            current_capitalization = capitalization[j]
            if current_capitalization in ('upper_initial', 'upper_ambiguous'):
                if tok.startswith('<span'):
                    # Special case for HTML visualization
                    i1 = tok.index('>')+1
                    tok = tok[:i1] + tok[i1].upper() + tok[i1+1:]
                elif tok[0] == '<' and tok[-1] == '>':
                    # Special case for tokens marked as just modified
                    tok = tok[0] + tok[1].upper() + tok[2:]
                else:
                    tok = tok[0].upper() + tok[1:]
            elif current_capitalization == 'upper_all':
                tok = tok.upper()
            j += 1
        text += tok
        i += 1
    text += spacing[-1]
    return text

def create_meter_dict(model_type):
    print("Generating " + model_type + '_meter_dict.pkl')
    vocab = tokenizer.get_vocab()
    vocab_size = len(vocab)
    meter_dict = {}
    word_pieces = torch.zeros([vocab_size])
    for tok in vocab:
        i = vocab[tok]
        pron = get_pron(tok)
        meter = get_meter(pron)
        if meter not in meter_dict:
            meter_dict[meter] = torch.zeros([vocab_size])
        meter_dict[meter][i] = 1.0
        if is_word_piece(model_type, tok):
            word_pieces[i] = 1.0

    pickle.dump((word_pieces, meter_dict),
                open(model_type + '_meter_dict.pkl', 'wb'))
    
def create_rhyme_matrix(model_type):
    print("Generating " + model_type + '_rhyme_matrix.pkl')
    vocab = tokenizer.get_vocab()
    vocab_size = len(vocab)
    rhyme_matrix = sparse.lil_matrix((vocab_size, vocab_size))
    rhymable_words = torch.zeros([vocab_size])
    rhyme_groups = {}
    for tok in vocab:
        i = vocab[tok]
        pron = get_pron(tok)
        rhyme = get_rhyme(pron)
        if rhyme not in rhyme_groups:
            rhyme_groups[rhyme] = []
        rhyme_groups[rhyme].append((i, pron))
    for rhyme in rhyme_groups:
        if len(rhyme_groups[rhyme]) < 2:
            continue
        for i, pron1 in rhyme_groups[rhyme]:
            rhymable = False
            for j, pron2 in rhyme_groups[rhyme]:
                # Words with identical pronunciations can't be used as rhymes
                if pron1 != pron2:
                    rhyme_matrix[i,j] = 1.0
                    rhymable = True
            if rhymable:
                rhymable_words[i] = 1.0

    rhyme_matrix = sparse.csc_matrix(rhyme_matrix)
    pickle.dump((rhymable_words, rhyme_matrix), open(model_type + '_rhyme_matrix.pkl', 'wb'))
    
vocab = None
vocab_size = None
meter_dict = {}
word_pieces = None
rhymable_words = None
rhyme_matrix = None
rhyme_tensors = {}
rhyme_and_meter_loaded = None
def initialize_rhyme_and_meter(model, meter=False, rhymes=False):
    global vocab, vocab_size, word_pieces, meter_dict, rhymable_words, rhyme_matrix, rhyme_and_meter_loaded
    if rhyme_and_meter_loaded == model:
        return
    else:
        rhyme_and_meter_loaded = model
    vocab = tokenizer.get_vocab()
    vocab_size = len(vocab)
    if meter:
        try:
            f = open(model + '_meter_dict.pkl', 'rb')
        except FileNotFoundError:
            create_meter_dict(model)
            f = open(model + '_meter_dict.pkl', 'rb')
        word_pieces, meter_dict = pickle.load(f)
    else:
        try:
            f = open(model + '_meter_dict.pkl', 'rb')
        except FileNotFoundError:
            create_meter_dict(model)
            f = open(model + '_meter_dict.pkl', 'rb')
        word_pieces, _ = pickle.load(f)
    if rhymes:
        global rhyme_matrix
        try:
            f = open(model + '_rhyme_matrix.pkl', 'rb')
        except FileNotFoundError:
            create_rhyme_matrix(model)
            f = open(model + '_rhyme_matrix.pkl', 'rb')
        rhymable_words, rhyme_matrix = pickle.load(f)

def initialize_model(model_type, model_path):        
    global tokenizer, model, loaded_model_type, loaded_model_path, bos_token, eos_token
    if loaded_model_type != model_type or loaded_model_path != model_path:
        loaded_model_type = model_type
        loaded_model_path = model_path
        if model_type.startswith('distilbert'):
            tokenizer = DistilBertTokenizer.from_pretrained(model_path or model_type)
            model = DistilBertForMaskedLM.from_pretrained(model_path or model_type)
            bos_token = '[CLS]'
            eos_token = '[SEP]'
        if model_type.startswith('bert'):
            tokenizer = BertTokenizer.from_pretrained(model_path or model_type)
            model = BertForMaskedLM.from_pretrained(model_path or model_type)
            bos_token = '[CLS]'
            eos_token = '[SEP]'
        if model_type.startswith('roberta'):
            tokenizer = RobertaTokenizer.from_pretrained(model_path or model_type)
            model = model = RobertaForMaskedLM.from_pretrained(model_path or model_type)
            bos_token = tokenizer.bos_token
            eos_token = tokenizer.eos_token
        model.to(device)
        model.eval()

# Computes the model's predictions for a text with a given set of ranges
# masked by single mask tokens.
def compute_replacement_probs_for_masked_tokens(model, tokenized_text,
                                                masked_indices):
    n = len(masked_indices)
    dim = [vocab_size] * n

    tokenized_text = tokenized_text.copy()
    shift = 0
    for i1, i2 in masked_indices:
        i1 -= shift
        i2 -= shift
        shift += (i2 - i1)
        tokenized_text[i1:i2+1] = ['[MASK]']
        
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens]).to(device)

    with torch.no_grad():
        outputs = model(tokens_tensor)
        predictions = outputs[0]

    replacement_probs = [None] * n
    shift = 0
    for k, (i1, i2) in enumerate(masked_indices):
        i1 -= shift
        i2 -= shift
        shift += (i2 - i1)
        replacement_probs[k] = [predictions[0, i1, :]]

    return replacement_probs

# Computes the model's predictions for a text with a given set of ranges
# masked.
def compute_probs_for_masked_tokens(model, tokenized_text, masked_indices):
    n = len(masked_indices)
    dim = [vocab_size] * n

    multipart_words = False
    tokenized_text = tokenized_text.copy()
    for i1, i2 in masked_indices:
        if i2 > i1:
            multipart_words = True
        tokenized_text[i1:i2+1] = ['[MASK]'] * (i2 - i1 + 1)

    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens]).to(device)

    with torch.no_grad():
        outputs = model(tokens_tensor)
        predictions = outputs[0]

    probs = [None] * n
    for k, (i1, i2) in enumerate(masked_indices):
        word_probs = []
        for i in range(i1, i2+1):
            word_probs.append(predictions[0, i, :])
        probs[k] = word_probs

    if multipart_words:
        # We need to compute a separate probability with only one mask token
        # for each word, so that we can replace the multipart words with
        # single-part words.
        replacement_probs \
            = compute_replacement_probs_for_masked_tokens(model,
                                                          tokenized_text,
                                                          masked_indices)

    else:
        replacement_probs = probs

    return probs, replacement_probs

# Find words that could, if chosen for the masked indices, take us back to an
# arrangement that has already been tried. Because we are compiling independent
# lists of forbidden words for each index, this method can overcorrect.
def find_forbidden_words(tokenized_text, masked_indices, forbidden_texts):
    forbidden_words = [torch.ones((vocab_size,))
                       for i in range(len(masked_indices))]
    d = forbidden_texts
    def f(d, start):
        i = start
        for tok in tokenized_text[start:]:
            mask_num = None
            mask_len = 0
            for k, (i1, i2) in enumerate(masked_indices):
                if i == i1:
                    mask_num = k
                    mask_len = i2 - i1 + 1
                    break
            if mask_num is not None:
                reached_end = False
                for option_tok in d.keys():
                    if f(d[option_tok], i+mask_len):
                        option_idx = tokenizer \
                                     .convert_tokens_to_ids([option_tok])[0]
                        forbidden_words[mask_num][option_idx] = 0.0
                        reached_end = True
                return reached_end
            else:
                if tok in d:
                    d = d[tok]
                else:
                    return False
            i += 1
        return True
    if f(d, 0):
        return forbidden_words
    else:
        return None

# Function to adjust the output of the model based on various options.
def adjust_probs(model, probs, tokenized_text, start, end, masked_indices,
                 modifier=None, match_meter=None, forbidden_texts=None,
                 random_factor=False, discouraged_words=None,
                 rhyme_with=None, rhymable_only=False, rhymable_with_meters=False,
                 allow_punctuation=None, no_word_pieces=False,
                 strong_topic_bias=False, topicless_probs=None):
        
    if forbidden_texts is not None:
        forbidden_words = find_forbidden_words(tokenized_text,
                                               masked_indices,
                                               forbidden_texts)
    else:
        forbidden_words = None

    adj_probs = [[u.clone() for u in t] for t in probs]
    for k in range(len(adj_probs)):
        for j in range(len(adj_probs[k])):
            if random_factor:
                noise = torch.randn_like(adj_probs[k][j])
                noise = noise * random_factor + 1.0
                adj_probs[k][j] *= noise

            adj_probs[k][j] = m(adj_probs[k][j])

            # Do not produce word pieces. There is no way to keep the model
            # behaving reliably if we allow it to produce words that are not
            # actually in its vocabulary.
            if no_word_pieces:
                adj_probs[k][j] *= (1.0 - word_pieces.to(device))
                
            if rhymable_only:
                adj_probs[k][j] *= rhymable_words
            if rhymable_with_meters:
                for rhyme_meter in rhymable_with_meters:
                    test_meter = get_meter(get_pron(rhyme_meter))
                    meter_tensor = meter_dict[test_meter].to('cpu')
                    meter_matrix = sparse.dia_matrix((meter_tensor, [0]),
                                                     shape=(vocab_size, vocab_size))
                    # Take the dot product of meter and rhyme
                    mat = meter_matrix.dot(rhyme_matrix)
                    vec = torch.from_numpy(mat.sum(0)).squeeze().to(dtype=bool)
                    adj_probs[k][j] *= vec.to(device)

            if forbidden_words is not None:
                adj_probs[k][j] *= forbidden_words[k].to(device)

            if allow_punctuation is False:
                adj_probs[k][j] *= (1.0 - meter_dict['p'].to(device))

            if match_meter is not None:
                test_meter = get_meter(get_pron(match_meter[k]))
                meter_tensor = meter_dict[test_meter].to(device)
                if allow_punctuation is True:
                    adj_probs[k][j] *= (meter_tensor + meter_dict['p'].to(device))
                else:
                    adj_probs[k][j] *= meter_tensor

            if modifier is not None:
                adj_probs[k][j] *= modifier
            if discouraged_words is not None:
                adj_probs[k][j] *= discouraged_words

            if rhyme_with is not None:
                for rhyme_word in rhyme_with:
                    rhyme_idx = tokenizer.convert_tokens_to_ids([rhyme_word])[0]
                    rhyme_tensor = rhyme_matrix[rhyme_idx, :].todense()
                    rhyme_tensor = torch.from_numpy(rhyme_tensor)
                    rhyme_tensor = rhyme_tensor.squeeze()
                    adj_probs[k][j] *= rhyme_tensor.to(device)

            if strong_topic_bias:
                bias_factor = (m(probs[k][j]) / m(topicless_probs[k][j])) ** strong_topic_bias
                adj_probs[k][j] *= bias_factor.to(device)

    return adj_probs

# Compute a score indicating how well the model's predictions improve the
# probability for certain words. If multiple words are chosen, it is
# assumed that they are supposed to rhyme.
def compute_score_for_tokens(probs1, probs2, tokenized_text,
                             indices, relative):
    n = len(indices)
    dim = [vocab_size] * n
    
    mask_token_id = tokenizer.convert_tokens_to_ids(['[MASK]'])[0]
    
    existing_token_ids = [None] * n
    for k, (i1, i2) in enumerate(indices):
        existing_token_ids[k] = []
        for i in range(i1, i2+1):
            token = tokenized_text[i]
            index = tokenizer.convert_tokens_to_ids([token])[0]
            existing_token_ids[k].append(index)
            
    existing_words_prob = 1.0
    if probs1:
        for k in range(n):
            existing_word_prob = 0.0
            for i, tok_id in enumerate(existing_token_ids[k]):
                prob_tensor = probs1[k][i]
                existing_word_prob += prob_tensor[tok_id]
            existing_word_prob /= len(existing_token_ids[k])
            existing_words_prob *= existing_word_prob
        
    if n == 1:
        prob_tensor = probs2[0][0]
        prediction_prob = torch.max(prob_tensor)
        idx = prob_tensor.argmax().item()
        predicted_token_ids = [idx]

    elif n == 2:
        # We compute scores for possible rhyme pairs using sparse matrix
        # arithmetic. We use scipy instead of torch because torch's sparse
        # tensors do not support the .max() function.
        left_mat = sparse.dia_matrix((probs2[0][0].to('cpu'), [0]), shape=dim)
        mat = left_mat.dot(rhyme_matrix)
        right_mat = sparse.dia_matrix((probs2[1][0].to('cpu'), [0]), shape=dim)
        mat = mat.dot(right_mat)
        prediction_prob = mat.max()
        idx = mat.argmax()
        predicted_token_ids = list(numpy.unravel_index(idx, dim))
    
    if probs1:
        if relative:
            score = existing_words_prob / prediction_prob
        else:
            score = existing_words_prob
    else:
        score = prediction_prob
        
    predicted_tokens = [None] * n
    for i in range(n):
        predicted_tokens[i] \
            = tokenizer.convert_ids_to_tokens([predicted_token_ids[i]])[0]

    return predicted_tokens, score

# Tokenize a text and figure out (as best we can) its rhyme scheme.
def process_text(model, text, start, end, match_rhyme, strip_punctuation=False):
    lines = text.split('\n')
    
    tok_index = start
    toks = []
    rhyme_types = {}
    multipart_words = {}
    fixed = False
    fixed_toks = set()
    line_ends = set()
    for line in lines:
        if model.startswith('roberta') or model.startswith('gpt2'):
            line = ' ' + line

        # Check for the special '{}' characters that indicate fixed text.
        line_new = ''
        shift = 0
        fixed_chars = set()
        for i, ch in enumerate(line):
            if (model.startswith('bert') or model.startswith('distilbert')) and ch == ' ':
                # BERT tokenizer strips spaces, so we must account for that.
                shift += 1
            if ch == '{':
                fixed = True
                shift += 1
            elif ch == '}':
                fixed = False
                shift += 1
            else:
                line_new += ch
                if fixed:
                    fixed_chars.add(i - shift)
        
        line_toks = tokenizer.tokenize(line_new)
        line_fixed_toks = set()
        i = 0
        for j, tok in enumerate(line_toks):
            tok = tokenizer.convert_tokens_to_string([tok])
            if tok.startswith('##'):
                tok = tok[2:]
            nchars = len(tok)
            for k in range(nchars):
                if i+k in fixed_chars:
                    line_fixed_toks.add(j + tok_index)
                    break
            i += nchars
        
        if strip_punctuation:
            stripped_line_toks = []
            stripped_fixed_toks = set()
            shift = 0
            for j, tok in enumerate(line_toks):
                if is_punctuation(tok):
                    shift += 1
                else:
                    stripped_line_toks.append(tok)
                    if j + tok_index in line_fixed_toks:
                        stripped_fixed_toks.add(j + tok_index - shift)
            line_toks = stripped_line_toks
            line_fixed_toks = stripped_fixed_toks
        
        toks += line_toks
        fixed_toks.update(line_fixed_toks)

        # Check for multipart words.
        word_bounds = []
        after_apostrophe = False
        for i, tok in enumerate(line_toks):
            if is_word_piece(model, tok) or \
                    (after_apostrophe and tok in ("'", "s", "d", "st", "ve", "re", "nt", "ll", "t", "m")):
                if not word_bounds:
                    word_bounds.append([i, i])
                else:
                    word_bounds[-1][1] = i
            else:
                word_bounds.append([i, i])
            after_apostrophe = tok == '\''
        for i1, i2 in word_bounds:
            if i1 == i2:
                continue
            for i in range(i1, i2+1):
                multipart_words[i + tok_index] = (i1 + tok_index,
                                                  i2 + tok_index)

        if match_rhyme:
            rhyme_type = None
            # Only check rhyme for the last non-punctuation word of a line.
            word = ''
            i = len(line_toks) - 1
            while i >= 0:
                if i + tok_index in multipart_words:
                    i1, i2 = multipart_words[i + tok_index]
                    word = join_word_pieces(line_toks[i1-tok_index:i2-tok_index+1])
                    i = multipart_words[i + tok_index][0] - tok_index
                else:
                    word = line_toks[i]

                pron = get_pron(word)
                if pron != []:
                    rhyme_type = get_rhyme(pron)
                    if rhyme_type is not None:
                        if not rhyme_type in rhyme_types:
                            rhyme_types[rhyme_type] = []
                        rhyme_types[rhyme_type].append(tok_index + i)
                        break
                
                i -= 1
            
        tok_index += len(line_toks)
        line_ends.add(tok_index)

    if match_rhyme:
        rhyme_groups = {}
        for rhyme in rhyme_types:
            tok_list = rhyme_types[rhyme]
            # Rhyme groups of more than two not currently supported, so we
            # split the groups up into pairs
            for i in range(0, len(tok_list), 2):
                group = tok_list[i:i+2]
                for index in group:
                    rhyme_groups[index] = group

        return toks, fixed_toks, multipart_words, rhyme_groups, line_ends
    
    else:
        return toks, fixed_toks, multipart_words, {}, line_ends

# Alters a text iteratively, word by word, using the model to pick
# replacements.
def depoeticize(text, max_iterations=100,
                match_meter=False, match_rhyme=False, title=None, author=None,
                randomize=False, cooldown=0.01, modifier=None,
                forbid_reversions=True, preserve_punctuation=False,
                strong_topic_bias=False, stop_score=1.0,
                discourage_repetition=False, stopwords=stopwords.words('english'),
                model_type='bert-base-uncased', model_path=None,
                preserve_spacing_and_capitalization=True,
                allow_punctuation=None, sequential=False, verbose=True,
                output_metric=None, outfile=None):
    stopwords = set(stopwords)

    if modifier is not None:
        modifier = modifier()

    initialize_model(model_type, model_path)
    initialize_rhyme_and_meter(model_type, meter=match_meter or allow_punctuation is not None,
                               rhymes=match_rhyme)

    if title:
        toks1 = tokenizer.tokenize("{0}The following poem is titled {1}:\n****\n"
                                   .format(bos_token, title))
    else:
        toks1 = tokenizer.tokenize("{0}".format(bos_token))
    if author:
        toks3 = tokenizer.tokenize("\n****\nThe preceding poem is by {0}.\n{1}"
                                   .format(author, eos_token))
    else:
        toks3 = tokenizer.tokenize("{0}".format(eos_token))
    start = len(toks1)
    end = len(toks3)

    toks2, fixed_toks, multipart_words, rhyme_groups, line_ends \
        = process_text(model_type, text, start, end, match_rhyme)
    tokenized_text = toks1 + toks2 + toks3
    n = len(tokenized_text)
    
    if preserve_spacing_and_capitalization:
        spacing, capitalization = scan_tokenization(model_type, text, toks2)

    forbidden_texts = {}

    if output_metric is not None:
        outfile = open(outfile, 'w')
        outfile.write(f'''
<!DOCTYPE html>
<html>
<head></head>
<body>
''')

    if sequential:
        max_iterations = len(toks2)
    for k in range(max_iterations):
        last_score = 0.0
        
        if sequential and k >= len(tokenized_text) - start - end:
            break
            
        # Discourage the selection of words already in the text, save for stopwords.
        if discourage_repetition is not False:
            discouraged_words = torch.ones((vocab_size,))
            for i in range(start, n-end):
                tok = tokenized_text[i]
                if tok in stopwords:
                    continue
                idx = tokenizer.convert_tokens_to_ids([tok])[0]
                discouraged_words[idx] = discourage_repetition
        else:
            discouraged_words = None
        
        # Compute the scores used to choose which word to change
        outputs = [(None, None, float("inf"), None)] * n
        if sequential:
            test_range = [start + k]
        else:
            test_range = range(start, n-end)
        for i in test_range:
            if preserve_punctuation:
                if is_punctuation(tokenized_text[i]):
                    continue
            if i in fixed_toks:
                continue
            if i in multipart_words and i != multipart_words[i][0]:
                # Only try the first part of a multipart word
                continue
                
            if match_rhyme and i in rhyme_groups:
                if i != rhyme_groups[i][0]:
                    # Only try each rhyme group once
                    continue
                indices = rhyme_groups[i]
            else:
                indices = [i]
                
            indices = [multipart_words.get(idx, [idx, idx])
                       for idx in indices]
            if match_meter:
                meter = [join_word_pieces(tokenized_text[i1:i2+1])
                         for (i1, i2) in indices]
            else:
                meter = None
                
            if sequential:
                probs1 = None
                probs2 = compute_replacement_probs_for_masked_tokens(model,
                                                                     tokenized_text,
                                                                     indices)
            else:
                probs1, probs2 \
                    = compute_probs_for_masked_tokens(model,
                                                      tokenized_text,
                                                      indices)

            # The strong topic bias feature compares the probs with and
            # without the topic and biases the results in favor of words
            # that are more probable with it.
            if strong_topic_bias:
                topicless_indices = [(i1-start, i2-start)
                                     for (i1, i2) in indices]
                if sequential:
                    topicless_probs1 = None
                    topicless_probs2 \
                        = compute_replacement_probs_for_masked_tokens(model,
                                                          tokenized_text[start:-end],
                                                          topicless_indices)
                else:
                    topicless_probs1, topicless_probs2 \
                        = compute_probs_for_masked_tokens(model,
                                                          tokenized_text[start:-end],
                                                          topicless_indices)
            else:
                topicless_probs1 = None
                topicless_probs2 = None

            raw_probs = m(probs2[0][0])
            if not sequential:
                probs1 = adjust_probs(model, probs1, tokenized_text, start,
                                      end, indices, modifier,
                                      random_factor=randomize,
                                      allow_punctuation=allow_punctuation,
                             strong_topic_bias=strong_topic_bias,
                             topicless_probs=strong_topic_bias and topicless_probs1)
            probs2 = adjust_probs(model, probs2, tokenized_text, start,
                                  end, indices, modifier,
                                  meter, forbidden_texts,
                                  discouraged_words=discouraged_words,
                                  random_factor=randomize,
                                  allow_punctuation=allow_punctuation,
                                  no_word_pieces=True,
                         strong_topic_bias=strong_topic_bias,
                         topicless_probs=strong_topic_bias and topicless_probs2)
            
            predicted_tokens, score \
                = compute_score_for_tokens(probs1, probs2,
                                           tokenized_text, indices,
                                           relative=True)
            outputs[i] = (indices, predicted_tokens, score, raw_probs)
            
        # Output a visualization of the selected metric.
        if output_metric is not None:
            vals = []
            min_val = float("inf")
            max_val = 0
            for indices, predicted_tokens, score, probs in outputs:
                if output_metric == 'entropy':
                    if probs is None:
                        val = 0
                    else:
                        val = scipy.stats.entropy(probs)
                elif output_metric == 'score':
                    if score == float("inf"):
                        val = 0
                    else:
                        val = -math.log(score)
                if val < min_val:
                    min_val = val
                if val > max_val:
                    max_val = val
                vals.append(val)

            html = ""
            viz_toks = []
            for i, val in list(enumerate(vals))[start:-end]:
                if i in multipart_words:
                    i1, i2 = multipart_words[i]
                    if i > i1:
                        continue
                else:
                    i1 = i
                    i2 = i
                s = tokenizer.convert_tokens_to_string(tokenized_text[i1:i2+1])
                val_relative = (val - min_val) / (max_val - min_val)
                if output_metric == 'entropy':
                    color = hex(int((1.0 - val_relative) * 255))[2:].zfill(2)
                    color = f"#FF{color}{color}"
                elif output_metric == 'score':
                    color = hex(int((1.0 - val_relative*0.6) * 255))[2:].zfill(2)
                    color = f"#{color}{color}FF"
                viz_toks.append(f"<span style='background-color: {color}'>{s}</span>")
            if preserve_spacing_and_capitalization:
                html = detokenize(model_type, viz_toks, spacing, capitalization, html=True)
            else:
                html = tokenizer.clean_up_tokenization(tokenizer.convert_tokens_to_string(viz_toks))
            html = html.replace('\n', '<br />')
            html += "<hr />\n"

            outfile.write(html)
            outfile.flush()

        # Choose a word to change
        outputs.sort(key=lambda t: t[2])
        chosen_indices = None
        for (indices, predicted_tokens, score, _) in outputs:
            if score >= stop_score:
                break
            if predicted_tokens is None:
                continue
            chosen_indices = indices
            chosen_tokens = predicted_tokens
            last_score = score
            break

        if chosen_indices is None:
            if sequential:
                continue
            else:
                break

        # To prevent loops, we forbid the model from reverting to texts that it
        # has already tried. The texts are stored in a trie (prefix tree) for
        # efficient searchability.
        if forbid_reversions:
            d = forbidden_texts
            for tok in tokenized_text:
                if not tok in d:
                    d[tok] = {}
                d = d[tok]

        # Make the actual revision and make note of what we've done.
        change_made = False
        new_token_indices = []
        shift = 0
        for j, (i1, i2) in enumerate(chosen_indices):
            i1 -= shift
            i2 -= shift
            shift += (i2 - i1)
            n -= (i2 - i1)
            token = chosen_tokens[j]
            if i2 > i1:
                change_made = True
                tokenized_text[i1:i2+1] = [token]
                new_token_indices.append(i1)
            elif tokenized_text[i1] != token:
                change_made = True
                tokenized_text[i1] = token
                new_token_indices.append(i1)

            fixed_toks_new = set()
            for fixed_tok in fixed_toks:
                if fixed_tok > i1 and fixed_tok <= i2:
                    pass
                elif fixed_tok > i2:
                    fixed_toks_new.add(fixed_tok - (i2 - i1))
                else:
                    fixed_toks_new.add(fixed_tok)
            fixed_toks = fixed_toks_new
                
            for i in range(i1, i2+1):
                if i in multipart_words:
                    del multipart_words[i]
            replacements = {}
            for i in list(multipart_words.keys()):
                if i > i2:
                    j1, j2 = multipart_words[i]
                    del multipart_words[i]
                    replacements[i - (i2 - i1)] = (j1 - (i2 - i1),
                                                   j2 - (i2 - i1))
            for i in replacements:
                multipart_words[i] = replacements[i]
                    
            replacements = {}
            for i_old in list(rhyme_groups.keys()):
               group = rhyme_groups[i_old].copy()
               if i_old > i1:
                   i_new = i_old - (i2 - i1)
               else:
                   i_new = i_old
               group = [(idx - (i2 - i1) if idx > i1 else idx)
                        for idx in group]
               replacements[i_new] = group
            rhyme_groups = replacements

        if not change_made:
            if sequential:
                continue
            else:
                break

        if verbose:
            sample = tokenized_text[start:-end].copy()
            for i in new_token_indices:
                sample[i-start] = '<' + sample[i-start] + '>'
            if preserve_spacing_and_capitalization:
                text = detokenize(model_type, sample, spacing, capitalization)
            else:
                text = tokenizer.clean_up_tokenization(tokenizer.convert_tokens_to_string(sample))
            print('-----------------------')
            print('Iteration {0}, score = {1}'.format(k+1, last_score))
            print(text)

        if randomize and cooldown:
            randomize *= (1.0 - cooldown)

    if output_metric is not None:
        outfile.write("</body></html>\n")
        outfile.flush()
            
    if preserve_spacing_and_capitalization:
        text = detokenize(model_type, tokenized_text[start:-end], spacing, capitalization)
    else:
        text = tokenizer.clean_up_tokenization(tokenizer.convert_tokens_to_string(tokenized_text[start:-end]))
    return text

# Generates a wholly new text by running a decoder model forward with the specified
# constraints. This doesn't work very well.
def parody(text, match_meter=False, match_rhyme=False, topic=None,
           randomize=False, modifier=None, verbose=True,
           topic_prefix="", model='gpt2'):
    model_type = model

    if modifier is not None:
        modifier = modifier()
    
    global tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_type)
    model = GPT2LMHeadModel.from_pretrained(model_type)
    model.to(device)
    model.eval()
    eos_token = tokenizer.eos_token
    
    initialize_rhyme_and_meter(model_type, meter=True, rhymes=match_rhyme)
    eol_token = tokenizer.convert_tokens_to_ids(['Ġ'])[0]

    if topic:
        toks1 = tokenizer.tokenize("{0} {1} {2}. "
                                   .format(eos_token, topic_prefix, topic))
    else:
        toks1 = [eos_token]
    start = len(toks1)

    # We strip punctuation because, not being able to look ahead, the GPT-2
    # model cannot reliably produce text that matches the punctuation of the
    # original; the only way to get coherent output is to let the model decide
    # on the punctuation.
    toks2, fixed_toks, multipart_words, rhyme_groups, line_ends \
        = process_text(model_type, text, start, 0, match_rhyme,
                       strip_punctuation=True)
    
    tokenized_text = toks1 + toks2
    n = len(tokenized_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    
    # As in Beckett's "The Unnamable," we force the model to keep writing
    # even when it wants to stop.
    discouraged_words = torch.ones((vocab_size,)).to(device)
    eos_token_id = tokenizer.convert_tokens_to_ids([eos_token])[0]
    discouraged_words[eos_token_id] = 0.0
    newline_token_id = tokenizer.convert_tokens_to_ids(['\n'])[0]
    discouraged_words[newline_token_id] = 0.0

    out_toks = indexed_tokens[:start]
    i = start
    just_added_punctuation = False
    just_rhymed = False
    while i < n:
        if i in fixed_toks:
            tok = indexed_tokens[i]
            out_toks.append(tok)
            tok = tokenizer.convert_ids_to_tokens([tok])[0]
            just_added_punctuation = is_punctuation(tok)
            i += 1
            continue

        if indexed_tokens[i] == eol_token:
            out_toks.append(eol_token)
            i += 1
            continue
            
        if match_rhyme and i in rhyme_groups:
            rhyming = True
            # We can't look ahead with this model, so the rhyming constraint
            # only looks at words already chosen.
            rhyme_words = [tokenizer.convert_ids_to_tokens([indexed_tokens[idx]])[0]
                           for idx in rhyme_groups[i] if idx < i]
            # ...but we do need to make sure to choose a word that can be rhymed
            # with at least one word of the meter of later rhyming words.
            rhyme_meters = [tokenizer.convert_ids_to_tokens([indexed_tokens[idx]])[0]
                            for idx in rhyme_groups[i] if idx > i]
        else:
            rhyming = False
            rhyme_words = None
            rhyme_meters = None
        
        i1, i2 = multipart_words.get(i, [i, i])
        if match_meter:
            meter = [join_word_pieces(tokenized_text[i1:i2+1])]
        else:
            meter = None
            
        with torch.no_grad():
            tokens_tensor = torch.tensor([out_toks]).to(device)
            outputs = model(tokens_tensor)
            predictions = outputs[0]
            
        no_word_pieces = (i == start) or rhyme_words or just_added_punctuation or just_rhymed

        probs = [[predictions[0, -1, :]]]
        probs = adjust_probs(model, probs, None, 0, 0, None,
                             modifier, meter,
                             random_factor=randomize,
                             discouraged_words=discouraged_words,
                             allow_punctuation=not just_added_punctuation and not rhyme_words,
                             no_word_pieces=no_word_pieces,
                             rhyme_with=rhyme_words,
                             rhymable_only=not match_meter and rhyming,
                             rhymable_with_meters=match_meter and rhyme_meters)
        
        idx = probs[0][0].argmax().item()
        if idx == eos_token_id:
            break
        
        tok = tokenizer.convert_ids_to_tokens([idx])[0]
        out_toks.append(idx)
        
        just_rhymed = not not rhyme_words

        # Only proceed to the next input token if the output is a
        # non-punctuation token.
        if meter_dict['p'][idx] == 0.0:
            if verbose and i in line_ends:
                print('')
            # Record the chosen token for rhyming purposes.
            indexed_tokens[i] = idx
            i += i2 - i1 + 1
            just_added_punctuation = False
        else:
            # We don't allow multiple punctuation tokens in a row. This is
            # because the model can potentially get stuck in a loop where it
            # generates nothing but punctuation, in which case the process
            # would never end.
            just_added_punctuation = True
            
        if verbose:
            string = tokenizer.convert_tokens_to_string([tok])
            print(string, end='')

    if verbose:
        print('')
    out = tokenizer.convert_ids_to_tokens(out_toks[start:])
    text = tokenizer.convert_tokens_to_string(out)
    return tokenizer.clean_up_tokenization(text)

# Add modifier=metalness_modifier to bias the results toward words that occur
# frequently in heavy metal lyrics. First you will need to download the data set
# available at https://github.com/ijmbarr/pythonic-metal.
def metalness_modifier():
    f = open('metalness.json', 'r')
    metalness = json.load(f)
    f.close()
    vocab = tokenizer.get_vocab()
    metalness_modifier = [0.0] * len(vocab)
    for i, tok in enumerate(vocab):
        if tok in metalness:
            metalness_modifier[i] = metalness[tok]
    return m(torch.tensor(metalness_modifier))

# Depoeticizes a text piece by piece examining only an n-word window at a time, with
# a certain amount of context to the left and right. This procedure can handle longer
# texts than depoeticize().
def banalify(text, window_size=10, context_size=10,
             max_iterations=100, match_meter=False, match_rhyme=False,
             title=None, author=None, randomize=False, cooldown=0.01, modifier=None,
             forbid_reversions=True, preserve_punctuation=False,
             strong_topic_bias=False, stop_score=1.0,
             discourage_repetition=False, stopwords=stopwords.words('english'),
             model_type='bert-base-uncased', model_path=None,
             allow_punctuation=None, sequential=False, verbose=True):
    initialize_model(model_type, model_path)
    initialize_rhyme_and_meter(model_type, meter=match_meter or allow_punctuation is not None,
                               rhymes=match_rhyme)
    
    toks = tokenizer.tokenize(text)
    spacing, capitalization = scan_tokenization(model_type, text, toks)
    
    out_toks = []
    left_context_toks = []
    left_context_text = ''
    left_context_size = 0
    
    i = 0
    spacing_idx = 0
    bracket_left_open = False
    while i < len(toks):
        # Count the current_window excluding { and }
        window_end = i
        num_non_word_pieces = 0
        bracket_indices = set()
        j = 0
        while j < window_size and window_end < len(toks):
            if not is_word_piece(model_type, toks[window_end]):
                if toks[window_end] == '{' or toks[window_end] == '}':
                    bracket_indices.add(num_non_word_pieces)
                else:
                    j += 1
                num_non_word_pieces += 1
            window_end += 1
            
        # Extend the window if it ends in the middle of a word
        while window_end < len(toks):
            if is_word_piece(model_type, toks[window_end]):
                window_end += 1
            else:
                break
                
        window_toks = toks[i:window_end]
        window_text = tokenizer.convert_tokens_to_string(window_toks)
        window_spacing = spacing[spacing_idx:spacing_idx+num_non_word_pieces] + ['']
        window_capitalization = capitalization[spacing_idx:spacing_idx+num_non_word_pieces]
        
        # Cut out the spacing and capitalization for instances of { and }
        new_spacing = []
        tmp = ''
        for k, s in enumerate(window_spacing):
            if k in bracket_indices:
                tmp = s
            else:
                new_spacing.append(tmp + s)
                tmp = ''
        window_spacing = new_spacing
        
        new_capitalization = []
        tmp = ''
        for k, s in enumerate(window_capitalization):
            if k in bracket_indices:
                tmp = s
            else:
                if tmp.startswith('upper') and s == 'lower':
                    new_capitalization.append('upper_initial')
                else:
                    new_capitalization.append(s)
                tmp = ''
        window_capitalization = new_capitalization
        
        # Complete { and } if the window ended up inbetween them
        firstopen = window_text.find('{')
        firstclose = window_text.find('}')
        lastopen = window_text.rfind('{')
        lastclose = window_text.rfind('}')
        if bracket_left_open:
            window_text = '{' + window_text
        if firstclose != -1 and (firstopen == -1 or firstopen > firstclose):
            bracket_left_open = False
        if (lastopen != -1 or bracket_left_open) and (lastclose == -1 or lastclose < lastopen):
            window_text = window_text + '}'
            bracket_left_open = True
        
        # Count the right context, as above
        right_context_end = window_end
        j = 0
        while j < context_size and right_context_end < len(toks):
            if not is_word_piece(model_type, toks[right_context_end]):
                if toks[right_context_end] != '{' and toks[right_context_end] != '}':
                    j += 1
            right_context_end += 1
        while right_context_end < len(toks):
            if is_word_piece(model_type, toks[right_context_end]):
                right_context_end += 1
            else:
                break
                
        right_context_toks = toks[window_end:right_context_end]
        right_context_toks = [tok for tok in right_context_toks if tok not in ('{', '}')]
        right_context_text = tokenizer.convert_tokens_to_string(right_context_toks)
        if left_context_text:
            contextualized_text = f'{{{left_context_text}}} {window_text} {{{right_context_text}}}'
        else:
            contextualized_text = f'{window_text} {{{right_context_text}}}'
        
        #print(contextualized_text)
        contextualized_text = depoeticize(contextualized_text, max_iterations,
                                          match_meter, match_rhyme, title, author,
                                          randomize, cooldown, modifier,
                                          forbid_reversions, preserve_punctuation,
                                          strong_topic_bias, stop_score,
                                          discourage_repetition, stopwords,
                                          model_type, model_path, False,
                                          allow_punctuation, sequential, verbose)
        
        # Trim off the previous and next windows from the output
        window_toks = tokenizer.tokenize(contextualized_text)
        window_toks = window_toks[len(left_context_toks) if left_context_toks else 0:
                                  -len(right_context_toks) if right_context_toks else None]
        out_toks += window_toks
        
        window_text = detokenize(model_type, window_toks, window_spacing, window_capitalization)
        print(window_text, end='')
        
        # Advance the window and the end of the left context
        left_context_toks += window_toks
        left_context_toks = [tok for tok in left_context_toks if tok not in ('{', '}')]
        left_context_size += window_size
        spacing_idx += num_non_word_pieces
        i = window_end
        
        # Advance the beginning of the left context
        while left_context_size > context_size:
            left_context_toks = left_context_toks[1:]
            if not is_word_piece(model_type, left_context_toks[0]):
                left_context_size -= 1
        while left_context_toks and is_word_piece(model_type, left_context_toks[0]):
            left_context_toks = left_context_toks[1:]
        left_context_text = tokenizer.convert_tokens_to_string(left_context_toks)
        left_context_text = left_context_text.replace('{', '').replace('}', '')
    return tokenizer.convert_tokens_to_string(out_toks)

# Bouts-rimés (rhymed ends) is an old French pastime in which one person selects
# series of rhyming words and another person composes a poem using them. This
# function gets the depoeticizer to play this game by generating metered verse.
def bouts_rimés(rhymes, meter='u-u-u-u-u-',
                max_iterations=100, title=None, author=None,
                randomize=0.5, cooldown=0.3, modifier=None,
                forbid_reversions=True,
                strong_topic_bias=False, stop_score=1.0,
                discourage_repetition=False, stopwords=stopwords.words('english'),
                model_type='bert-base-uncased', model_path=None,
                sequential=False, verbose=True):
    
    initialize_model(model_type, model_path)
    initialize_rhyme_and_meter(model_type, meter=True, rhymes=True)
    
    # Start by generating words in order that match the meter.
    comma = True
    first = True
    toks = []
    lines = []
    for rhyme_word in rhymes:
        rhyme_word_meter = get_meter(get_pron(rhyme_word))
        rhyme_word_nsyls = len(rhyme_word_meter)
        required_meter = meter[:-rhyme_word_nsyls]
        required_nsyls = len(required_meter)

        line_toks = []
        while required_nsyls > 0:
            permitted_words = torch.zeros([vocab_size])
            for i in range(1, len(required_meter)+1):
                test_meter = required_meter[:i]
                if test_meter in meter_dict:
                    permitted_words += meter_dict[test_meter]
            permitted_words *= 1.0 - word_pieces

            if first:
                probs = permitted_words / sum(permitted_words)
                tok_id = torch.multinomial(probs, 1)
            else:
                probs = compute_replacement_probs_for_masked_tokens \
                        (model, toks + ['[MASK]'], [[len(toks), len(toks)]])
                probs = probs[0][0]
                noise = torch.randn_like(probs)
                noise = noise * 0.75 + 1.0
                probs *= noise
                probs = m(probs)
                probs *= permitted_words
                tok_id = probs.argmax()
            
            tok = tokenizer.convert_ids_to_tokens([tok_id])[0]
            line_toks.append(tok)
            toks.append(tok)
            nsyls = len(get_meter(get_pron(tok)).replace('p', ''))
            required_meter = required_meter[nsyls:]
            required_nsyls -= nsyls
            first = False

        if comma:
            punct = ','
        else:
            punct = '.'
        comma = not comma
        toks += [rhyme_word, punct]
        lines.append(' '.join(line_toks + ['{' + rhyme_word + '}']) + punct)

    text = '\n'.join(lines)
    print(text)

    return depoeticize(text, max_iterations, match_meter=True,
                       match_rhyme=False, title=title, author=author,
                       randomize=randomize, cooldown=cooldown,
                       modifier=modifier, forbid_reversions=forbid_reversions,
                       preserve_punctuation=False,
                       strong_topic_bias=strong_topic_bias,
                       stop_score=stop_score,
                       discourage_repetition=discourage_repetition,
                       stopwords=stopwords,
                       model_type=model_type, model_path=model_path,
                       allow_punctuation=None,
                       sequential=sequential, verbose=verbose)

