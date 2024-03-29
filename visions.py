import json
import math
import pickle
import random
import re
import time
import unicodedata

import numpy
import scipy
from scipy import sparse
import torch
from transformers import BertTokenizer, BertForMaskedLM
from transformers import DistilBertTokenizer, DistilBertForMaskedLM
from transformers import RobertaTokenizer, RobertaForMaskedLM
from transformers import DebertaTokenizer, DebertaForMaskedLM
from transformers import DebertaV2Tokenizer, DebertaV2ForMaskedLM
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

re_word = re.compile(r"^[▁a-zA-Z' ]+$")
re_space = re.compile(r"^[ \n\t]+$")
re_vowel = re.compile(r"[aeiouy]")
re_space_and_brackets = re.compile(r"^[\s{}]+$")
def get_pron(tok):
    if tok.startswith('madeupword'):
        return []
    try:
        tok = tokenizer.convert_tokens_to_string([tok])
    except KeyError:
        pass
    if tok.startswith('##'):
        tok = tok[2:]
    if tok.startswith(' ') or tok.startswith('▁'):
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
    elif model.startswith('microsoft/deberta') and '-v2' in model:
        return re_word.match(tok) and not tok.startswith('▁')
    elif model.startswith('roberta') or model.startswith('gpt2') or (model.startswith('microsoft/deberta') and '-v2' not in model):
        try:
            tok = tokenizer.convert_tokens_to_string([tok])
        except ValueError:
            pass
        except KeyError:
            pass
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
    elif model.startswith('microsoft/deberta') and '-v2' in model:
        return re_word.match(tok) and tok.startswith('▁')
    elif model_type.startswith('roberta') or model.startswith('gpt2') or (model.startswith('microsoft/deberta') and '-v2' not in model):
        try:
            tok = tokenizer.convert_tokens_to_string([tok])
        except ValueError:
            pass
        except KeyError:
            pass
        return re_word.match(tok) and (tok.startswith(' ') or tok.startswith('▁'))

def is_punctuation(tok):
    if tok == mask_token:
        return False
    try:
        tok = tokenizer.convert_tokens_to_string([tok])
    except ValueError:
        pass
    except KeyError:
        pass
    if tok.startswith('▁'):
        tok = tok[1:]
    return not re_word.match(tok)

def is_space(tok):
    if tok == mask_token:
        return False
    try:
        tok = tokenizer.convert_tokens_to_string([tok])
    except ValueError:
        pass
    except KeyError:
        pass
    if tok.startswith('▁'):
        tok = tok[1:]
    return re_space.match(tok)

# Scan a text to determine spacing and capitalization so that they can be
# preserved after detokenization.
def scan_tokenization(model, text, toks):
    spacing = []
    capitalization = []
    char_idx = 0
    tok_idx = 0
    tok_char_idx = 0
    current_spacing = ''
    current_capitalization = None
    after_apostrophe = False
    after_double_quote = False
    start_of_text = True
    while char_idx < len(text):
        char = text[char_idx]
        if char == '{' or char == '}':
            char_idx += 1
            continue
        word_piece = False
        try:
            tok = toks[tok_idx]
            if model.startswith('microsoft/deberta') and '-v2' not in model:
                tok = tokenizer.convert_tokens_to_string([tok])
            if (is_word_piece(model, tok) and tok_idx > 0 and not after_double_quote) or tok == "'" or \
                    (after_apostrophe and tok in ("s", "d", "st", "ve", "re", "nt", "ll", "t", "m")):
                tok = join_word_pieces([tok])
                word_piece = True
        except IndexError:
            tok = ''
        if tok_char_idx == 0 and (tok.startswith('Ġ') or tok.startswith('Ċ') or tok.startswith(' ') or tok.startswith('▁')) and len(tok) > 1:
            if char != ' ':
                # Advance the counter when the token contains an extraneous space
                tok_char_idx += 1
            elif current_spacing.endswith('\n') or start_of_text:
                # We have to do this because the tokenizer always adds a space to tokens at
                # the start of a line, which is stripped out in detokenize(). To account
                # for this, we have to add an extra space in cases where a space really does
                # exist at the start of a line.
                current_spacing += ' '
        try:
            tok_char = tok[tok_char_idx]
        except IndexError:
            tok_char = ''
        if tok_char in ('Ġ', 'Ċ', '▁'):
            tok_char = ' '
        # print(f'{char_idx}: \'{char}\' \'{tok}\' \'{tok_char}\'{" word_piece" if word_piece else ""}'); time.sleep(0.001)
        # RoBERTa uses '▁' for both space and newline.
        if not char.isspace() or char == tok_char or tok == '▁':
            if tok_char_idx == 1 if (tok.startswith('Ġ') or tok.startswith('Ċ') or tok.startswith(' ') or tok.startswith('▁')) else tok_char_idx == 0:
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
            start_of_text = False
            tok_char_idx += 1
            if tok_char_idx == len(tok):
                tok_idx += 1
                tok_char_idx = 0
                after_apostrophe = tok == "'"
                after_double_quote = tok in ('"', ' "', '▁"', 'Ġ"')
                if not word_piece:
                    spacing.append(current_spacing)
                    capitalization.append(current_capitalization)
                    current_spacing = ''
                    current_capitalization = None
        elif tok_char_idx == 0 or ((tok.startswith('Ġ') or tok.startswith('Ċ') or tok.startswith(' ') or tok.startswith('▁')) and tok_char_idx == 1):
            current_spacing += char
            char_idx += 1
            start_of_text = False
        else:
            print("WARNING: Text scanner found an unexpected character. This probably indicates a bug in the detokenizer.")
            char_idx += 1
            print(f'Character {char_idx}: \'{char}\' / token: \'{tok}\' / token character: \'{tok_char}\'{" word_piece" if word_piece else ""}'); time.sleep(0.1)
    spacing.append(current_spacing)
    return (spacing, capitalization)
    
def detokenize(model, toks, spacing, capitalization, html=False, start_of_line=True):
    text = ''
    i = 0
    j = 0
    current_capitalization = None
    after_apostrophe = False
    after_double_quote = False
    while i < len(toks):
        tok = toks[i]
        if model.startswith('microsoft/deberta') and '-v2' not in model:
            if tok.startswith('<span'):
                i1 = tok.index('>')+1
                i2 = tok[1:].index('<')
                try:
                    tok = tok[:i1] + tokenizer.convert_tokens_to_string([tok[i1:i2]]) + tok[i2:]
                except ValueError:
                    pass
                except KeyError:
                    pass
            elif tok.startswith('<'):
                try:
                    tok = '<' + tokenizer.convert_tokens_to_string([tok[1:-1]]) + '>'
                except ValueError:
                    pass
                except KeyError:
                    pass
            else:
                try:
                    tok = tokenizer.convert_tokens_to_string([tok])
                except ValueError:
                    pass
                except KeyError:
                    pass
        if (is_word_piece(model, tok) and i > 0 and not after_double_quote) or tok == "'" or \
                    (after_apostrophe and tok in ("s", "d", "st", "ve", "re", "nt", "ll", "t", "m")):
            tok = join_word_pieces([tok])
            if current_capitalization == 'upper_all':
                tok = tok.upper()
        else:
            current_spacing = spacing[j]
            tok = tok.replace('Ġ', ' ')
            tok = tok.replace('Ċ', '\n')
            tok = tok.replace('▁', ' ')
            if (i == 0 and start_of_line) or '\n' in current_spacing:
                # Remove the extra space created by the tokenizer if we are at the start of a line.
                if '< ' in tok:
                    tok = tok.replace('< ', '<')
                if '> ' in tok:
                    tok = tok.replace('> ', '>')
                if tok.startswith(' ') and len(tok) > 1:
                    tok = tok[1:]
            if html:
                current_spacing = current_spacing.replace(' ', '&nbsp;')
            text += current_spacing
            current_capitalization = capitalization[j]
            if current_capitalization in ('upper_initial', 'upper_ambiguous'):
                if tok.startswith('<span'):
                    # Special case for HTML visualization
                    i1 = tok.index('>')+1
                    if tok[i1] == ' ' and i1 < len(tok)-1:
                        i1 += 1
                    tok = tok[:i1] + tok[i1].upper() + tok[i1+1:]
                elif tok[0] == '<' and tok[-1] == '>':
                    # Special case for tokens marked as just modified
                    if tok[1] == ' ' and len(tok) > 2:
                        tok = tok[0:2] + tok[2].upper() + tok[3:]
                    else:
                        tok = tok[0] + tok[1].upper() + tok[2:]
                elif tok[0] == ' ' and len(tok) > 1:
                    tok = tok[0] + tok[1].upper() + tok[2:]
                else:
                    tok = tok[0].upper() + tok[1:]
            elif current_capitalization == 'upper_all':
                tok = tok.upper()
            elif current_capitalization == 'lower':
                tok = tok.lower()
            j += 1
        text += tok
        i += 1
        after_apostrophe = tok == "'"
        after_double_quote = tok in ('"', ' "', '▁"', 'Ġ"')
    text += spacing[-1]
    return text

def create_meter_dict(model_type):
    print("Generating " + model_type.replace('/', '_') + '_meter_dict.pkl')
    vocab = tokenizer.get_vocab()
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
                open(model_type.replace('/', '_') + '_meter_dict.pkl', 'wb'))
    
def create_rhyme_matrix(model_type):
    print("Generating " + model_type.replace('/', '_') + '_rhyme_matrix.pkl')
    vocab = tokenizer.get_vocab()
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
    pickle.dump((rhymable_words, rhyme_matrix), open(model_type.replace('/', '_') + '_rhyme_matrix.pkl', 'wb'))
    
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
    if meter:
        try:
            f = open(model.replace('/', '_') + '_meter_dict.pkl', 'rb')
        except FileNotFoundError:
            create_meter_dict(model)
            f = open(model.replace('/', '_') + '_meter_dict.pkl', 'rb')
        word_pieces, meter_dict = pickle.load(f)
        word_pieces = word_pieces.to(device)
        meter_dict = {k: v.to(device) for k, v in meter_dict.items()}
    else:
        try:
            f = open(model.replace('/', '_') + '_meter_dict.pkl', 'rb')
        except FileNotFoundError:
            create_meter_dict(model)
            f = open(model.replace('/', '_') + '_meter_dict.pkl', 'rb')
        word_pieces, _ = pickle.load(f)
        word_pieces = word_pieces.to(device)
    if rhymes:
        global rhyme_matrix
        try:
            f = open(model.replace('/', '_') + '_rhyme_matrix.pkl', 'rb')
        except FileNotFoundError:
            create_rhyme_matrix(model)
            f = open(model.replace('/', '_') + '_rhyme_matrix.pkl', 'rb')
        rhymable_words, rhyme_matrix = pickle.load(f)

def initialize_model(model_type, model_path):        
    global tokenizer, model, loaded_model_type, loaded_model_path, bos_token, eos_token, mask_token, pad_token_id, vocab_size
    if loaded_model_type != model_type or loaded_model_path != model_path:
        loaded_model_type = model_type
        loaded_model_path = model_path
        if model_type.startswith('distilbert'):
            tokenizer = DistilBertTokenizer.from_pretrained(model_path or model_type)
            model = DistilBertForMaskedLM.from_pretrained(model_path or model_type)
            bos_token = '[CLS]'
            eos_token = '[SEP]'
            mask_token = '[MASK]'
        elif model_type.startswith('bert'):
            tokenizer = BertTokenizer.from_pretrained(model_path or model_type)
            model = BertForMaskedLM.from_pretrained(model_path or model_type)
            bos_token = '[CLS]'
            eos_token = '[SEP]'
            mask_token = '[MASK]'
        elif model_type.startswith('roberta'):
            tokenizer = RobertaTokenizer.from_pretrained(model_path or model_type)
            model = RobertaForMaskedLM.from_pretrained(model_path or model_type)
            bos_token = tokenizer.bos_token
            eos_token = tokenizer.eos_token
            mask_token = tokenizer.mask_token
        elif model_type.startswith('microsoft/deberta') and '-v2' in model_type:
            tokenizer = DebertaV2Tokenizer.from_pretrained(model_path or model_type)
            model = DebertaV2ForMaskedLM.from_pretrained(model_path or model_type)
            bos_token = tokenizer.cls_token
            eos_token = tokenizer.sep_token
            mask_token = tokenizer.mask_token
        elif model_type.startswith('microsoft/deberta'):
            tokenizer = DebertaTokenizer.from_pretrained(model_path or model_type)
            model = DebertaForMaskedLM.from_pretrained(model_path or model_type)
            bos_token = tokenizer.cls_token
            eos_token = tokenizer.sep_token
            mask_token = tokenizer.mask_token
        vocab_size = model.config.vocab_size
        pad_token_id = model.config.pad_token_id
        model = torch.nn.DataParallel(model)
        model.to(device)
        model.eval()

# Computes the model's predictions for a text with a given set of ranges
# masked.
def compute_probs_for_masked_tokens(model, tokenized_texts, masked_index_lists, batch_size,
                                    replacements_only=False):
    indexed_tokens = []
    tensor_indices = {}
    wwm_tensor_indices = {}
    for j1, (tokenized_text, masked_indices) in enumerate(zip(tokenized_texts, masked_index_lists)):
        for j2, masked_index_set in enumerate(masked_indices):
            n = len(masked_index_set)
            multipart_words = False
            tokens = tokenized_text.copy()
            for i1, i2 in masked_index_set:
                if i2 > i1:
                    multipart_words = True
                    if replacements_only:
                        break
                tokens[i1:i2+1] = [mask_token] * (i2 - i1 + 1)
            if not replacements_only or not multipart_words:
                tensor_indices[(j1, j2)] = len(indexed_tokens)
                indexed_tokens.append(tokenizer.convert_tokens_to_ids(tokens))
            # If one of the ranges covers a multipart word, we need to compute probabilities
            # both for the text with individual tokens masked and with the whole word masked.
            if multipart_words:
                wwm_tokens = tokenized_text.copy()
                shift = 0
                for i1, i2 in masked_index_set:
                    i1 -= shift
                    i2 -= shift
                    shift += (i2 - i1)
                    wwm_tokens[i1:i2+1] = [mask_token]
                wwm_tensor_indices[(j1, j2)] = len(indexed_tokens)
                indexed_tokens.append(tokenizer.convert_tokens_to_ids(wwm_tokens))
    
    # Add padding so all index sequences are the same length.
    max_len = 0
    for indices in indexed_tokens:
        n = len(indices)
        if n > max_len:
            max_len = n
    attention_mask = []
    for i in range(len(indexed_tokens)):
        n = len(indexed_tokens[i])
        if n < max_len:
            indexed_tokens[i] = indexed_tokens[i] + [pad_token_id]*(max_len-n)
        attention_mask.append([1]*n + [0]*(max_len-n))
    tokens_tensor = torch.tensor(indexed_tokens, device='cpu')
    attention_mask = torch.tensor(attention_mask, device='cpu')

    all_predictions = []
    ntexts = tokens_tensor.shape[0]
    nbatches = math.ceil(ntexts / batch_size)
    for batchnum in range(nbatches):
        batch_start = batchnum * batch_size
        batch_end = min(batch_start + batch_size, ntexts)
        toks_slice = tokens_tensor[batch_start:batch_end].to(device)
        mask_slice = attention_mask[batch_start:batch_end].to(device)
        with torch.no_grad():
            outputs = model(toks_slice,
                            attention_mask=mask_slice)
        del toks_slice
        del mask_slice
        all_predictions.append(outputs[0].to('cpu'))
        del outputs
    del tokens_tensor
    del attention_mask
    if len(all_predictions) == 0:
        return [None]*len(tokenized_texts), [None]*len(tokenized_texts)

    all_probs = []
    all_replacement_probs = []
    for j1, (tokenized_text, masked_indices) in enumerate(zip(tokenized_texts, masked_index_lists)):
        probs = []
        replacement_probs = []
        for j2, masked_index_set in enumerate(masked_indices):
            n = len(masked_index_set)
            multipart_words = (j1, j2) in wwm_tensor_indices
            if not replacements_only or not multipart_words:
                j = tensor_indices[(j1, j2)]
                index_set_probs = [None] * n
                for k, (i1, i2) in enumerate(masked_index_set):
                    word_probs = []
                    for i in range(i1, i2+1):
                        jbatch = j // batch_size
                        jpreds = j % batch_size
                        word_probs.append(all_predictions[jbatch][jpreds, i, :])
                    index_set_probs[k] = word_probs
                if not replacements_only:
                    probs.append(index_set_probs)
            if multipart_words:
                j = wwm_tensor_indices[(j1, j2)]
                index_set_probs = [None] * n
                shift = 0
                for k, (i1, i2) in enumerate(masked_index_set):
                    i1 -= shift
                    i2 -= shift
                    shift += (i2 - i1)
                    jbatch = j // batch_size
                    jpreds = j % batch_size
                    index_set_probs[k] = [all_predictions[jbatch][jpreds, i1, :]]
            replacement_probs.append(index_set_probs)
        all_probs.append(probs)
        all_replacement_probs.append(replacement_probs)
    for predictions in all_predictions:
        del predictions

    if replacements_only:
        return [None]*len(tokenized_texts), all_replacement_probs
    else:
        return all_probs, all_replacement_probs

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

    adj_probs = [[u.clone().to(device) for u in t] for t in probs]
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
                adj_probs[k][j] *= (1.0 - word_pieces)
                
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
                adj_probs[k][j] *= (1.0 - meter_dict['p'])

            if match_meter is not None:
                test_meter = get_meter(get_pron(match_meter[k]))
                meter_tensor = meter_dict[test_meter]
                if allow_punctuation is True:
                    adj_probs[k][j] *= (meter_tensor + meter_dict['p'])
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
                adj_probs[k][j] /= m(topicless_probs[k][j].to(device)) ** strong_topic_bias
                # Sometimes funky scores can arise from this division; we just avoid
                # choosing those words.
                nan_mask = adj_probs[k][j].isnan()
                adj_probs[k][j].masked_fill_(nan_mask, 0.0)
                inf_mask = adj_probs[k][j].isinf()
                adj_probs[k][j].masked_fill_(inf_mask, 0.0)

    return adj_probs

# Compute a score indicating how well the model's predictions improve the
# probability for certain words. If multiple words are chosen, it is
# assumed that they are supposed to rhyme.
def compute_score_for_tokens(probs1, probs2, tokenized_text,
                             indices, require_replacement, relative):
    n = len(indices)
    dim = [vocab_size] * n
    
    mask_token_id = tokenizer.convert_tokens_to_ids([mask_token])[0]
    
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
            if len(existing_token_ids[k]) > 1:
                for i, tok_id in enumerate(existing_token_ids[k]):
                    prob_tensor = probs1[k][i]
                    existing_word_prob += prob_tensor[tok_id].log()
                existing_word_prob /= len(existing_token_ids[k])
                existing_words_prob *= existing_word_prob.exp()
            else:
                existing_words_prob = probs1[k][0][existing_token_ids[k]]

    if require_replacement:
        for k in range(n):
            probs2[k][0][existing_token_ids[k][0]] = 0.0
        
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

        # Hack to deal with int32 overflow when the vocab is large
        if idx < 0:
            idx += 1 << 32
        predicted_token_ids = list(numpy.unravel_index(idx, dim))
        while mat[predicted_token_ids[0], predicted_token_ids[1]] < prediction_prob:
            idx += 1 << 32
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

    return predicted_tokens, float(score)

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
    first_line = True
    for line in lines:
        if (model.startswith('roberta') or model.startswith('gpt2') or (model.startswith('microsoft/deberta') and '-v2' not in model)) and not first_line and not line.startswith(' '):
            line = ' ' + line
        first_line = False

        # Check for the special '{}' characters that indicate fixed text.
        line_new = ''
        shift = 0
        fixed_chars = set()
        for i, ch in enumerate(line):
            if (model.startswith('bert') or model.startswith('distilbert') or (model.startswith('microsoft/deberta') and '-v2' in model)) and ch == ' ': 
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
            if model.startswith('microsoft/deberta') and '-v2' not in model:
                tok = tokenizer.convert_tokens_to_string([tok])
            if tok.startswith('##'):
                tok = tok[2:]
            if tok.startswith('▁'):
                tok = tok[1:]
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
        after_double_quote = False
        for i, tok in enumerate(line_toks):
            if model.startswith('microsoft/deberta') and '-v2' not in model:
                tok = tokenizer.convert_tokens_to_string([tok])
            if (is_word_piece(model, tok) and not after_double_quote) or tok == "'" or \
                    (after_apostrophe and tok in ("s", "d", "st", "ve", "re", "nt", "ll", "t", "m")):
                if not word_bounds:
                    word_bounds.append([i, i])
                else:
                    word_bounds[-1][1] = i
            else:
                word_bounds.append([i, i])
            after_apostrophe = tok == "'"
            after_double_quote = tok in ('"', ' "', '▁"', 'Ġ"')
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
def depoeticize(text, max_iterations=100, batch_size=10,
                match_meter=False, match_rhyme=False, title=None, author=None,
                randomize=False, cooldown=0.01, modifier=None,
                forbid_reversions=True, preserve_punctuation=False,
                strong_topic_bias=False, stop_score=1.0,
                discourage_repetition=False, stopwords=stopwords.words('english'),
                model_type='bert-base-uncased', model_path=None,
                preserve_spacing_and_capitalization=True,
                allow_punctuation=None, sequential=False, verbose=True,
                outfile=None, top_n=10, require_new_rhymes=False,
                num_changes_per_iter=1):
    stopwords = set(stopwords)

    # The detokenizer doesn't properly handle cases where the input is all space. It is necessary
    # to implement good behavior in this case because it can arise when this function is called
    # by banalify.
    if re_space_and_brackets.match(text):
        return text.replace('{', '').replace('}', '')
    
    # Stripping smart quotes because some of the models don't seem to handle them properly.
    text = text.replace('“','"').replace('”','"').replace('‘','\'').replace('’','\'').replace('\r\n', '\n')

    initialize_model(model_type, model_path)
    initialize_rhyme_and_meter(model_type, meter=match_meter or allow_punctuation is not None,
                               rhymes=match_rhyme)

    if modifier is not None:
        modifier = modifier().to(device)

    topicless_toks1 = tokenizer.tokenize(f'{bos_token}Title: {mask_token} / Author: {mask_token} {mask_token} / Text: \n\n')
    if title and author:
        toks1 = tokenizer.tokenize(f'{bos_token}Title: {title} / Author: {author} / Text: \n\n')
    elif title:
        toks1 = tokenizer.tokenize(f'{bos_token}Title: {title} / Author: {mask_token} {mask_token} / Text: \n\n')
    elif author:
        toks1 = tokenizer.tokenize(f'{bos_token}Title: {mask_token} / Author: {author} / Text: \n\n')
    else:
        toks1 = [bos_token]
    toks3 = [eos_token]
    start = len(toks1)
    end = len(toks3)

    toks2, fixed_toks, multipart_words, rhyme_groups, line_ends \
        = process_text(model_type, text, start, end, match_rhyme)
    tokenized_text = toks1 + toks2 + toks3
    n = len(tokenized_text)
    
    if preserve_spacing_and_capitalization:
        spacing, capitalization = scan_tokenization(model_type, text, toks2)

    forbidden_texts = {}

    if outfile is not None:
        outfile = open(outfile, 'w')
        html = f'''
<!DOCTYPE html>
<html>
<head>
<link rel="stylesheet" href="viz.css">
<script src='jquery.js'></script>
<script src='viz.js'></script>
</head>
<body>
Model: {model_type}{' "' + model_path + '"' if model_path else ''}<br />
Max iterations: {max_iterations}<br />
'''
        if strong_topic_bias:
            html += f'Strong topic bias: {strong_topic_bias}<br />'
        if randomize:
            html += f'Randomizing, cooldown={cooldown}<br />'
        if stop_score != 1.0:
            html += f'Stop score: {stop_score}<br />'
        if match_meter:
            html += 'Matching meter<br />'
        if match_rhyme:
            html += 'Matching rhyme<br />'
        if require_new_rhymes:
            html += 'Requiring new rhymes<br />'
        if forbid_reversions:
            html += 'Forbidding reversions<br />'
        if preserve_punctuation:
            html += 'Preserving punctuation<br />'
        if discourage_repetition:
            html += 'Discouraging repetition<br />'
        if allow_punctuation is True:
            html += 'Always allowing punctuation<br />'
        if allow_punctuation is False:
            html += 'Never allowing punctuation<br />'
        if modifier is not None:
            html += 'Modifier provided<br />'
        if sequential:
            html += 'Running in sequential mode<br />'
        html += '<hr />'
        if title:
            html += f'Title: {title}<br />'
        if author:
            html += f'Author: {author}<br />'
        html += '''<hr />
Highlight: <select name="highlighting" id="highlighting">
  <option>Score</option>
  <option>Entropy</option>
  <option>None</option>
</select>
<input type="checkbox" id="changes" name="changes" value="Changes">
<label for="changes"> Indicate changes</label><br />Double-click on words to see predictions<hr />'''

        outfile.write(html)

    if sequential:
        max_iterations = len(toks2)
    new_token_indices = []
    if require_new_rhymes:
        original_rhymes = {}
        for i in rhyme_groups:
            original_rhymes[i] = tokenized_text[i]
    for k in range(max_iterations):
        last_score = 0.0
        if verbose:
            iter_start_time = time.time()
        
        if sequential and k >= len(tokenized_text) - start - end:
            break

        if require_new_rhymes:
            fallback_indices = None
            fallback_predicted_tokens = None
            fallback_score = None
            
        # Discourage the selection of words already in the text, save for stopwords.
        if discourage_repetition is not False:
            discouraged_words = torch.ones((vocab_size,))
            for i in range(start, n-end):
                tok = tokenized_text[i]
                if tok in stopwords:
                    continue
                idx = tokenizer.convert_tokens_to_ids([tok])[0]
                discouraged_words[idx] = discourage_repetition
            discouraged_words = discouraged_words.to(device)
        else:
            discouraged_words = None
        
        # Compute the scores used to choose which word to change
        outputs = []
        if sequential:
            test_range = [start + k]
        else:
            test_range = range(start, n-end)

        # First, figure out the indices to test for replacements. This is non-trivial because
        # we want to replace multipart words as whole units and rhyme groups together.
        masked_indices = []
        if strong_topic_bias:
            topicless_masked_indices = []
        for i in test_range:
            if preserve_punctuation:
                if is_punctuation(tokenized_text[i]):
                    continue
            if is_space(tokenized_text[i]):
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
            masked_indices.append(indices)
            if strong_topic_bias:
                topicless_indices = [(i1-start+len(topicless_toks1), i2-start+len(topicless_toks1))
                                      for (i1, i2) in indices]
                topicless_masked_indices.append(topicless_indices)

        # Next, run all the predictions in batches. 
        if strong_topic_bias:
            topicless_tokenized_text = topicless_toks1 + tokenized_text[start:-end] + toks3
            (all_probs1, all_topicless_probs1), (all_probs2, all_topicless_probs2) \
                = compute_probs_for_masked_tokens(model,
                                                (tokenized_text, topicless_tokenized_text),
                                                (masked_indices, topicless_masked_indices),
                                                batch_size,
                                                replacements_only=sequential)
        else:
            (all_probs1,), (all_probs2,) \
                = compute_probs_for_masked_tokens(model,
                                                (tokenized_text,),
                                                (masked_indices,),
                                                batch_size,
                                                replacements_only=sequential)
            all_topicless_probs1 = None
            all_topicless_probs2 = None

        # Finally, adjust the probabilities and compute the final scores.
        for i, indices in enumerate(masked_indices):
            probs1 = all_probs1 and all_probs1[i]
            probs2 = all_probs2 and all_probs2[i]
            topicless_probs1 = all_topicless_probs1 and all_topicless_probs1[i]
            topicless_probs2 = all_topicless_probs2 and all_topicless_probs2[i]

            if match_meter:
                meter = [join_word_pieces(tokenized_text[i1:i2+1])
                         for (i1, i2) in indices]
            else:
                meter = None

            if require_new_rhymes and len(indices) > 1:
                require_replacement = True
                for i1, i2 in indices:
                    if i1 in original_rhymes and original_rhymes[i1] != tokenized_text[i1]:
                        require_replacement = False
                        break
            else:
                require_replacement = False

            raw_probs = m(probs2[0][0]).to('cpu')
            raw_topicless_probs = topicless_probs2 and m(topicless_probs2[0][0]).to('cpu')
            if not sequential:
                probs1 = adjust_probs(model, probs1, tokenized_text, start,
                                      end, indices, modifier,
                                      random_factor=randomize,
                                      allow_punctuation=allow_punctuation,
                             strong_topic_bias=strong_topic_bias,
                             topicless_probs=strong_topic_bias and topicless_probs1)
            probs2 = adjust_probs(model, probs2, tokenized_text, start,
                                  end, indices, modifier,
                                  meter, forbidden_texts if num_changes_per_iter == 1 else {},
                                  discouraged_words=discouraged_words,
                                  random_factor=randomize,
                                  allow_punctuation=allow_punctuation,
                                  no_word_pieces=True,
                         strong_topic_bias=strong_topic_bias,
                         topicless_probs=strong_topic_bias and topicless_probs2)
            
            adjusted_probs = probs2[0][0].to('cpu')
            predicted_tokens, score \
                = compute_score_for_tokens(probs1, probs2,
                                           tokenized_text, indices,
                                    relative=True,
                                    require_replacement=require_replacement)

            if require_replacement:
                fallback_indices = indices
                fallback_predicted_tokens = predicted_tokens
                fallback_score = score
            outputs.append((indices, predicted_tokens, score, raw_topicless_probs,
                            raw_probs, adjusted_probs))

        del all_probs1
        del all_probs2
        del all_topicless_probs1
        del all_topicless_probs2
            
        # Output an HTML visualization.
        if outfile is not None:
            vals = {}
            min_entropy = float("inf")
            max_entropy = 0
            min_score = float("inf")
            max_score = 0
            for indices, predicted_tokens, score, probs1, probs2, probs3 in outputs:
                def get_entropy(probs):
                    if probs is None:
                        return 0
                    else:
                        return scipy.stats.entropy(probs.cpu())
                entropy1 = get_entropy(probs1)
                entropy2 = get_entropy(probs2)
                entropy3 = get_entropy(probs3)
                if title is not None:
                    selected_entropy = entropy1
                else:
                    selected_entropy = entropy2
                if selected_entropy < min_entropy:
                    min_entropy = selected_entropy
                if selected_entropy > max_entropy:
                    max_entropy = selected_entropy
                if score == float("inf"):
                    score_val = 0
                elif score == 0:
                    score_val = -float("inf")
                else:
                    score_val = -math.log(score)
                    if score_val < min_score:
                        min_score = score_val
                    if score_val > max_score:
                        max_score = score_val
                for i1, i2 in indices:
                    for i in range(i1, i2+1):
                        vals[i] = (entropy1, entropy2, entropy3, probs1, probs2, probs3, score_val, predicted_tokens)

            html = "<div class='iter'>"
            viz_toks = []
            for i in range(start, len(tokenized_text)-end):
                if i in vals:
                    entropy1, entropy2, entropy3, probs1, probs2, probs3, score_val, predicted_tokens = vals[i]
                else:
                    entropy1, entropy2, entropy3, probs1, probs2, probs3, score_val, predicted_tokens = 0.0, 0.0, 0.0, None, None, None, 0.0, None
                if i in multipart_words:
                    i1, i2 = multipart_words[i]
                    if i > i1:
                        continue
                else:
                    i1 = i
                    i2 = i
                s = tokenizer.convert_tokens_to_string(tokenized_text[i1:i2+1]).replace(" ' ", "'")
                if tokenized_text[i1][0] in ('Ġ', 'Ċ', '▁', ' '):
                    s = ' ' + s
                if max_entropy == min_entropy:
                    entropy_relative = 0.0
                else:
                    if title is not None:
                        selected_entropy = entropy1
                    else:
                        selected_entropy = entropy2
                    entropy_relative = (selected_entropy - min_entropy) / (max_entropy - min_entropy)
                if max_score == min_score:
                    score_relative = 0.0
                else:
                    score_relative = (score_val - min_score) / (max_score - min_score)
                changed = i in new_token_indices
                changed = " changed-tok" if changed else ""
                raw_probs = probs1
                raw_topicless_probs = probs2
                adjusted_probs = probs3
                def get_top(probs):
                    if probs is None:
                        return 'null'
                    out = torch.topk(probs, top_n)
                    top_options = zip(out.indices, out.values)
                    top_options = [(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens([j])).replace(' ', ''),
                                    float(p))
                                   for j, p in top_options]
                    top_options = json.dumps(top_options)
                    return top_options
                options1 = get_top(raw_probs)
                options2 = get_top(raw_topicless_probs)
                options3 = get_top(adjusted_probs)
                if predicted_tokens is not None:
                    replacement_tokens = json.dumps([tokenizer.convert_tokens_to_string([s]) for s in predicted_tokens])
                else:
                    replacement_tokens = 'null'
                viz_toks.append(f"<span class='tok{changed}' data-entropy1='{entropy1}' data-entropy2='{entropy2}' data-entropy3='{entropy3}' data-score='{score_val}' data-entropy-relative='{entropy_relative}' data-score-relative='{score_relative}' data-options1='{options1}' data-options2='{options2}' data-options3='{options3}' data-replacements='{replacement_tokens}'>{s}</span>")
            if preserve_spacing_and_capitalization:
                html += detokenize(model_type, viz_toks, spacing, capitalization, html=True)
            else:
                html += tokenizer.clean_up_tokenization(tokenizer.convert_tokens_to_string(viz_toks))
            html = html.replace('\n', '<br />')
            html += "</div><hr />\n"

            outfile.write(html)
            outfile.flush()

        # Choose words to change
        outputs.sort(key=lambda t: t[2])
        chosen_index_lists = []
        chosen_token_lists = []
        i = 0
        for indices, predicted_tokens, score, _, _, _ in outputs:
            if score >= stop_score or i >= num_changes_per_iter:
                break
            if predicted_tokens is None:
                continue
            i += 1
            chosen_index_lists.append(indices)
            chosen_token_lists.append(predicted_tokens)
            if i == 1:
                lowest_score = score

        if not chosen_index_lists:
            if sequential:
                continue
            elif require_new_rhymes and fallback_indices is not None:
                chosen_index_lists = [fallback_indices]
                chosen_token_lists = [fallback_predicted_tokens]
                last_score = fallback_score
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
        new_tokenized_text = tokenized_text.copy()
        for change_num in range(len(chosen_index_lists)):
            chosen_indices = chosen_index_lists[change_num]
            chosen_tokens = chosen_token_lists[change_num]
            shift = 0
            for j, (i1, i2) in enumerate(chosen_indices):
                i1 -= shift
                i2 -= shift
                shift += (i2 - i1)
                n -= (i2 - i1)
                token = chosen_tokens[j]
                if i2 > i1:
                    change_made = True
                    new_tokenized_text[i1:i2+1] = [token]
                    new_token_indices.append(i1)
                elif tokenized_text[i1] != token:
                    change_made = True
                    new_tokenized_text[i1] = token
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

                for change_num2 in range(len(chosen_index_lists)):
                    replacement = []
                    for i, (j1, j2) in enumerate(chosen_index_lists[change_num2]):
                        if j1 > i2:
                            replacement.append((j1 - (i2 - i1),
                                               j2 - (i2 - i1)))
                        else:
                            replacement.append((j1, j2))
                    chosen_index_lists[change_num2] = replacement
                
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
                        
                if require_new_rhymes:
                    replacements = {}
                    for i_old in list(original_rhymes.keys()):
                        rhyme_tok = original_rhymes[i_old]
                        if i_old > i1:
                            i_new = i_old - (i2 - i1)
                        else:
                            i_new = i_old
                        replacements[i_new] = rhyme_tok
                    original_rhymes = replacements

        if forbid_reversions and num_changes_per_iter > 1:
            # There's no clear way to implement this when choosing tokens if we
            # are going to change multiple tokens at once, so we just stop
            # when we reach a reversion.
            d = forbidden_texts
            for tok in new_tokenized_text:
                if tok in d:
                    d = d[tok]
                else:
                    break
            else:
                change_made = False

        if change_made:
            tokenized_text = new_tokenized_text
        else:
            if sequential:
                continue
            else:
                break

        if verbose:
            iter_end_time = time.time()
            sample = tokenized_text[start:-end].copy()
            for i in new_token_indices:
                sample[i-start] = '<' + sample[i-start] + '>'
            if preserve_spacing_and_capitalization:
                text = detokenize(model_type, sample, spacing, capitalization)
            else:
                text = tokenizer.clean_up_tokenization(tokenizer.convert_tokens_to_string(sample))
            print('-----------------------')
            print('Iteration {0}, lowest score = {1}, running time = {2}s'.format(k+1, lowest_score,
                                                                           iter_end_time - iter_start_time))
            print(text)

        if randomize and cooldown:
            randomize *= (1.0 - cooldown)

    if outfile is not None:
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

# Add modifier=json_modifier('<filename>') to bias the results in favor of certain
# words, as read from a JSON file. The file should contain an object mapping words to
# numbers. You can generate file like this using generate_modifier.py. Lower the
# factor parameter to decrease the effect.
def json_modifier(filename, factor=1.0, default_score=-10.0):
    def modifier():
        f = open(filename, 'r')
        scores = json.load(f)
        f.close()
        vocab = tokenizer.get_vocab()
        score_vector = [default_score] * vocab_size
        for tok in vocab:
            i = vocab[tok]
            if tok.startswith('Ġ') or tok.startswith('Ċ') or tok.startswith(' ') or tok.startswith('▁'):
                tok = tok[1:]
            tok = tok.lower()
            if tok in scores:
                score_vector[i] = scores[tok]
        score_tensor = m(torch.tensor(score_vector))
        mean_score = 1.0 / score_tensor.shape[0]
        return (1.0 - factor) * mean_score + factor * score_tensor
    return modifier

# Add modifier=metalness_modifier to bias the results toward words that occur
# frequently in heavy metal lyrics. First you will need to download the data set
# available at https://github.com/ijmbarr/pythonic-metal.
metalness_modifier = json_modifier('metalness.json')

# Depoeticizes a text piece by piece examining only an n-word window at a time, with
# a certain amount of context to the left and right. This procedure can handle longer
# texts than depoeticize().
def banalify(text, window_size=10, context_size=10, batch_size=10,
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
    
    # Stripping characters that some of the models don't seem to handle properly.
    text = text.replace('“','"').replace('”','"').replace('‘','\'').replace('’','\'').replace('\r', '').replace('—', '').replace('…', '...').replace(' ', ' ')
    text = "".join([c for c in unicodedata.normalize('NFKD', text) if not unicodedata.combining(c)])

    toks, fixed_toks, _, _, _ = process_text(model_type, text, 0, 0, False, False)
    spacing, capitalization = scan_tokenization(model_type, text, toks)

    if model_type.startswith('microsoft/deberta') and '-v2' not in model_type:
        open_bracket = tokenizer.tokenize('{')[0]
        close_bracket = tokenizer.tokenize('}')[0]
    else:
        open_bracket = '{'
        close_bracket = '}'
    
    out_text = ''
    left_context_toks = []
    left_context_text = ''
    left_context_size = 0
    
    i = 0
    spacing_idx = 0
    bracket_left_open = False
    while i < len(toks):
        # Count the current_window
        window_end = i
        num_non_word_pieces = 0
        bracket_indices = set()
        spaced_bracket_indices = set()
        j = 0
        after_apostrophe = False
        after_double_quote = False
        while j < window_size and window_end < len(toks):
            tok = toks[window_end]
            if model_type.startswith('microsoft/deberta') and '-v2' not in model_type:
                tok = tokenizer.convert_tokens_to_string([tok])
            if not ((is_word_piece(model_type, tok) and not after_double_quote) or tok == "'" or \
                    (after_apostrophe and tok in ("s", "d", "st", "ve", "re", "nt", "ll", "t", "m"))):
                j += 1
                num_non_word_pieces += 1
            window_end += 1
            after_apostrophe = tok == "'"
            after_double_quote = tok in ('"', ' "', '▁"', 'Ġ"')
            
        # Extend the window if it ends in the middle of a word
        after_apostrophe = False
        after_double_quote = False
        while window_end < len(toks):
            tok = toks[window_end]
            if model_type.startswith('microsoft/deberta') and '-v2' not in model_type:
                tok = tokenizer.convert_tokens_to_string([tok])
            if (is_word_piece(model_type, tok) and not after_double_quote) or tok == "'" or \
                    (after_apostrophe and tok in ("s", "d", "st", "ve", "re", "nt", "ll", "t", "m")):
                window_end += 1
            else:
                break
            after_apostrophe = tok == "'"
            after_double_quote = tok in ('"', ' "', '▁"', 'Ġ"')
                
        window_toks = toks[i:window_end]
        window_spacing = spacing[spacing_idx:spacing_idx+num_non_word_pieces] + ['']
        window_capitalization = capitalization[spacing_idx:spacing_idx+num_non_word_pieces]
        
        # Add { and } around fixed text within the window
        bracketed_window_toks = []
        fixed = False
        for k, tok in enumerate(window_toks):
            tok_idx = i + k
            if tok_idx in fixed_toks and not fixed:
                fixed = True
                bracketed_window_toks.append(open_bracket)
            if tok_idx not in fixed_toks and fixed:
                fixed = False
                bracketed_window_toks.append(close_bracket)
            bracketed_window_toks.append(tok)
        if fixed:
            bracketed_window_toks.append(close_bracket)
        window_text = tokenizer.convert_tokens_to_string(bracketed_window_toks)
        
        # Count the right context, as above
        right_context_end = window_end
        j = 0
        after_apostrophe = False
        after_double_quote = False
        while j < context_size and right_context_end < len(toks):
            tok = toks[right_context_end]
            if model_type.startswith('microsoft/deberta') and '-v2' not in model_type:
                tok = tokenizer.convert_tokens_to_string([tok])
            if not ((is_word_piece(model_type, tok) and not after_double_quote) or tok == "'" or \
                    (after_apostrophe and tok in ("s", "d", "st", "ve", "re", "nt", "ll", "t", "m"))):
                j += 1
            right_context_end += 1
            after_apostrophe = tok == "'"
            after_double_quote = tok in ('"', ' "', '▁"', 'Ġ"')
        after_apostrophe = False
        after_double_quote = False
        while right_context_end < len(toks):
            tok = toks[right_context_end]
            if model_type.startswith('microsoft/deberta') and '-v2' not in model_type:
                tok = tokenizer.convert_tokens_to_string([tok])
            if (is_word_piece(model_type, tok) and not after_double_quote) or tok == "'" or \
                    (after_apostrophe and tok in ("s", "d", "st", "ve", "re", "nt", "ll", "t", "m")):
                right_context_end += 1
            else:
                break
            after_apostrophe = tok == "'"
            after_double_quote = tok in ('"', ' "', '▁"', 'Ġ"')
                
        right_context_toks = toks[window_end:right_context_end]
        right_context_text = tokenizer.convert_tokens_to_string(right_context_toks)
        if model_type.startswith('bert') or (model_type.startswith('microsoft/deberta') and '-v2' in model_type):
            maybe_space = ' '
        else:
            maybe_space = ''
        if left_context_text:
            contextualized_text = f'{{{left_context_text}}}{maybe_space}{window_text}{maybe_space}{{{right_context_text}}}'
        else:
            contextualized_text = f'{window_text}{maybe_space}{{{right_context_text}}}'
        
        contextualized_text = depoeticize(contextualized_text, max_iterations, batch_size,
                                        match_meter, match_rhyme, title, author,
                                        randomize, cooldown, modifier,
                                        forbid_reversions, preserve_punctuation,
                                        strong_topic_bias, stop_score,
                                        discourage_repetition, stopwords,
                                        model_type, model_path, False,
                                        allow_punctuation, sequential, verbose)
        
        # Trim off the previous and next windows from the output
        window_toks = tokenizer.tokenize(contextualized_text + ' x')[:-1]
        if model_type.startswith('microsoft/deberta') and '-v2' in model_type and tokenizer.convert_tokens_to_string(window_toks[:2]) in ('."', ".'"):
            # Special case involving how the DeBERTa v2 tokenizer handles '."'
            window_toks = window_toks[1:]
            window_toks[0] = tokenizer.tokenize('."')[0]
        window_toks = window_toks[len(left_context_toks) if left_context_toks else 0:
                                  -len(right_context_toks) if right_context_toks else None]
        
        start_of_line = out_text == '' or out_text.endswith('\n')
        window_text = detokenize(model_type, window_toks, window_spacing, window_capitalization, start_of_line=start_of_line)
        out_text += window_text
        print(window_text, end='')
        
        # Advance the window and the end of the left context
        left_context_toks += window_toks
        #print(left_context_toks)
        #print(left_context_size)
        #print(window_size)
        left_context_size = len(left_context_toks)
        spacing_idx += num_non_word_pieces - len(bracket_indices)
        i = window_end
        
        # Advance the beginning of the left context
        after_apostrophe = False
        after_double_quote = False
        while left_context_size > context_size:
            left_context_toks = left_context_toks[1:]
            tok = left_context_toks[0]
            if model_type.startswith('microsoft/deberta') and '-v2' not in model_type:
                tok = tokenizer.convert_tokens_to_string([tok])
            if not ((is_word_piece(model_type, tok) and not after_double_quote) or tok == "'" or \
                    (after_apostrophe and tok in ("s", "d", "st", "ve", "re", "nt", "ll", "t", "m"))):
                left_context_size -= 1
            after_apostrophe = tok == "'"
            after_double_quote = tok in ('"', ' "', '▁"', 'Ġ"')
        after_apostrophe = False
        after_double_quote = False
        while left_context_toks and (
                    (is_word_piece(model_type, left_context_toks[0]) and not after_double_quote)
                    or left_context_toks[0] == "'" or
                    (after_apostrophe and left_context_toks[0] in ("s", "d", "st", "ve", "re", "nt", "ll", "t", "m"))
                ):
            after_apostrophe = left_context_toks[0] == "'"
            after_double_quote = left_context_toks[0] in ('"', ' "', '▁"', 'Ġ"')
            left_context_toks = left_context_toks[1:]
        left_context_text = tokenizer.convert_tokens_to_string(left_context_toks)
    return out_text

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
            permitted_words = torch.zeros([vocab_size]).to(device)
            for i in range(1, len(required_meter)+1):
                test_meter = required_meter[:i]
                if test_meter in meter_dict:
                    permitted_words += meter_dict[test_meter]
            permitted_words *= 1.0 - word_pieces
            permitted_words = permitted_words.cpu()

            if first:
                probs = permitted_words / sum(permitted_words)
                tok_id = torch.multinomial(probs, 1)
            else:
                _, probs = compute_probs_for_masked_tokens(model, [toks + [{mask_token}]], [[[[len(toks), len(toks)]]]], 1, replacements_only=True)
                probs = probs[0][0][0][0]
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

