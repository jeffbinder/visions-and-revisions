from visions import *
f = open('mobydick.txt', 'r')
text = f.read()
output = banalify(text, window_size=10, context_size=100, max_iterations=500,
             match_meter=False, match_rhyme=False,
             title=None, author=None,
             randomize=False, cooldown=0.01, modifier=None,
             forbid_reversions=True,
             preserve_punctuation=True,
             allow_punctuation=False,
             strong_topic_bias=False, stop_score=1.0,
             model_type='bert-large-uncased-whole-word-masking', model_path=None,
             sequential=False, verbose=False)
of = open('mobydick-banalified-10-100.txt', 'w')
of.write(text)
