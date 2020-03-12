# A Hundred Visions and Revisions

    Tyger Tyger, Flowers lay,
    In the middle of the day;
    Are poetic words just words,
    To give you extra energy?

"A Hundred Visions and Revisions" is a computer program that alters poems using a neural-network language model. It works by replacing the individual words of the text, one by one, with other words that are more probable according to the BERT language model, while preserving rhyme and meter; in effect, this process banalifies the poem, replacing its linguistic distinctiveness with normativity. The program can also attempt to revise a poem to be about a different topic.

As an example, I started with the poem "Jerusalem" by William Blake:

And did those feet in ancient time
Walk upon Englands mountains green:
And was the holy Lamb of God,
On Englands pleasant pastures seen!
 
And did the Countenance Divine,
Shine forth upon our clouded hills?
And was Jerusalem builded here,
Among these dark Satanic Mills?
 
Bring me my Bow of burning gold:
Bring me my arrows of desire:
Bring me my Spear: O clouds unfold!
Bring me my Chariot of fire!
 
I will not cease from Mental Fight,
Nor shall my sword sleep in my hand:
Till we have built Jerusalem,
In Englands green & pleasant Land.

Here is the result after fifty iterations:

And did he who in our time
Shone upon our pleasant land:
And by the holy Grace of God,
Let our pleasant country stand!

And did the Glorious Divine,
Shine forth upon our pleasant land?
And was Jerusalem standing here,
Upon his great Almighty Hand?

Bring me my Sword of shining light:
Bring me my weapon of desire:
Bring me my Spear: O lord above!
Bring me my Instrument of fire!

I will not die from Our Fight,
Nor will my spear be in my hand:
For we have reached Jerusalem,
In our time & pleasant land.

As another example, here is Blake's "O Rose Thou Art Sick":

O Rose thou art sick. 
The invisible worm, 
That flies in the night 
In the howling storm: 

Has found out thy bed 
Of crimson joy: 
And his dark secret love 
Does thy life destroy.

Here is the revision:

By God thou art blessed.
The invisible man,
Who walks in the night
In a hooded cloak:

Has found both his source
Of body heat:
And his own power that
Makes his life complete.

Here is another version, based on different language model called RoBERTa:

They are his cold hands.
the aluminum shards,
his hands in the snow
and the melting ice:

that carve out his heart
from molten clay:
and his cold fingers that
take his life away.

## Changing the topic of a text

It is also possible to have the program revise a poem to be about a different topic while retaining rhyme, meter, and some other, subtler traces of the original. To do this, the program adds extra text at the beginning saying "The following poem is about x," along with a similar statement at the end. The program them computes probabilities both with and without the extra hints and biases the results in favor of words that the hints make more probable.

For example, here are some computer-generated variants of the first stanza of another famous Blake poem, "The Tyger." The following examples were generated with some degree of randomness, which generally leads to better results (in part since it helps the program avoid [local optima](https://en.wikipedia.org/wiki/Local_optimum)); I also set the program to leave the first two words of the text alone.

Original text:

Tyger Tyger, burning bright, 
In the forests of the night; 
What immortal hand or eye, 
Could frame thy fearful symmetry?

topic="strolling through a lovely flower garden":

Tyger Tyger, lily green,
in the center of the scene;
From horizon eye to eye,
You see that lovely gardener?

topic="charging into glorious battle":

Tyger Tyger, shining shield,
In the middle of the field;
Thy opponent eye to eye,
Where lies thy mortal enemy?

topic="the aesthetics of gothic cathedrals":

Tyger Tyger, magnus holm,
And the beauty of the dome;
Does cathedral dome have lights,
Which shine on gothic capitals?

topic="urban planning and traffic management":

Tyger Tyger, traffic light,
In the middle of the night;
When attackers came on foot,
Why break that traffic barrier?

topic="a little, fluffy kitty cat":

Tyger Tyger, kitty cat,
In the posture of a bat;
No protruding fangs or claws,
What is this furry animal?

All of these texts retain the rhyme, meter, and punctuation of the original (excepting the slant-rhyme of "eye" and "symmetry", which the current code cannot detect). If these formal constraints are lifted, the poem will degenerate into prose that bears little relation to the original, a fact best illustrated by the full sequence of steps by which this stanza is transformed into a text about "computational language modeling with artificial neural networks":

tyger tyger, burning bright, in the forests of the night; what immortal hand or eye, could frame such fearful symmetry?
tyger tyger, burning bright, in the forests of the night; what immortal hand or eye, could frame **create** fearful symmetry?
tyger tyger, burning bright, in the forests of the night; what immortal hand or eye, could **you** create fearful symmetry?
tyger tyger, burning bright, in the **middle** of the night; what immortal hand or eye, could you create fearful symmetry?
tyger tyger, burning bright, in the middle of the night; what immortal hand or eye, could you create **such** symmetry?
tyger tyger, burning bright, in the middle of the night; what immortal hand or eye, could you create such **things**?
tyger tyger, burning bright, in the middle of the night; what **artificial** hand or eye, could you create such things?
**john** tyger, burning bright, in the middle of the night; what artificial hand or eye, could you create such things?
john **lennon**, burning bright, in the middle of the night; what artificial hand or eye, could you create such things?
john lennon, burning **bridges**, in the middle of the night; what artificial hand or eye, could you create such things?
john lennon, burning bridges, in the middle of the night; what artificial **ear** or eye, could you create such things?
john lennon, burning bridges, in the middle of the night; **an** artificial ear or eye, could you create such things?
john lennon, burning bridges, in the middle of the night **with** an artificial ear or eye, could you create such things?
john lennon, burning bridges, in the middle of the night with an artificial ear or **something**, could you create such things?
john lennon, burning bridges, in the middle of the night with an artificial **intelligence** or something, could you create such things?
john lennon, **building** bridges, in the middle of the night with an artificial intelligence or something, could you create such things?
john lennon, building **computers**, in the middle of the night with an artificial intelligence or something, could you create such things?
john lennon, **using** computers, in the middle of the night with an artificial intelligence or something, could you create such things?
john lennon, using computers, in the middle of the night with an artificial intelligence or **computer**, could you create such things?
john lennon, using computers, in the middle of the night with an artificial intelligence **working** computer, could you create such things?
john lennon, using computers, in the middle of the night with an artificial intelligence working **nearby**, could you create such things?
john lennon, using computers, in the middle of the night with an artificial intelligence working nearby, could you **predict** such things?
john lennon, using computers, in the middle of the night with an artificial intelligence **expert** nearby, could you predict such things?
john lennon, using computers, in the middle of the night with an artificial intelligence expert **asking**, could you predict such things?
john lennon, using computers, in the middle of the night **hears** an artificial intelligence expert asking, could you predict such things?

## How it works

This program works with the [BERT](https://arxiv.org/pdf/1810.04805v2.pdf) language model, which is based on the [Transformer](https://arxiv.org/pdf/1706.03762.pdf) architecture. (BERT is related to the [GPT-2](https://openai.com/blog/better-language-models/) model used in [Talk to Transformer](https://talktotransformer.com/), although it uses a different network configuration and data set; whereas GPT-2 is trained on text from the internet, BERT is trained on books and Wikipedia articles.)

The BERT model is capable of guessing a word that is "masked"—that is, hidden from the model. To pick an example from the [documentation](https://pytorch.org/hub/huggingface_pytorch-transformers/) for the implementation I used, one could enter "Who was Jim Henson? Jim Henson was a [MASK]"; the model predicts that the masked word is "puppeteer." The point of this is to enable the computer to perform question-answering tasks, language modeling standing as a surrogate for more general intelligence. But it is also possible to use the model's predictions to alter an existing text.

To do this, my program tries masking each word in the text and guessing what word should be in that position. For instance, suppose we are looking at this text:

Tyger Tyger, burning bright, in the forests of the night

We try masking each word in order; for instance, at one point we will end up with this:

Tyger Tyger, burning bright, in the [MASK] of the night

The program uses the neural network to predict what word appears in the masked position, subject to various constraints such as rhyme and meter. In this case, the BERT model guesses "middle," with probability 0.6762. On the other hand, the word that is actually in that position—"forests"—gets probability 0.000076159. We divide the latter by the former to get a score for this 0.0001126. Since this score happens to be the lowest for any word in the text, the program selects this word for replacement, giving us this revision:

Tyger Tyger, burning bright, in the middle of the night

The program then repeats this process until there are no more "improvements" to be made.

To alter the topic of a text, the program adds additional text intended to influence the language model's predictions.

\[The following poem is about flowers:\] Tyger Tyger, burning bright, in the forests of the night \[The preceding poem was about flowers.\]

The brackets indicate that the program is not allowed to alter that text. If the "strong topic bias" feature is turned on, the program computes the probabilities both with and without these annotations and biases the probabilities by the formula ```(probability with annotations / probability without annotations) ** n```, where n is a factor indicating the strength of the topic bias (2.0 is recommended). In this case, the topic annotations cause the program to produce a different prediction:

Tyger Tyger, burning leaves, in the middle of the night

For more details about how it all works, see the code.

I have also experimented with GPT-2, but the results have not been very good. The problem is that, while BERT is able to look both forward and backward when predicting a word, GPT-2 only looks backward; accordingly, it is not good at generating words that fit into a pre-existing structure. I did, however, include a function that generates GPT-2 text constrained by the meter and rhyme scheme of a given poem. The results are so prosaic that it is difficult even to detect the meter, although it is indeed there:

Tyger Tyger, also known
in the English as "the lone
wolf," created this cat, named
"T-Rex," by writing poetry
