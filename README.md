# A Hundred Visions and Revisions

    The recognition of their presence in a tree:
    Sitting on the long, thick branch.

"A Hundred Visions and Revisions" is a computer program that alters poems using a neural-network language model. It works by replacing the individual words of the text, one by one, with other words that are more probable according to the BERT language model, while preserving rhyme and meter; in effect, this process banalifies the poem, replacing its linguistic distinctiveness with normativity. The program can also attempt to revise a poem to be about a different topic.

As an example, I started with the poem "The Sick Rose" by William Blake:

> O Rose thou art sick.  
> The invisible worm,  
> That flies in the night  
> In the howling storm:  
>   
> Has found out thy bed  
> Of crimson joy:  
> And his dark secret love  
> Does thy life destroy.

Here is the revision:

> By God thou art blessed.  
> The invisible man,  
> Who walks in the night  
> In a hooded cloak:  
>   
> Has found both his source  
> Of body heat:  
> And his own power that  
> Makes his life complete.

I have also tried finetuning the neural network on a corpus of about 10,000 poems so as to improve the predictions. Here are the results:

> Thank God you are safe.  
> The emotional wind,  
> That blows in the east  
> In a brutal gale:  
>   
> Tears leaked out like mud  
> From driving rain:  
> In your eyes only strands  
> Of your hair remain.

As you can see, the program can produce very different output depend on how it is set up. Here is an alternative version of "The Sick Rose," based on different language model called RoBERTa:

> They are his cold hands.  
> the aluminum shards,  
> his hands in the snow  
> and the melting ice:  
>   
> that carve out his heart  
> from molten clay:  
> and his cold fingers that  
> take his life away.

As another example, here is Blake's poem "Jerusalem":

> And did those feet in ancient time  
> Walk upon Englands mountains green:  
> And was the holy Lamb of God,  
> On Englands pleasant pastures seen!  
>   
> And did the Countenance Divine,  
> Shine forth upon our clouded hills?  
> And was Jerusalem builded here,  
> Among these dark Satanic Mills?  
>   
> Bring me my Bow of burning gold:  
> Bring me my arrows of desire:  
> Bring me my Spear: O clouds unfold!  
> Bring me my Chariot of fire!  
>   
> I will not cease from Mental Fight,  
> Nor shall my sword sleep in my hand:  
> Till we have built Jerusalem,  
> In Englands green & pleasant Land.

Here is the result after fifty iterations:

> And did he who in our time  
> Shone upon our pleasant land:  
> And by the holy Grace of God,  
> Let our pleasant country stand!  
>   
> And did the Glorious Divine,  
> Shine forth upon our pleasant land?  
> And was Jerusalem standing here,  
> Upon his great Almighty Hand?  
>   
> Bring me my Sword of shining light:  
> Bring me my weapon of desire:  
> Bring me my Spear: O lord above!  
> Bring me my Instrument of fire!  
>   
> I will not die from Our Fight,  
> Nor will my spear be in my hand:    
> For we have reached Jerusalem,    
> In our time & pleasant land.  

## Changing the topic of a text

[UPDATE June 2020: I have redone the "Tyger" examples in this section to use the new, finetuned model.]

It is also possible to have the program revise a poem to be about a different topic while retaining rhyme, meter, and some other, subtler traces of the original. When I created the finetuned neural network, I included annotations indicating the title and author of each poem. This enables the AI to pick up on patterns in the relation between title and poem. You can then feed in hints about the poem's title, and the AI will alter the text accordingly.

For example, here are some computer-generated variants of the first stanza of another famous Blake poem, "The Tyger," with various different titles. The following examples were generated with some degree of randomness, which generally leads to better results (in part since it helps the program avoid [local optima](https://en.wikipedia.org/wiki/Local_optimum)); I also set the program to leave the first two words of the text alone.

Original text:

> Tyger Tyger, burning bright,  
> In the forests of the night;  
> What immortal hand or eye,  
> Could frame thy fearful symmetry?

title="Strolling through a lovely flower garden":

> Tyger Tyger, lovely tree,  
> In a garden by the sea;  
> What admiring ear or eye,  
> Can see such lovely scenery?

title="Charging into glorious battle":

> Tyger Tyger, shining sword,  
> In the battle stands the lord;  
> Yet unable hand or foot,  
> Can fight this mighty warrior?

title="The aesthetics of Gothic cathedrals":

> Tyger Tyger, gothic spire,  
> In the middle of the shire;  
> Which cathedral bore that spire,  
> Which bore that gothic pinnacle?

title="Urban planning and traffic management":

> Tyger Tyger, traffic light,  
> In the middle of the night;  
> Does policeman check his lights,  
> Not check his traffic cameras?

title="A little, fluffy kitty cat":

> Tyger Tyger, kitty cat,  
> In the middle of a chat;  
> What electric freak was he,  
> To send such nasty messages?

title="Leaves of Grass":

> Tyger Tyger, broken glass,  
> In the middle of the grass;  
> Leaves forever blown by wind,  
> How much more petty agony?

As the latter example shows, the model appears to be going off of the words in the title, not recalling any specific information about the titles of pre-existing poems. The titles thus seem to function mainly as topical hints. I need, however, to do more research into exactly what is happening with the titles.

All of these revisions retain the rhyme, meter, and punctuation of the original (excepting the slant-rhyme of "eye" and "symmetry", which the current code cannot detect). If these formal constraints are lifted, the poem will degenerate into prose that bears little relation to the original, a fact best illustrated by the full sequence of steps by which this stanza is transformed into a text about "computational language modeling with artificial neural networks":

> tyger tyger, burning bright, in the forests of the night; what immortal hand or eye, could frame such fearful symmetry?  
> tyger tyger, burning bright, in the forests of the night; what immortal hand or eye, could frame **create** fearful symmetry?  
> tyger tyger, burning bright, in the forests of the night; what immortal hand or eye, could **you** create fearful symmetry?  
> tyger tyger, burning bright, in the **middle** of the night; what immortal hand or eye, could you create fearful symmetry?  
> tyger tyger, burning bright, in the middle of the night; what immortal hand or eye, could you create **such** symmetry?  
> tyger tyger, burning bright, in the middle of the night; what immortal hand or eye, could you create such **things**?  
> tyger tyger, burning bright, in the middle of the night; what **artificial** hand or eye, could you create such things?  
> **john** tyger, burning bright, in the middle of the night; what artificial hand or eye, could you create such things?  
> john **lennon**, burning bright, in the middle of the night; what artificial hand or eye, could you create such things?  
> john lennon, burning **bridges**, in the middle of the night; what artificial hand or eye, could you create such things?  
> john lennon, burning bridges, in the middle of the night; what artificial **ear** or eye, could you create such things?  
> john lennon, burning bridges, in the middle of the night; **an** artificial ear or eye, could you create such things?  
> john lennon, burning bridges, in the middle of the night **with** an artificial ear or eye, could you create such things?  
> john lennon, burning bridges, in the middle of the night with an artificial ear or **something**, could you create such things?  
> john lennon, burning bridges, in the middle of the night with an artificial **intelligence** or something, could you create such things?  
> john lennon, **building** bridges, in the middle of the night with an artificial intelligence or something, could you create such things?  
> john lennon, building **computers**, in the middle of the night with an artificial intelligence or something, could you create such things?  
> john lennon, **using** computers, in the middle of the night with an artificial intelligence or something, could you create such things?  
> john lennon, using computers, in the middle of the night with an artificial intelligence or **computer**, could you create such things?  
> john lennon, using computers, in the middle of the night with an artificial intelligence **working** computer, could you create such things?  
> john lennon, using computers, in the middle of the night with an artificial intelligence working **nearby**, could you create such things?  
> john lennon, using computers, in the middle of the night with an artificial intelligence working nearby, could you **predict** such things?  
> john lennon, using computers, in the middle of the night with an artificial intelligence **expert** nearby, could you predict such things?  
> john lennon, using computers, in the middle of the night with an artificial intelligence expert **asking**, could you predict such things?  
> john lennon, using computers, in the middle of the night **hears** an artificial intelligence expert asking, could you predict such things?

## Changing the author

The finetuned model also incorporates information about the authors of the poem it is trained on. Based on this, it is possible to give the program hints about the author of a poem as well as the title.

For example, here is the first stanza of "I Wandered Lonely as a Cloud" by William Wordsworth:

> I wandered lonely as a cloud  
> That floats on high o'er vales and hills,  
> When all at once I saw a crowd,  
> A host, of golden daffodils,  
> Beside the lake, beneath the trees,  
> Fluttering and dancing in the breeze.

I told the AI that this poem was called "Rime of the Ancient Mariner" by Samuel Taylor Coleridge. Here is its revision:

> He fired blindly at a whale  
> That shot straight out like a dart and died,  
> Till all at once he blew a gale,  
> A blast, till nothing unified,  
> Except the blast, engulfed the whale,  
> Shattering and snapping in the gale.

Entering "The Raven" by Edgar Allan Poe gave very different results:

> While standing staring at a bird  
> I watched it tick like a moth in space,  
> Till all at once I heard a word,  
> A phrase, no longer commonplace,  
> Without a rhyme, without a plot,  
> Fragmented and twisted in a knot.

## BERT-rimés

[Bouts-rimés](https://en.wikipedia.org/wiki/Bouts-Rimés) is an old French game in which one person selects a series of rhyming words and another person composes a poem using them. The idea is to pick words that are tricky to use–words that don't seem to make sense together–so that it is a challenge to create a coherent, natural-sounding poem. BERT, it turns out, is able to play this game, sort of.

Doing this was a little tricky, since BERT is not ideal for generating wholly new text. To start with, I wrote a function that generates words in order, following a specified meter and inserting the rhyming words at the ends of the lines. The results at this point are generally not so good; however, this stage is necessary so as to choose a syllable structure that fits the meter while roughly suiting the patterns of the English language. I then run the revision procedure to turn this initial text into something more coherent.

For example, I entered the rhyming words from the first stanza of "I Wandered Lonely as a Cloud": cloud, hills, crowd, daffodils, trees, breeze. I set the verse form to iambic quadrimeter (which is the form used by the original poem) and also entered the title and author of the original. Here is the almost totally nonsensical initial output:

> Opponents simulator cloud,  
> The education rocked a hills.  
> Recapture to a hui crowd,  
> Royale disposed the daffodils.  
> A vickers macy shrines and trees,  
> And overseeing of a breeze.

Here is the output after 35 rounds of revision:

> Another solitary cloud.  
> The disappearance of the hills.  
> Returning to the evening crowd,  
> Alone among the daffodils,  
> The flowers drifting through the trees,  
> In expectation of a breeze.

As another example, I asked for an iambic pentameter couplet using the rhyming words "storm" and "form." This time I did not specify any title or author. Here is the result:

> Results included project solar storm,  
> The first proposal for another form.

## Modifiers

I also included a feature that enables you to bias the output toward an arbitrary vocabulary. I tested this out using the data from Iain Barr's [analysis of the vocabulary of heavy metal lyrics](https://github.com/ijmbarr/pythonic-metal). Suppose, for instance, "I Wandered Lonely as a Cloud" is not metal enough for your tastes. Perhaps you would prefer this machine-generated alternative:

> Rage flooded slowly through her veins  
> That burned as cold as the sky and ground,  
> Then all at once she dropped her chains,  
> And spit, spit hatred underground,  
> Into her flesh, into the stone,  
> Vibrated and rattled in her bone.

To use this feature, you can run the `generate_modifer.py` script to analyze the vocabulary
of a given text, then supply the resulting JSON file by adding the parameter
`modifier=json_modifier('filename')` to any of the text rewriter functions. For
instance, I used the DeBERTa xxlarge model to generate a version of Alfred, Lord Tennyson's
"The Kraken" using the vocabulary of the PyTorch library documentation. Here
is the original:

> Below the thunders of the upper deep,
> Far, far beneath in the abysmal sea,
> His ancient, dreamless, uninvaded sleep
> The Kraken sleepeth: faintest sunlights flee
> About his shadowy sides; above him swell
> Huge sponges of millennial growth and height;
> And far away into the sickly light,
> From many a wondrous grot and secret cell
> Unnumbered and enormous polypi
> Winnow with giant arms the slumbering green.
> There hath he lain for ages, and will lie
> Battening upon huge sea worms in his sleep,
> Until the latter fire shall heat the deep;
> Then once by man and angels to be seen,
> In roaring he shall rise and on the surface die.

Here is the rewritten version, which replaces the deep sea with deep neural networks:

> Within the matrix of the hidden deep,
> Deep, deep below in the forgotten sea,
> Through hidden, hidden, universal sleep
> The Matrix passes: transient sunlights flee
> Across its infinite folds; within them swell
> Dense layers of infinity width and height;
> And deep below beyond the transient light,
> From many a hidden torch and hidden cell
> Persistent and persistent arises
> Outside each hidden torch the infinite green.
> Thus has it passed for ever, and will lie
> Heavily within its dense folds in deep sleep,
> Until the hidden matrix can float the deep;
> Then known to none and never to be seen,
> In safety it will float and let the matrix die.

## How it works

This program works with the [BERT](https://arxiv.org/pdf/1810.04805v2.pdf) language model, which is based on the [Transformer](https://arxiv.org/pdf/1706.03762.pdf) architecture. (BERT is related to the [GPT-2](https://openai.com/blog/better-language-models/) model used in [Talk to Transformer](https://talktotransformer.com/), although it uses a different network configuration and data set; whereas GPT-2 is trained on text from the internet, BERT is trained on books and Wikipedia articles.)

The BERT model is capable of guessing a word that is "masked"—that is, hidden from the model. To pick an example from the [documentation](https://pytorch.org/hub/huggingface_pytorch-transformers/) for the implementation I used, one could enter "Who was Jim Henson? Jim Henson was a [MASK]"; the model predicts that the masked word is "puppeteer." The point of this is to enable the computer to perform question-answering tasks, language modeling standing as a surrogate for more general intelligence. But it is also possible to use the model's predictions to alter an existing text.

To do this, my program tries masking each word in the text and guessing what word should be in that position. For instance, suppose we are looking at this text:

> Tyger Tyger, burning bright, in the forests of the night

We try masking each word in order; for instance, at one point we will end up with this:

> Tyger Tyger, burning bright, in the [MASK] of the night

The program uses the neural network to predict what word appears in the masked position, subject to various constraints such as rhyme and meter. In this case, the BERT model guesses "middle," with probability 0.6762. On the other hand, the word that is actually in that position—"forests"—gets probability 0.000076159. We divide the latter by the former to get a score for this potential change: 0.0001126. Since this score happens to be the lowest for any word in the text, the program selects the word "forests" for replacement, giving us this revision:

> Tyger Tyger, burning bright, in the middle of the night

The program then repeats this process until there are no more "improvements" to be made.

To alter the topic of a text, the program adds additional text intended to influence the language model's predictions.

> \{The following poem is titled flowers:
> ****\}
> Tyger Tyger, burning bright, in the forests of the night
> \{****
> The preceding poem is by Charles Baudelaire.\}

I also added similar text to the collection of poems on which I finetuned the model, so that the neural network would learn to recognize this type of annotation. The brackets indicate that the program is not allowed to alter that text. If the "strong topic bias" feature is turned on, the program computes the probabilities both with and without these annotations and computes the scores using the formula ```probability with annotations / probability without annotations ** n```, where n is a factor indicating the strength of the topic bias (0.5 is recommended). In this case, the topic annotations cause the program to produce a different prediction:

> Tyger Tyger, burning leaves, in the middle of the night

For more details about how it all works, see the code.

## Other experiments

The program can also perform an alternative procedure that replaces the words in the order in which they appear in the text, rather than choosing which words to replace based on their scores. This is exponentially faster than the default procedure, but the results are generally not as compelling, especially when a topic is specified. This, for instance, is the output for "The Sick Rose":

> Where else thou be thou.  
> The mysterious man,  
> Who slept in a bed  
> In a stormy night:  
>   
> He drew from his heart  
> His secret friend:  
> And whose own secret friend  
> Did his life depend.  

The sequential procedure does a bit better with "Jerusalem":

> And let your God in our hearts  
> Shine upon our pastures clean:  
> And say the mighty Word of Christ,  
> On our fertile pastures green!  
>   
> In may the Glorious Divine,  
> Come down upon many fertile lands?  
> And is Jerusalem hidden here,  
> Beneath these green Eternal Sands?  
>   
> Bring I my Spear of shining light:  
> Bring I my weapon of devotion:  
> Bring I my Sword: Watch it again!  
> Bring I my Instrument of motion!  
>   
> We shall not die of Our Wounds,  
> Nor let my heart rest in your hands:  
> For I have found Jerusalem,  
> And many rich & fertile lands.

I have also done some experiments with GPT-2, although my word-by-word revision technique does not work with GPT-style models. The problem is that, whereas BERT is able to look both forward and backward when predicting a word, GPT-2, like the Angel of History, only looks backward; accordingly, it is not good at generating words that fit into a pre-existing structure. I did, however, include a function that generates GPT-2 text constrained by the meter and rhyme scheme of a given poem. The results are so prosaic that it is difficult even to detect the rhyme and meter, although the output can, indeed, be read with the same rhythms as the original:

> Tyger Tyger, also known  
> in the English as "the lone  
> wolf," created this cat, named  
> "T-Rex," by writing poetry
