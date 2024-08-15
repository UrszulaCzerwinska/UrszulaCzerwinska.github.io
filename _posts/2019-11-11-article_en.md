---
priority: 0.9
title: Named Entity Recognition - keywords detection from Medium articles
excerpt: Demo for EGG Paris 2019 conference - SAEGUS
categories: works
permalink: works/egg_ner
background-image: text_cover.jpg
tags:
  - NLP
  - Python
  - Saegus
  - featured
---

## Introduction

Inspired by a solution developed for a customer in the Pharmaceutical industry,we presented at the [EGG PARIS 2019](https://paris.egg.dataiku.com/)conference an application based on NLP (Natural Language Processing) and developed on a [Dataiku](https://www.dataiku.com/) [DSS](https://www.dataiku.com/dss/) environment.

More precisely, we trained a deep learning model to recognize the keywords of a blog article, precisely from [Medium blogging platform](https://medium.com/).

This solution applied to blog articles can be used to **automatically generate tags and/or keywords** so that the content offered by the platforms is personalized and meets readers' expectations.

In a broad sense, entity detection allows automated and intelligent text analysis, especially useful for long and complex documents such as scientific or legal pieces.

To demonstrate its use, we have integrated a voice command, based on [Azure's cognitive services API](https://azure.microsoft.com/en-us/services/cognitive-services/). The *speech to text* module allows to return query text as input to the algorithm. The output is represented as a recommendation of articles, classified by relevance according to the field of research.

This article explains our approach to creating the underlying NLP model.

<p align="center"><iframe width="560" height="315" src="https://www.youtube.com/embed/zg0pTe-GyF0" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></p>
<p align="center"><sub><sup>[to view the comments, please enable subtitles] A video that illustrates our web application  created for the EGG Dataiku 2019 conference</sup></sub></p>

***

## Why extract keywords from Medium blog articles?

Medium has two categorization systems: **tags** and **topics**.

The topics are imposed by the platform. The author cannot select his topic. They correspond to fairly generic categories such as data science or machine learning.

The tags, a maximum of 5 per article, are essentially keywords that the author decides to list under his post in order to make it visible. Thus an article may have tags that may have nothing to do with the content of the article/history.

If you label it with common terms, such as "TECHNOLOGY" or "MINDFULNESS" or "LOVE" or "LIFE LESSONS", your item will be easier to find. But it makes life more difficult for a reader who is looking for a specific subject.

We will therefore try to self-tag articles to increase their relevance.

Thanks to these "new tags", or "keywords", we could quickly search for the articles that mention them and thus increase the effectiveness of our search.

We could go even further and build a recommendation system, by advising articles close to the one we are reading, or by advising ourselves of new articles related to our reading habits.

## The NER (Named Entity Recognition) approach

Using the NER (Named Entity Recognition) approach, it is possible to extract entities from different categories. There are several basic  pre-trained models, such as [en_core_web_md](https://github.com/explosion/spacy-models/releases/tag/en_core_web_md-2.2.0), which is able to recognize people, places, dates...

Let's take the example of the sentence *I think Barack Obama puts founder of Facebook at occasion of a release of a new NLP algorithm*. The en_core_web_md model detects Facebook and Barack Obama as entities.

<script src="https://gist.github.com/UrszulaCzerwinska/11a8fab0cc4c936b67e374e2b55e0fa0.js"></script>

<div><span class="image fit"><img src="{{ site.baseurl }}/images/NER_img1.png" alt=""></span></div>
<p align="center"><sub><sup>Dependency graph: result of line 9 (# 1)</sup></sub></p>

<div><span class="image fit"><img src="{{ site.baseurl }}/images/NER_img2.png" alt=""></span></div>
<p align="center"><sub><sup>Entity detection: result of line 10 (# 2)</sup></sub></p>

In our use case : extracting topics from Medium articles, we would like the model to recognize an additional entity in the "TOPIC" category: "NLP algorithm".

With some annotated data we can "teach" the algorithm to detect a new type of entities.

The idea is simple: an article tagged Data Science, AI, Machine Learning, Python can concern very different technologies. Our algorithm would thus be able to detect a specific technology cited in the article, for example GAN, reinforcement learning, or the names of python libraries used. It also retains the ability of a basic model to recognize places, names of organizations and names of people.

During training, the model learns to recognize the keywords, without knowing them a priori. The model will be able to recognize for example the topic: random forest without even being present in the learning data. Based on articles that discuss other algorithms (e.g. linear regression), the NER model will be able to recognize the phrase turn of phrase that indicates that we are talking about an algorithm.

## The model

### SpaCy Framework

[SpaCy](https://spacy.io/) is an open-source library for advanced natural language processing in Python. It is designed specifically for use in production and helps to build applications that handle large volumes of text. It can be used to build information extraction systems, natural language comprehension systems or text preprocessing systems for in-depth learning. Among the functions offered by SpaCy are: Tokenization, Parts-of-Speech (PoS) Tagging, Text Classification and Named Entity Recognition.

SpaCy provides an exceptionally efficient statistical system for NER in python. In addition to entities included by default, SpaCy also gives us the freedom to add arbitrary classes to the NER model, training the model to update it with new examples formed.
SpaCy's NER model is based on **CNN** (**Convolutional Neural Networks**).
For the curious, the details of how SpaCy's NER model works are explained in the video:

<p align="center"><iframe width="560" height="315" src="https://www.youtube.com/embed/sqDHBH9IjRU" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></p>

### Training data

To start training the model to recognize tech keywords, we recovered some Medium articles through **web scraping**.

<script src="https://gist.github.com/UrszulaCzerwinska/db0aa37b1cb10ec94205d847f63ddc4f.js"></script>


<div><span class="image fit"><img src="{{ site.baseurl }}/images/NER_img3.png" alt=""></span></div>
<p align="center"><sub><sup>An extract from the table containing the contents of the medium articles</sup></sub></p>

The text of each article has been divided into sentences to facilitate annotation.

There are annotation tools for NER such as Prodigy or other, mentioned here. We used a simple spreadsheet and in the dedicated columns we marked the entities.

<div><span class="image fit"><img src="{{ site.baseurl }}/images/NER_img4.png" alt=""></span></div>

To give an idea of the volume required, with about twenty articles (~600 sentences) our model started to show an interesting performance (>0.78 accuracy on test set).
The train and test data were separated to evaluate the model.

<div><span class="image fit"><img src="{{ site.baseurl }}/images/NER_img5.png" alt=""></span></div>


{% highlight python %}
TRAIN_DATA_ALL =list(train_table.apply(lambda x : mark_targets(x, ['ORG', 'PERSON', 'LOC', 'TOPIC', 'GPE','DATE', 'EVENT', 'WORK_OF_ART'], "sents", ['ORG', 'PERSON', 'LOC', 'TOPIC', 'GPE','DATE', 'EVENT', 'WORK_OF_ART']), axis=1))
{% endhighlight %}



<div><span class="image fit"><img src="{{ site.baseurl }}/images/NER_img6.png" alt=""></span></div>

Then, we fine-tuned our algorithm by playing on several parameters: number of iterations, drop rate, learning rate and batch size.


### Model assesment

In addition to the loss metric of the model, we have implemented the indicators: precision, recall and F1 score, to more accurately measure the performance of our model.


<script src="https://gist.github.com/UrszulaCzerwinska/c23ce9e0edffe6f9790a2bbf8f018a4b.js"></script>

Once trained on all the annotated data, the performance of the best model on our test set was quite impressive. Especially if we take into account the modest size of the train data: ~3000 sentences.

{% highlight bash %}
precision :  0.9588053949903661
recall :  0.9211764705882353
f1_score :  0.9396221959858323

It is is designed to interoperate with the Python numerical and scientific libraries NumPy and SciPy.

TOPIC Python
TOPIC NumPy
TOPIC SciPy
{% endhighlight %}

In the **Flow** on DSS, the process can be summarized by the graph:

<div><span class="image fit"><img src="{{ site.baseurl }}/images/NER_img7.png" alt=""></span></div>
<p align="center"><sub><sup>Flow on Dataiku's DSS platform: the annotated dataset is divided into train and test, the model learned on the train data is evaluated on the train and test batches.</sup></sub></p>

To return to the example on Barack Obama, our algorithm detects the *NLP algorithm* entity as TOPIC in addition to the ORG (organization) and PERSON entities.
We have succeeded! 🚀

<div><span class="image fit"><img src="{{ site.baseurl }}/images/NER_img8.png" alt=""></span></div>

The finalized model can be compiled as an independent python library (instructions here) and installed with `pip`. This is very practical for deploying the model in another environment and for production setup.


<div><span class="image fit"><img src="{{ site.baseurl }}/images/NER_img9.jpg" alt=""></span></div>


## Exploitation of the model

### Analysis of an article Medium

In our mini webapp, presented at the EGG, it is possible to display the most frequent entities of a Medium article.

Thus, for the article: [https://towardsdatascience.com/cat-dog-or-elon-musk-145658489730](https://towardsdatascience.com/cat-dog-or-elon-musk-145658489730), the most frequent entities were: model, MobileNet, Transfer learning, network, Python. We also detected people: Elon Musk, Marshal McLuhan and organizations: Google, Google Brain.

<div><span class="image fit"><img src="{{ site.baseurl }}/images/NER_img10.png" alt=""></span></div>


Inspired by [Xu LIANG's](https://towardsdatascience.com/@bramblexu) [post](https://towardsdatascience.com/textrank-for-keyword-extraction-by-python-c0bae21bcec0), we also used his way of representing the relationship between words in the form of a graph of linguistic dependencies. Unlike in his method, we did not use TextRank or TFIDF to detect keywords but we only applied our pre-trained NER model.

Then, like [Xu LIANG](https://towardsdatascience.com/@bramblexu), we used the capacity of Parts-of-Speech (PoS) Tagging, inherited by our model from the original model ([en_core_web_md](https://github.com/explosion/spacy-models/releases/tag/en_core_web_md-2.2.0)), to link the entities together with the edges, which forms the graph below.


<div><span class="image fit"><img src="{{ site.baseurl }}/images/NER_img11.png" alt=""></span></div>
<sub><sup>The graph of dependencies between the entities detected in the article "Cat, Dog, or Elon Musk?"</sup></sub>


Thus, we get a graph where the keywords are placed around their category: Tech topic, Person and Organization.

This gives a quick overview of the content of a Medium article.

Here is how to get the graph from a Medium article url link:

<script src="https://gist.github.com/UrszulaCzerwinska/d1d77f0bf8bd089103994eb3883db28f.js"></script>

## To go further
Our Saegus Showroom including the functional webapp is coming soon. Feel free to follow our page [https://medium.com/data-by-saegus](https://medium.com/data-by-saegus) to be kept informed.

**The project we have outlined here can easily be transposed into the various fields of industry: technical, legal and medical documents. It could be very interesting to analyse the civil, criminal and law... with this approach for a better efficiency in the research that all legal professionals do.**

## *Disclaimer*

This article is a result of a teamwork realized at [Saegus](http://saegus.com/fr/). Published originally in French at [Medium](https://medium.com/data-by-saegus/ner-medium-articles-saegus-7ffec0f3188c).


<section>
<!-- Go to www.addthis.com/dashboard to customize your tools --> <script type="text/javascript" src="//s7.addthis.com/js/300/addthis_widget.js#pubid=ra-584ec4ce89deed84"></script>
<!-- Go to www.addthis.com/dashboard to customize your tools --> <div class="addthis_inline_share_toolbox"></div>
<section>

