---
priority: 0.9
title: GenAI Workshop: Insights on implementing RAG in production environments
excerpt: My experience with Generative AI and Retrieval-Augmented Generation through a hands-on workshop.
permalink: /works/genai-workshop-implementation
categories: works
background-image: genai-course-background.jpg
published: false
tags:
  - GenerativeAI
  - RAG
  - DeepLearning
  - AI
  - LLM
  - AITraining
  - Embeddings
  - featured
author: urszulaczerwinska
description: An overview of the GenAI course and workshop I attended, focusing on the challenges and strategies for implementing RAG in production environments, optimizing performance, and evaluating the results.
---

<head>
  <meta name="description" content="An overview of the GenAI course and workshop I attended, focusing on the challenges and strategies for implementing RAG in production environments, optimizing performance, and evaluating the results.">
</head>

## My experience at the GenAI course: Implementing RAG in production environments

This past quarter, I had the opportunity to participate in a comprehensive internal Generative AI training course of [Adevinta](https://adevinta.com/) and [Leboncoin](https://leboncoincorporate.com/), which culminated in a two-day workshop focused on hands-on applications. The training covered a wide array of topics, with the key focus on **Retrieval-Augmented Generation (RAG)**, an emerging approach in leveraging large language models (LLMs) in real-world applications.

Adevinta is a global classifieds specialist with market-leading positions in key European markets that aims to find perfect matches between its users and the platforms’ goods.

Before this workshop, I had encountered RAG in research papers, but seeing it in action, especially when applied to real-world problems, was a valuable experience. I learned that creating a flashy demo is one thing, but scaling that demo to work in production environments and taking into account ethical considerations is not as easy. Here's a closer look at what we experienced and the lessons learned along the way.


### 1. **Overview of the GenAI Training: Structure and Goals**


The course was split into three main segments:

1. **Self-paced online learning modules** designed to introduce foundational concepts of GenAI.
2. **Customized live sessions** focused on addressing specific use cases relevant to participants' day-to-day work.
3. A **two-day intensive workshop**, where we were tasked with building Minimum Viable Products (MVPs) using RAG and other GenAI technologies.


The course was divided into three sections. The first involved self-paced learning, where I had to work through a series of online modules on AI and machine learning.

The courses suggested in the self-paced learning modules were:
1. How Diffusion Models Work - free course
2. Generative AI with LLMs - Coursera 45 eur


#### My feedback on ["How Diffusion Models Work"](https://www.deeplearning.ai/short-courses/how-diffusion-models-work/) course
This a flash course covers basics of Diffusion Models, introducing simple code script explaining the magic of going from the noise to a brand new generated image, image of a sprite. It also explains the UNet architecture, the use of DDPM noise schedule and possible optimization with DDIM noise. It also gives a primer on controlling the network with classes vector. Even thought in a very short time it covers lot of topic, I personally found it quite confusing. It does covers lots of topics but the "artificial" setup of sprite generation left me we with little understanding how to make the next step towards real world application. I would recommend this course to someone who is looking for a quick overview of the topic, but not to someone who is looking for a deep dive into the topic.

If you are looking for a free and more complete course on Diffusion and Stable Diffusion, I highly recommend you the [Huggingface stable diffusion course](https://huggingface.co/learn/diffusion-course/en/unit0/1). It is longer and "slower" pace but it gives you a better understanding of the topic, imo.

#### My feedback on ["Generative AI with LLMs"](https://www.coursera.org/learn/generative-ai-with-llms) course

I happily received a certificate of accomplishing this course. It gives a good overview of LLM architectures, explaining transformers, attention and text generation. Explanations are clear and it is a good refresher on the topic. There was also an introduction to prompting.

From more advanced topics, I appreciated the explanation of PEFT with LoRA but also Soft tuning that I discoverd in that course. I understood way lass the module on RLHF and the practical use of it. Finally the use of chain of thoughts was a good primer.

It uses Bedrock with AWS to run the labs. I found the labs the least interesting part of the course. The setup was easy and smooth but the use cases seemed "too easy". Using only models and examples adapted to LoRA or PEFT out of the box. The data was already prepared and there was no need to do any data preprocessing. I would have appreciated more real world examples and more complex use cases. Running the labs was, in my opinion, not very useful and reduced to executing a few jupyter cells with not much understanding of what is happening in the background.

#### The practical, in house, curriculum

While overall these were informative, I found the real learning happened during the second segment, where we had live sessions tailored to the specific problems we faced in our own work environments. Here, discussions around industry-specific challenges helped me think about how the theory behind RAG could translate into everyday business problems.


1. Using GenAI in Adevinta (our company)
2. Prompt Engineering
3. Using Bedrock with langchain
4. Retrieval augmented generation
5. Agents
6. Security and risks

Throughout the sessions, we worked with tools such as [**LangChain**](https://www.langchain.com/), **LLM APIs** such as [Amazon Bedrock](https://aws.amazon.com/fr/bedrock/?gclid=Cj0KCQjw3bm3BhDJARIsAKnHoVUupzZaHfVWRIQi8FvbGme27pzjfIrGOiLeqJ713jiTHLw1ujZj6NEaAu8JEALw_wcB&trk=f1f5028e-0107-40fd-a47b-6bf2ad7d99f5&sc_channel=ps&ef_id=Cj0KCQjw3bm3BhDJARIsAKnHoVUupzZaHfVWRIQi8FvbGme27pzjfIrGOiLeqJ713jiTHLw1ujZj6NEaAu8JEALw_wcB:G:s&s_kwcid=AL!4422!3!692062117823!e!!g!!bedrock!21054970526!158684164545), and integrated **embeddings**. While the self-paced learning helped us grasp the basics, the live sessions were where we could apply these concepts to real-world scenarios.
I am really grateful to our teachers, our colleagues and the company for giving us the opportunity to learn and apply these new technologies. The content was well-prepared and delivered with passion and patience.


#### The two-day workshop

The final part, a two-day workshop, pushed us out of our comfort zones. We were tasked diving deep in one of the concepts. My team focused on building a RAG to retrieve data from additional sources and applying different RAG techniques. By the end of the first day, we had something functional—but it was far from production-ready and we had lots of ideas. We spent most of our time improving the solution, which taught me that even the smallest bottlenecks can make or break an AI system.

Some other projects developed by my colleagues were:
- SAFA, a data analyst assistant to answer to fraud questions .
- filling out a classified Ad from their photos.
- scoring the match between an Ad and a CV
- system computing how good a RAG system is.
- a data analyst assistant



##### **What is RAG and Why Is It Important for GenAI?**

Let's step back to clarify the concept of RAG and its significance in the GenAI landscape.

Retrieval-Augmented Generation is a hybrid approach that combines the power of **retrieval systems** with **generative models** to enhance the accuracy and reliability of large language models. Unlike pure generative models that sometimes hallucinate or provide irrelevant information, RAG utilizes an external knowledge base to retrieve contextually relevant information, feeding it into a language model for generation.

RAG is especially useful as **question-answering systems** and use cases that require accurate, up-to-date, and verifiable information. In our company, my colleagues build "ADA bot" that can answer user questions in a safe environment that prevents data leakage to LLM providers. It is also connected to some of the internal databases to provide the most up-to-date information on specific internal topics.

To learn more about rag you can have a look into these ressources:
- [What is RAG ?](https://www.datacamp.com/blog/what-is-retrieval-augmented-generation-rag)
- [RAG by AWS](https://aws.amazon.com/what-is/retrieval-augmented-generation/)
- [IBM RAG](https://research.ibm.com/blog/retrieval-augmented-generation-RAG)

<p align="center"><iframe width="560" height="315" src="https://www.youtube.com/embed/T-D1OfcDW1M?si=vZ2HQE9-yxGDdMJL" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></p>
<p align="center"><sub><sup>What is RAG ? Video by [IBM](https://research.ibm.com/blog/retrieval-augmented-generation-RAG)</sup></sub></p>

#### The two-day workshop RAG project

In the workshop we first added selected content from three different sources: company Github, company internal Confluence, and company intranet. We used the LangChain to connect to these sources.

We also generated with LLM Claude set of QA about each added document to be able to test the retrieval afterwards. Already at this step some prompt engineering was needed to make sure the QA were relevant.

We experimented three approaches to RAG:
- simple RAG with basic prompt searching in all connected sources
- LLM classified RAG with a classifier that would decide which source to search
- ReRanker RAG that would search in all sources and then rerank the results

We were highly inspired by [Nir Diamant repo](https://github.com/NirDiamant/RAG_Techniques/tree/main)

As over two days we need to focus on specific optimization axes we decided not to spend too much time on prompt engineering or testing different LLM models. We used "anthropic.claude-3-haiku-20240307-v1:0" as our LLM model and "cohere.embed-multilingual-v3" for embeddings as there were proof tested before as good for our use case.

We used existing [streamlit](https://streamlit.io/) interface of ADA bot to display the results adding our RAG-augmented LLM version.

As the outcome, we managed to index a few dozens of pages from selected sources the simple rag would work correctly searching in a common database.

Then we implemented the idea of classifying the question to indicate the right source. This could have advantages of optimization and relevance improvement. We tested two ways of question classification : with an LLM model and with a BERT classifier. LLM prompt was challenging but we ended up with >80% precision. The classifier was easier to implement and gave similar precision with a short training time. Both options seems to be valid for our use case and could be used depending on the resources available and the overall system design. In the real world, we would need to find a solution for a question that needs several sources to be searched.

Finally, we implemented the Reranker. The reranker was able to rerank the results. The reranker was able to rerank the results based on the relevance of the sources and the questions. The results were better than the simple rag and the classifier.


### 2. **Challenges in Scaling GenAI and RAG Applications**

One of the main takeaways from the workshop was the difficulty in scaling GenAI applications, particularly RAG systems. While creating a POC was straightforward, thanks to frameworks like **LangChain**, taking it to a production-ready solution would be a whole different challenge.

Some of the critical challenges in scaling GenAI applications include:

#### **Data availability and retrieval performance**

Effective RAG systems depend on robust, well-maintained knowledge bases. Scaling such systems requires integrating large and frequently updated datasets, which increases the complexity of both **retrieval speed** and **index management**. Throwing in not curated and not cleaned data can lead to garbage in garbage out. The way data can be cleaned is not fully solved, especially if we are talking of years of company knowledgebase or codebase. I

In any production-level AI system, performance optimization is crucial. Here are some strategies that can be explored for improving the efficiency and reliability of RAG-based systems:
- **Using pre-computed embeddings**: To minimize computational overhead during inference, a solution could be to pre-computed embeddings for frequently accessed data. This allows faster lookup times when retrieving relevant documents.
- **Batch processing**: Rather than processing every query individually, batching similar queries allows to take advantage of parallelism, significantly speeding up the retrieval process.
- **Indexing strategies**: Different indexing algorithms, such as **Faiss** and **Annoy**, can optimize how the knowledge base is queried. The right indexing technique can drastically reduce the time required for retrieving documents from large datasets.

If you are interested in the topic, I recommend you to have a look at [this article](https://arxiv.org/abs/2403.09727) and also [this overview](https://hyperight.com/6-ways-for-optimizing-rag-performance/).

#### **Mitigating hallucinations**

Even though RAG reduces hallucinations by leveraging external data, managing these risks becomes more challenging when scaling across different industries and datasets. Fact-checking and ensuring the relevance of retrieved information is not difficult in a sandbox environment but much more difficult with billions of documents. Here is some reading on this topic: [RAG Hallucination: What is It and How to Avoid It](https://www.k2view.com/blog/rag-hallucination/), [Tricks to reduce RAG hallucinations](https://www.wired.com/story/reduce-ai-hallucinations-with-rag/), [RAGs Do Not Reduce Hallucinations in LLMs — A Math Deep Dive](https://medium.com/autonomous-agents/rag-does-not-reduce-hallucinations-in-llms-math-deep-dive-900107671e10)
#### **Adversarial attacks**
 During the workshop, we touched on the importance of securing models against adversarial prompts that could manipulate the retrieval mechanism, potentially leading to incorrect or malicious outputs. This can be a real challenge in a company setup, especially if the bot is connected to the confidential information. In the course we haven several approaches to test-proof the system against adversarial attacks, but it is still a challenge to make sure the system is secure. Here is [a great in depth article in this topic](https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/).
#### **Scoring and evaluating RAG system**
We also discussed the importance of evaluating the performance of RAG systems, particularly in real-world scenarios. This involves measuring the relevance of retrieved information, response accuracy, and latency under load. We tested a few ways to evaluate the system, mostly through another LLM designed to evaluate the system. The pitfall is that this is kind of evaluation is resource consuming and not always possible in a real world setup. Also we were not sure if we can really trust an LLM evaluation. I have explored some litterature on this topic and I found [this article on Top 5 Open-Source LLM Evaluation Frameworks (2024)](https://dev.to/guybuildingai/-top-5-open-source-llm-evaluation-frameworks-in-2024-98m) very interesting. Also [this article that discuss the traditional evaluations like BLEU or ROUGE](https://www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation). Finally [this article](https://eugeneyan.com/writing/llm-evaluators/?utm_source=brevo&utm_campaign=december_23_newsletter&utm_medium=email) that emphasizes the importance of the human alignement.

### 3. **Implementing RAG workshop: lessons learned**

During the two-day project sprint, my team worked on developing a **data retrieval assistant** applying different RAG strategies. We imagined several challenges when transitioning from POC to a more robust solution.

Getting a POC up and running with hashtag#LangChain and LLM API is actually easier than you might think.

However, taking that to the next level—building a robust, production-ready solution that avoids hallucinations and can handle adversarial attacks—is a whole new challenge.

Taken into account the overall panorama of presented projects, compared to those early post-ChatGPT hacathons, today’s approaches are more mature. We're now solving real-world problems rather than just chasing the shiny tech. More precisely, we’ve got a clearer picture of what LLMs can do—and what they can’t.

### 4. **Ethical Considerations and AI Governance**

AI ethics played a significant role in our discussions, particularly around how to manage the risks of bias and data privacy in RAG systems. For example, we had to think about how biased data in a knowledge base could influence the model’s outputs. To address this, we discussed [methods for detecting and mitigating bias](https://shelf.io/blog/10-step-rag-system-audit-to-eradicate-bias-and-toxicity/) during the retrieval process.

We also considered the issue of data privacy. Retrieving sensitive information, especially in fields like healthcare or finance, can raise significant privacy concerns. Proper anonymization and data governance strategies are essential to ensure that personal or confidential data is protected.

### 8. **Final Thoughts: Moving beyond the workshop**

The workshop gave me a clearer understanding of the practical challenges associated with RAG and other GenAI applications. Implementing these systems in real-world production environments is far more nuanced than building a POC. The key lies in continuous **performance optimization**, **scaling**, and **ethical governance**. As GenAI technologies evolve, I’m excited to continue exploring how these advancements can be applied to solve complex, real-world problems.

---

### Frequently Asked Questions (FAQs)

1. **How can you implement RAG in LLMs?**
   - RAG can be implemented by combining a retrieval mechanism with a generative model, enabling the model to fetch relevant information from external databases before generating a response.

2. **What are the challenges in scaling GenAI applications?**
   - Major challenges include handling large-scale data retrieval, maintaining data consistency, avoiding hallucinations, and securing the system against adversarial attacks.

3. **How do you optimize RAG systems for better performance?**
   - Performance optimization strategies include pre-computing embeddings, batch processing, and selecting the appropriate indexing algorithms for faster data retrieval.

4. **How do you evaluate the performance of RAG systems?**
   - Evaluation focuses on response accuracy, relevance of retrieved information, latency under load, and error analysis.

5. **Can RAG be used for question-answering systems?**
   - Yes, RAG is particularly effective for question-answering systems that need to pull relevant data from knowledge bases to provide accurate, up-to-date answers.

6. **What are the ethical concerns when implementing RAG systems?**
   - Ethical concerns include the risk of bias in retrieved data, privacy issues when handling sensitive information, and the need for robust AI governance.


<footer>
  <p>Exported from <a href="https://medium.com">Medium</a> on June 06,
    2023.</p>
  <p><a
      href="https://medium.com/adevinta-tech-blog/deep-dive-in-paddleocr-inference-e86f618a0937">View
      the original. This article was orignally co-authored by Cognition team members, special credits to Joaquin Cabezas</a></p>
</footer>
<script type="text/javascript"
  src="//s7.addthis.com/js/300/addthis_widget.js#pubid=ra-584ec4ce89deed84"></script>
<div class="addthis_inline_share_toolbox"></div>

