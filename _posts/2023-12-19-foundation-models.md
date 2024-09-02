---
title: AI Foundation Models
categories: thoughts
categories:
      - thoughts
      - featured
permalink: /thoughts/foundation-models-a-new-vision-for-e-commerce
excerpt: A new vision for E-commerce
author: urszulaczerwinska
icon: star
date: 2023-12-19
background-image: foundation_models.jpg
description: An in-depth exploration of AI foundation models and their role in transforming e-commerce, including real-world applications, challenges, and strategies for leveraging these powerful tools.
---
<head>
  <meta name="description" content="Discover how AI foundation models are revolutionizing e-commerce by enhancing product development, driving sustainability, and fostering collaboration. Explore the challenges and strategies for successful implementation.">
</head>


> **_“Foundation models are models that are trained on broad data and can be adapted to a wide range of downstream tasks. (…) In choosing this term, we take ‘foundation’ to designate the function of these models: a foundation is built first and it alone is fundamentally unfinished, requiring (possibly substantial) subsequent building to be useful. ‘Foundation’ also conveys the gravity of building durable, robust, and reliable bedrock through deliberate and judicious action.”_**
>  — the Stanford Institute for Human-Centred AI founded the [Center for Research on Foundation Models](https://arxiv.org/pdf/2108.07258.pdf) (CRFM), 2021

<span class="image fit">
![Members of the Cognition team at Adevinta discussing the latest trends in AI and computer vision.](https://cdn-images-1.medium.com/max/800/0*PDC2WWQyZPpIcKZU)
</span>

In the competitive realm of digital commerce, embracing technological advancements is not a luxury but a necessity for maintaining success. Among ML tools, foundation models are emerging as a formidable force. But what are foundation models, and why have they become a focal point among technologists and business leaders?

As members of ‘Cognition’, a team dedicated to computer vision at [Adevinta](https://adevinta.com/), we are eager to share the insights we have gathered through our technology trends watch and recent attendance of the [ICCV 2023](https://iccv2023.thecvf.com/) conference. This practice not only ensures our internal services remain up to date but also improves the standards experienced by our users, enhancing the services provided to customers across [Adevinta’s brands](https://adevinta.com/our-brands).

## What are foundation models ?

**Foundation models** are a breed of artificial intelligence (AI) models pre-trained on a vast amount of data, laying a robust groundwork for further customisation on specific tasks.

Unlike traditional machine learning models, which require from-scratch training or fine tuning for every new task, **foundation models offer a substantial head start**. They have already learned a good deal from the data they were initially trained on, which includes recognising patterns, objects, and in the domain of computer vision, even understanding the semantics of a scene.

Foundation models can be leveraged in various ways, each with its own balance of resource consumption and performance enhancement. The most resource-efficient method involves extracting features from an image, “freezing” them, and then using them directly as a zero-shot retrieval, classifier or detector. This zero-shot approach requires no further learning, allowing for immediate application.

Alternatively, these embeddings can serve as inputs to other models, such as an MLP or an XGBoost classifier, through transfer learning. This strategy necessitates a minimal training dataset, yet it remains swift and cost-effective. [Pastore et al](https://arxiv.org/abs/2209.07932) reported that there can be **10x to 100x speed increase** coupled with limited accuracy decrease (1–5% on average), depending on the dataset, when using a kernel classifier on top of frozen features. For a well-known CIFAR100 dataset, the authors observed 10x to 12x speed increase and −3.70% accuracy decrease. From our preliminary experiments, preparing for deploying image embedding services for Adevinta marketplaces, we noted a 5x to 10x speed increase with less than 3% accuracy drop for ImageNet1K dataset with Dinov2 frozen features compared to fine tuning a CNN backbone.

For those seeking even greater performance enhancements, fine-tuning either the last layers or the entire network is an option. This process may demand a deeper understanding of machine learning and a larger dataset for model refinement, but can lead to substantial improvements. A key challenge in this approach is maintaining the model’s generalisability and preventing it from “forgetting” previously learned datasets.

<span class="image fit">
![A visual representation of the AI model development process, highlighting the efficiency gained by using foundation models.](https://cdn-images-1.medium.com/max/800/0*yvtVGxc_UkJgw2q6)
</span>

To further enrich the model with bespoke data, one can explore self-supervised learning techniques for pre-training or distillation on domain-specific data. Moreover, to ensure the model remains current with new data, continuous learning methodologies can be employed. These advanced techniques not only enhance the model’s performance, but also tailor it more closely to specific business needs and data environments.

<span class="image fit">
![An illustration of the fine-tuning process in AI model development, emphasizing the balance between performance and resource use](https://cdn-images-1.medium.com/max/800/0*2ccVFc0mWHKRw6L0)
</span>

Beyond the singular applications of foundation models lies the potential for a transformative synergy. By harnessing models trained on diverse datasets with various loss functions, we can unlock new heights of performance. This approach was masterfully demonstrated by [Krishnan et al](https://arxiv.org/abs/2306.00984?utm_campaign=The%20Batch&utm_medium=email&_hsmi=281785463&utm_content=281787502&utm_source=hs_email), who capitalised on images synthesised by Stable Diffusion. They adeptly trained another model (StableRep) using a contrastive loss approach and ViT backbone to achieve remarkable success in a classification benchmark. This strategy showcases the innovative fusion of generative and discriminative model capabilities, setting a new standard for adaptive and robust AI applications.

> **_“Different foundation models understand different aspects of the world. It’s exciting that a large diffusion model, which is good at generating images, can be used to train a large vision transformer, which is good at analysing images!”_**

> **_—_** [The Batch @Deeplearning.ai newsletter](https://info.deeplearning.ai/openai-empowers-developers-ai-risk-in-the-spotlight-decoding-schizophrenic-language-synthetic-data-helps-image-classification-1?ecid=ACsprvum6jLSdI3_MtO8GVvBlJrsfj1iNuU7d7wJk3k6DmAu6jDwlmqxh0ZqOYQpd5P6T9SkgLDq&utm_campaign=The%20Batch&utm_medium=email&_hsmi=281785463&_hsenc=p2ANqtz-_AfPCrj7xxHY4m3H4td4jKSdynMLio8p3y-HqpQE0KbMIn5qoGh6dicKnKqf-6eVEcThLfdSR4_uMpwahHLcZqGQKfVg&utm_content=281787502&utm_source=hs_email)

## The rise of foundation models

The history of foundation models is closely tied to the rise of deep learning, particularly with the advent of large-scale models like GPT (Generative Pre-trained Transformer) by OpenAI and BERT (Bidirectional Encoder Representations from Transformers) by Google, which demonstrated the feasibility and effectiveness of pre-training models on vast datasets and then fine-tuning them for specific tasks.

As technology advanced, so did the scale and capabilities of these models, with models like GPT-3, GPT-4 and T5 showcasing unprecedented levels of generalisation and adaptability across numerous domains including natural language processing, computer vision, and even multimodal tasks combining both vision and text. The success of these models started **a new era where the focus shifted from training task-specific models from scratch to developing robust, versatile foundation models.** This new type of model could be fine-tuned or used in transfer-learning to excel at a broad spectrum of tasks. This shift not only catalysed significant advancements in AI research but also broadened adoption of AI across various industries, paving the way for more sophisticated and capable foundation models that continue to push the boundaries of what’s achievable with Artificial Intelligence.

Notable examples of foundation models abound in the tech landscape. For instance, DINOv2 and MAE (Masked Autoencoder) by Meta AI for image understanding. On the other hand, models like CLIP and BLIP from OpenAI have shown the potential of bridging the gap between vision and language. These models, pre-trained on diverse and voluminous datasets, encapsulate a broad spectrum of knowledge that can be adapted for more specialised tasks, making them particularly advantageous for industries with data-rich environments like e-commerce.

<span class="image fit">
![An illustration demonstrating the innovative use of foundation models to achieve advanced AI capabilities.](https://cdn-images-1.medium.com/max/800/0*EHZAOKUjrq7Sz6uP)
</span>

Here is a short description of a few of those models:

**DINOv2:** Developed by Meta, [DINOv2](https://ai.meta.com/blog/dino-v2-computer-vision-self-supervised-learning/) is recognised for its self-supervised learning approach in training computer vision models, achieving significant results.

The model underscores the potency of self-supervised learning in advancing computer vision capabilities​​.

**Masked Autoencoders (**[**MAE**](https://arxiv.org/abs/2111.06377#:~:text=This%20paper%20shows%20that%20masked,based%20on%20two%20core%20designs)**):** MAE is a scalable self-supervised learning approach for computer vision that involves masking random patches of the input image and reconstructing the missing pixels.

Meta AI demonstrated the effectiveness of MAE pre-pre training for billion-scale pretraining, combining self-supervised (1st stage) and weakly-supervised learning (2nd stage) for improved performance​.

[**CLIP**](https://arxiv.org/abs/2103.00020) **(Contrastive Language-Image Pre-Training):** Developed by OpenAI, CLIP is a groundbreaking model that bridges computer vision and natural language processing, leveraging an abundantly available source of supervision: the text paired with images found across the internet.

CLIP is the first multimodal model tackling computer vision, trained on a variety of (image, text) pairs, achieving competitive zero-shot performance on a variety of image classification datasets. It brings many of the recent developments from the realm of natural language processing into the mainstream of computer vision, including unsupervised learning, transformers, and multimodality​

**Segment Anything Model (**[**SAM**](https://segment-anything.com/)**):** Developed by Meta’s FAIR lab, SAM is a state-of-the-art image segmentation model that aims to revolutionise the field of computer vision by identifying which pixels in an image belong to which object, producing detailed object masks from input prompts.

SAM is built on foundation models that have significantly impacted natural language processing (NLP), and focuses on promptable segmentation tasks, adapting to diverse downstream segmentation problems using prompt engineering​.

[**OneFormer**](https://arxiv.org/abs/2211.06220)**/** [**SegFormer**](https://arxiv.org/abs/2105.15203)**:** A state-of-the-art multi-task image segmentation framework implemented using transformers. Parameters: 219 million. Architecture: ViT

[**Florence**](https://www.microsoft.com/en-us/research/publication/florence-a-new-foundation-model-for-computer-vision/#:~:text=While%20existing%20vision%20foundation%20models,videos)**:** Introduced by Microsoft, this foundation model has set new benchmarks on several leaderboards such as TextCaps Challenge 2021, nocaps, Kinetics-400/Kinetics-600 action classification, and OK-VQA Leaderboard. Florence aims to expand representations from coarse (scene) to fine (object), and from static (images) to dynamic (videos)​.

[**Stable Diffusion**](https://arxiv.org/abs/2112.10752)**:** A generative model utilising AI and deep learning to generate images, functioning as a diffusion model with a sequential application of denoising autoencoders​.

It employs a U-Net model, specifically a Residual Neural Network (ResNet), originally developed for image segmentation in biomedicine, to denoise images and control the image generation process without retraining​​.

[**DALL-E**](https://openai.com/dall-e-3)**:** Developed by OpenAI, DALL-E is a generative model capable of creating images from textual descriptions, showcasing a unique blend of natural language understanding and image generation. It employs a version of the GPT-3 architecture to generate images, demonstrating the potential of transformer models in tasks beyond natural language processing​

The tech titans, often bundled as GAFA (Google, Amazon, Facebook and Apple), alongside several other companies such as Hugging Face, Anthropic, AI21 Labs, Cohere, Aleph Alpha, Open AI and Salesforce have been instrumental in developing, utilising and advancing foundation models. Substantial investments in these models underscore their potential, as these corporations harness foundation models to augment various facets of their operations, setting a benchmark for [other sectors](https://crfm.stanford.edu/2021/10/18/reflections.html#:~:text=Simultaneously%2C%20in%20industry%2C%20several%20startups,that%20impact%20billions%20of%20people).

Insights from industry leaders at [Google](https://venturebeat.com/ai/foundation-models-2022s-ai-paradigm-shift/#:~:text=Foundation%20models%20like%20DALL,computer%20science%20%20department%20at), [Microsoft](https://www.microsoft.com/en-us/research/academic-program/accelerate-foundation-models-research-fall-2023/#:~:text=About%20the%20program,society%20while%20mitigating%20risks) and [IBM](https://research.ibm.com/topics/foundation-models), alongside academic institutions, provide a rich tapestry of knowledge and perspectives​.

Percy Liang, a director of the Center for Research on Foundation Models, emphasised in [this article](https://www.protocol.com/enterprise/foundation-models-ai-standards-stanford) that foundation models like DALL-E and GPT-3 herald new creative opportunities and novel interaction mechanisms with systems, showcasing the innovation that these models can bring to the table. He also mentions potential risks of such powerful models​.

At the [ICCV 2023 conference](https://iccv2023.thecvf.com/), held this year in Paris, foundation models were a very present topic. William T. Freeman, Professor of Computer Science, MIT, talked about the foundation models in his talk in [QUO VADIS Computer Vision](https://gkioxari.github.io/Tutorials/iccv2023/) workshop. He cited reasons why he [does not like foundation models](https://drive.google.com/file/d/1HfSrxSMS54c6-rYQNKBZnqgk_eRYqwOx/view) as an academic:

1.  _They don’t tell us how vision works._
2.  _They’re not fundamental (and therefore not stable)_
3.  _They separate academia from industry_

This highlights the importance of foundation models for the future of computer vision and their established position and pragmatic aspect of those models focusing on performance.

IBM Research posits that [foundation models will significantly expedite AI adoption](https://research.ibm.com/topics/foundation-models) in business settings. The general applicability of these models, enabled through self-supervised learning and fine-tuning, allows for a wide range of AI applications, thereby accelerating AI deployment across various business domains​.

Microsoft Research highlights that foundation models are instigating [a fundamental shift in computing research](https://research.ibm.com/topics/foundation-models) and across various scientific domains. This shift is underpinned by the models’ ability to fuel industry-led advances in AI, thereby contributing to a vibrant and diverse research ecosystem that’s poised to unlock the promise of AI for societal benefit while addressing associated risks.

Experts also underscore the critical role of computer vision foundation models in solving real-world applications, emphasising their [adaptability to a myriad of downstream](https://crfm.stanford.edu/2021/10/18/reflections.html#:~:text=Simultaneously%2C%20in%20industry%2C%20several%20startups,that%20impact%20billions%20of%20people) tasks due to training on diverse, large-scale datasets​. Moreover, foundation models like CLIP [enable zero-shot learning](https://arxiv.org/abs/2103.00020), allowing for versatile applications like classifying video frames, identifying scene changes and building semantic image search engines without necessitating prior training.

In another workshop of ICCV 2023, [BigMAC](https://bigmac-vision.github.io/): Big Model Adaptation for Computer Vision, the [robustness of the CLIP model](https://bigmac-vision.github.io/pdfs/ludwig.pdf) on the popular ImageNet benchmark was discussed. In conclusion, thanks to training on a large, versatile dataset means that zero-shot predictions of the CLIP model are less vulnerable to data drift than popular CNN models trained and fine tuned on imageNet. In this [recording of Ludwig’s presentation](https://www.youtube.com/watch?v=XiouM3MEOKs&t=4546s) different ways to preserve CLIP robustness while fine-tuned are discussed.

On a side note, the ICCV conference was quite an event. With five days of workshops, talks and demos! Big tech companies such as Meta marked their presence with impressive hubs, answering attendees’ questions. Numerous poster sessions gave us a chance to interact with authors and select some ideas we would like to contribute to the tech stack at Adevinta.

In the subsequent sections, we will dig into real-world instances, underscoring their impact on e-commerce and elaborate how investing in this technology can galvanise collaboration and innovation across various teams within a company.

<span class="image fit">
![An overview of popular foundation models and their applications in various AI domains.](https://cdn-images-1.medium.com/max/800/0*ZtACVLp3aedfQYAy)
</span>

## Real-World Adoption of foundation models

Major tech companies have paved the way in producing and distributing ready-to-use foundation models, which are now being utilised by various businesses to [enhance or create new products](https://www.forbes.com/sites/moorinsights/2023/07/21/the-extraordinary-ubiquity-of-generative-ai-and-how-major-companies-are-using-it/) for tech-savvy consumers[​](https://www.forbes.com/sites/moorinsights/2023/07/21/the-extraordinary-ubiquity-of-generative-ai-and-how-major-companies-are-using-it/).

## E-commerce and retail

In the sphere of e-commerce, companies like Pinterest and eBay, have [invested in deep learning](https://developer.nvidia.com/blog/pinterest-uses-ai-to-enhance-its-recommendations-system/#:~:text=Developers%20from%20Pinterest%2C%20along%20with,objects%20saved%20has%20crossed) and machine learning technologies to enhance user experiences. Pinterest has developed PinSage for advertising and shopping recommendations and a multi-task deep metric [learning system for unified image embedding](https://blog.acolyer.org/2019/10/11/learning-a-unified-embedding-for-visual-search-at-pinterest/#:~:text=The%20foundation%20of%20Pinterest%E2%80%99s%20approach,task%20learning) to aid in [visual search](https://arxiv.org/abs/1908.01707) and recommendation systems​[​](https://blog.acolyer.org/2019/10/11/learning-a-unified-embedding-for-visual-search-at-pinterest/#:~:text=The%20foundation%20of%20Pinterest%E2%80%99s%20approach,task%20learning). eBay, on the other hand, utilises a convolutional neural network for its [image search feature](https://www.ebayinc.com/stories/news/an-easier-way-to-search-ebay-computer-vision-with-find-it-on-ebay-and-image-search-is-now-live/#:~:text=When%20you%20upload%20images%20to,the%20live%20listings%20on%20eBay), “Find It On eBay.”​

Computer vision applications are transforming e-commerce, aiding in creating seamless omnichannel shopping experiences​. When it comes to the importance of visuals in shopping experiences, a study by PowerReviews found that **88% of consumers specifically** [**look for visuals**](https://www.ebayinc.com/stories/news/an-easier-way-to-search-ebay-computer-vision-with-find-it-on-ebay-and-image-search-is-now-live/#:~:text=When%20you%20upload%20images%20to,the%20live%20listings%20on%20eBay) submitted by other consumers prior to making a purchase​.

### Broader tech industry

In the broader tech industry, Microsoft has introduced [Florence](https://www.ebayinc.com/stories/news/an-easier-way-to-search-ebay-computer-vision-with-find-it-on-ebay-and-image-search-is-now-live/#:~:text=When%20you%20upload%20images%20to,the%20live%20listings%20on%20eBay), a novel foundation model for computer vision. The underlying technology of foundation models is designed to provide a solid base that can be fine-tuned for various specific tasks, an advantage that has been recognised and harnessed by industry giants.

Take Copenhagen-based startup Modl.ai for instance, which relies on foundation models, self-supervised training and computer vision for [developing AI bots](https://the-decoder.com/ai-startup-wants-to-bring-foundation-models-to-game-development/#:~:text=Copenhagen,with%20and%20against%20human%20players) to test video games for bugs and performance. Such applications demonstrate the versatility and potential of foundation models in different sectors​.

The practical implementations of foundation models in these different sectors underscores their potential to drive innovation, enhance user experiences and foster cross-functional collaboration within and beyond the e-commerce spectrum. The flexibility and adaptability of foundation models, as demonstrated by these real-world examples, make them a valuable asset for companies striving to stay ahead in the competitive e-commerce landscape.

## Investing in foundation models: Cost-benefit analysis

The investment in foundation models for computer vision transcends the mere financial outlay. It encapsulates a strategic foresight to harness advanced AI technologies for bolstering e-commerce operations.

Investing in foundation models for computer vision in e-commerce does entail upfront costs such as acquiring computational resources and the requisite expertise. OpenAI’s GPT-3 model, for example, reportedly cost $4.6M to train. According to another OpenAI report, the cost of training a large AI model is [projected to rise](https://encord.com/blog/visual-foundation-models-vfms-explained/#:~:text=OpenAI%E2%80%99s%20GPT%2D3%20model%2C%20for%20example%2C%20reportedly%20cost%20%244.6MM%20to%20train.%20According%20to%20another%20OpenAI%20report%2C%20the%20cost%20of%20training%20a%20large%20AI%20model%20is%20projected%20to%20rise%20from%20%24100MM%20to%20%24500MM%20by%202030.) from $100M to $500M by 2030.

However, the potential benefits could justify the investment. For instance, the **global visual search market,** which is significantly powered by computer vision technology, is projected to reach **$15 billion by 2023**. Early adopters who incorporate visual search on their platforms could see [**revenues increase by 30%**](https://blog.taskmonk.ai/what-role-will-computer-vision-play-in-the-future-of-ecommerce/#:~:text=Early%20adopters%20who%20incorporate%20visual,of%20their%20online%20shopping%20experience). The computer vision market itself is soaring with an expected **annual growth rate of 19.5%**, predicted to reach a value of $100.4 billion by 2023​[​](https://encord.com/blog/visual-foundation-models-vfms-explained/#:~:text=April%2024%2C%202023%20%E2%80%A2%205,9Bn%20in%202022).

These figures suggest that the integration of computer vision, particularly through foundation models, can be a lucrative venture in the long-term. Consumers are increasingly leaning towards platforms that offer visual search and other AI-driven features. Therefore, the cost of investment could be offset by the subsequent increase in revenue, enhanced user engagement and improved operational efficiency brought about by the advanced capabilities of foundation models in computer vision.​

> **_“Foundation models cut down on data labelling requirements anywhere from a factor of like 10 times, 200 times, depending on the use case”_**— [Dakshi Agrawal](https://venturebeat.com/ai/foundation-models-2022s-ai-paradigm-shift/#:~:text=%E2%80%9CFoundation%20models%20cut%20down%20on%20data%20labeling%20requirements%20anywhere%20from%20a%20factor%20of%20like%2010%20times%2C%20200%20times%2C%20depending%20on%20the%20use%20case%2C%E2%80%9D%20Dakshi%20Agrawal%2C%20IBM%20fellow%20and%20CTO%20of%20IBM%20AI%2C), IBM fellow and CTO of IBM AI

Moreover, the global computer vision market, which encompasses technologies enabling such visual experiences, is expected to [grow substantially](https://www.syte.ai/blog/visual-ai/how-visual-ai-is-changing-omnichannel-retail/), indicating the increasing importance of investment in visual technologies for retail and e-commerce​. The role of visual AI, which includes [computer vision](https://research.aimultiple.com/computer-vision-retail/#:~:text=The%20global%20computer%20vision%20market,improve%20efficiency%20in%20omnichannel), is also highlighted in how it’s changing omnichannel retail, showcasing the intertwined relationship between visual technology and [enhanced shopping experiences](https://losspreventionmedia.com/computer-vision-future-of-retail/) across channels​.

## Examples of application of foundation models in e-commerce

Because of their pre-training on expansive datasets, foundation models in computer vision bring a treasure trove of capabilities to the table. **The pre-trained nature of foundation models significantly accelerates the deployment of computer vision applications in e-commerce, as they require less data and resources for fine-tuning compared to training models from scratch.** Let’s illustrate this through real-world examples within the e-commerce sector.

*   **Product Categorisation**: Leveraging a foundation model for automated product categorisation can be a time and resource-saver.
*   **Visual Search**: Implementing visual search features can be expedited with foundation models. Their pre-trained knowledge can be leveraged to recognise fashion or product trends, making visual search more intuitive.
*   **Counterfeit Detection**: Counterfeit detection is a complex task; however, with a foundation model, the pre-existing knowledge about different objects can be fine-tuned to identify subtle discrepancies between genuine and counterfeit products
*   **Moderation**: Detection of unwanted or harmful content can be done through a classification head added on top of image embeddings generated by a foundation model.


<span class="image fit">
![A diagram showcasing the various applications of foundation models in e-commerce, from product categorization to counterfeit detection.](https://cdn-images-1.medium.com/max/800/0*xDEvavmYRc4MF2xE)
</span>

Beyond these examples, foundation models also hold promise in enhancing user experiences in recommendation systems and augmented reality (AR) shopping.

Most of this use-case could be applied to Adevinta marketplaces or replace existing services based on more traditional models.

## Empowering teams across the e-commerce spectrum

Foundation models in computer vision open up avenues for fostering cross-functional collaboration, expediting product development, and making data-driven decision-making a norm across an e-commerce enterprise. Let’s delve into how these models can act as catalysts in harmonising the efforts of various teams and speeding up the journey from conception to market-ready solutions.

<span class="image fit">
![Members of the Cognition team at Adevinta discussing the latest trends in AI and computer vision.](https://cdn-images-1.medium.com/max/800/0*3hnJ3nPcTBr68D70)
</span>

## Accelerating the product development cycle

The pre-trained nature of foundation models significantly **cuts down the time traditionally required to develop, train and deploy machine learning models**. This acceleration in the product development cycle is invaluable in the fiercely competitive e-commerce market, where being the first to introduce innovative features can provide a substantial competitive edge. Moreover, the resource efficiency of foundation models ensures that **teams can iterate and improve upon models swiftly**, aligning with dynamic market trends and customer expectations.

## Stepping stone to broader business objectives

Foundation models can act as a springboard towards achieving broader business goals such as sustainability and promoting the second-hand goods trade. By enabling smarter product listings and verifications through image recognition and visual search capabilities, these models can streamline the process of listing and verifying second-hand goods. This, in turn, **promotes a circular economy, encouraging the reuse and recycling of products**, which aligns with the sustainability goals of many modern e-commerce platforms.

## Challenges and overcoming strategies

Incorporating foundation models for computer vision within an e-commerce setting comes with a range of challenges, but with the right strategies, these hurdles can be navigated to unlock the models’ full potential.

## Computational requirements

Foundation models are computationally intensive due to their large-scale nature, which necessitates [significant computational resources](https://snorkel.ai/foundation-models/#:~:text=Cost,for%20their%20end%20use%20caes) for training and fine-tuning. The good news is that, once the substantial work of domain-learning or fine tuning is done, numerous teams and projects can benefit from the foundation model with minimal additional effort and cost.

## Bias and fairness

Foundation models may inherit biases present in the training data, which can lead to unfair or discriminatory behaviour. For instance, [DALL-E and CLIP have shown biases](https://datagen.tech/blog/the-opportunities-and-risks-of-foundation-models/) regarding gender and race when generating images or interpreting text and images​. Implementing robust data preprocessing and bias mitigation strategies will help to address potential biases in training data.

## Interpretability and control

Understanding and controlling the behaviour of foundation models like CLIP remains a challenge due to their black-box nature. This makes it difficult to interpret the models’ predictions, which is a hurdle in applications where [explainability is crucial](https://arxiv.org/abs/2103.00020)​​. CRFM released recently a [Foundation Model Transparency Index](https://crfm.stanford.edu/fmti/?utm_campaign=The%20Batch&utm_medium=email&_hsmi=280825441&utm_content=280827829&utm_source=hs_email) “scoring 10 popular models on how well their makers disclosed details of their training, characteristics and use.”

Foundation models, if widely adopted, could introduce **single points of failure** in machine learning systems. If adversaries find vulnerabilities in a foundation model, they could [exploit these weaknesses](https://arxiv.org/abs/2103.11251) across multiple systems utilising the same model​.

Foundation models are **not the answer to all** machine learning problems.

> **_“Foundation models are neither ‘foundational’ nor the foundations of AI. We deliberately chose ‘foundation’ rather than ‘foundational,’ because we found that ‘foundational’ implied that these models provide fundamental principles in a way that ‘foundation’ does not. (…) Further, ‘foundation’ describes the (role of) model and not AI; we neither claim nor believe that foundation models alone are the foundation of AI, but instead note they are ‘only one component (though an increasingly important component) of an AI system.’”_**

> — the Stanford Institute for Human-Centred AI founded the Center for [Research on Foundation Models](https://arxiv.org/pdf/2108.07258.pdf) (CRFM), 2021

## Conclusion

The transformative potential of foundation models in computer vision is unmistakable and pivotal for advancing the e-commerce domain. They encapsulate a significant stride towards creating smarter, more intuitive and user-centric online shopping experiences. The notable successes of early adopters, alongside the burgeoning global visual search market, exhibit the financial promise inherent in embracing these models​.

The real-world implications extend beyond just improved product discovery and categorisation, to fostering a sustainable trading ecosystem for second-hand goods. **The expertise and investment in these models can expedite the product development cycle, encourage data-driven decision-making and stimulate cross-functional collaboration across various company departments.**

However, it’s crucial to acknowledge the technical and ethical challenges that come with the deployment of foundation models. The computational costs, potential biases and the necessity for robust infrastructures demand a well-thought-out strategic approach. Yet, with the right investment in computational infrastructure, continuous learning and a commitment to ethical AI practices, these hurdles can be navigated successfully.

While the promise of foundation models in computer vision is evident, discerning which model will perform optimally with your specific data remains a complex challenge.

This uncertainty underscores the vital **need for comprehensive benchmarks** that can guide businesses in selecting the most appropriate model. Investing time in testing and evaluation is crucial, as it enables a more informed decision-making process. A recent study highlighted in the article “[A Comprehensive Study on Backbone Architectures for Regular and Vision Transformers](https://arxiv.org/pdf/2310.19909.pdf)” delves into this subject by testing different model backbones across a range of downstream tasks and datasets. Such research is invaluable for businesses looking to capitalise on foundation models, as it provides critical insights into model performance and applicability, ensuring that their investment in AI is both strategic and effective.

In Adevinta, as an e-commerce leader, we are evaluating the pros and cons of foundation models to best leverage their potential within our company. In Cognition, we are also working on internal benchmarks that will help to chose right foundation model for the task, estimate ressources needed and showcase its potential performance on the marketplace data.

With industry behemoths and experts leading the era of foundation models, the call to action for e-commerce directors is clear: **Embrace the paradigm shift that foundation models represent, and consider them as a long-term strategic asset for maintaining a competitive edge in the rapidly evolving e-commerce landscape.**

Check out this mine of knowledge about foundation models: [https://cs.uwaterloo.ca/~wenhuche/teaching/cs886/](https://cs.uwaterloo.ca/~wenhuche/teaching/cs886/)

<footer>
  <p>Exported from <a href="https://medium.com">Medium</a> on December 19
    2023.</p>
  <p><a
      href="https://medium.com/adevinta-tech-blog/foundation-models-a-new-vision-for-e-commerce-76904a3066e8">View
      the original</a></p>
</footer>
<script type="text/javascript"
  src="//s7.addthis.com/js/300/addthis_widget.js#pubid=ra-584ec4ce89deed84"></script>
<div class="addthis_inline_share_toolbox"></div>

