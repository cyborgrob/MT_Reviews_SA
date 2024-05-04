## 1. Project Overview

**Objective:** The goal of the project was to extract useful data from online reviews (such as those found on Google and Yelp) that could be applied to guide future business focus and adjustments. Specifically, the aim of the project was to identify various ‘aspects’ mentioned in reviews, such as ‘food’, ‘service’, ‘ambiance’ etc, and assign an associated ‘sentiment’ score - ‘positive’, ‘negative’, or ‘neutral’. With an aggregate of this information, business leaders could identify which specific aspects of their offering were driving sentiment, be it positive or negative. Additionally, in combination with date information, they could identify trends or shifts in sentiment over time or from a certain date (such as the introduction of a new product or initiative).

**Background:** What is aspect-based sentiment analysis (ABSA) and why is it useful?

ABSA is a subset of the more broad task of ‘sentiment analysis’, which attempts to assign an overall sentiment score to a piece of text. In the context of the hospitality business (or most businesses for that matter), assigning a general sentiment score to online reviews is trivial - on a 1-5 scale, one can simply score 4 or 5-star reviews as ‘positive’, 1 or 2-star reviews as ‘negative’, and 3-star reviews as ‘neutral’. The exact breakdown might vary based on individual preferences, but the idea is the same.

As a result, there is no use for a product that tells you what the overall sentiment of a review is, as one can simply glance at the star rating of a review to deduce it. What’s much more valuable is knowing *what specific aspects of your offering* generated the sentiment. Consider the following sample review:

> This place is really cool! I love the beer and the amazing pizza. The only downside is parking, there are never any spots! Still, overall a great place. 5 stars.
> 

Clearly, the reviewer expressed positive sentiment toward the aspects ‘beer’ and ‘pizza’ and negative sentiment toward ‘parking’. But what one would find upon doing a basic analysis such as keyword extraction and rules-based application is that the system would correctly extract the keywords ‘beer’, ‘pizza’, and ‘parking’, but then incorrectly apply a ‘positive’ sentiment to all of them simply because, in the end, the reviewer gave 5 stars. It would incorrectly assume that because the overall sentiment of the review was positive, all the extracted aspects would also be positive.

The need for a more granular and discriminatory tool is clear - enter ABSA. ABSA ML models are trained to not only identify aspects, but also to associate a sentiment with each specific aspect. Using the review above, a properly-trained ABSA model that performs an accurate inference would correctly be able to identify the differing sentiments, and would output something like this:

```jsx
{'beer': 'positive',
 'pizza': 'positive',
 'parking': 'negative'}
```

One can probably see the usefulness of this more granular analysis - a business leader would be able to clearly see what specific aspects of their product or service are driving sentiment, rather than the less-desirable options of having to guess, or having to manually parse through reviews themselves.

## 2. Data

**Data Source:** Online reviews of the business - namely Google, Yelp, and TripAdvisor (the majority were from Google, however). A 3rd-party tool called Outscraper was used to get reviews off of Google; the Yelp and TripAdvisor reviews were scraped programattically (https://github.com/cyborgrob/MT_Reviews_SA/blob/master/main.py)

**Data Description:** The initial dataset included 425 reviews scraped from Google, Yelp, and TripAdvisor. Each sample included the review text, the star rating assigned, and the date the review was posted. Eventually, the 425 text features were split into 844 samples of only text, along with the annotated aspects and associated sentiments, for the purpose of fine-tuning the ABSA model.

**Data Preparation:** The review text was initially cleaned, lemmatized, stop words removed, etc for the purpose of EDA. This process is documented here: https://github.com/cyborgrob/MT_Reviews_SA/blob/master/consolidate_reviews.ipynb. Ultimately, this rules-based approach was abandoned in favor of using a ML model specifically trained for the ABSA function (see the **Challenges and Learnings** section for more info).

Once the goal became to fine-tune and use an ABSA model, the challenge was to get the data into a format that the model expected. First, each review’s text portion was split into individual sentences (more or less - it split by ‘.’, so there were some edge cases where the reviewer either forgot to use punctuation, or they used different punctuation - ‘?’ or ‘!’, for instance). This is how the data set went from 425 original samples to 844. Next, the dataset was split into train/validation/test sets of sizes 590/127/127 respectively.

Finally, the data was annotated using a custom-built annotation tool (https://github.com/cyborgrob/MT_Reviews_SA/blob/master/annotation_tool.py) to morph it into the format expected by the model.

## 3. Methodology

**Model Selection:** There are various ABSA-specific models available on the internet; they are officially judged based on the subtasks Aspect Term Extraction (ATE), Aspect Term Sentiment Classification (ATSC), and Aspect-Sentiment Pair Extraction(ASPE - the one that best suits our purposes). The official dataset used to train and test are the SemEval2014 (see https://paperswithcode.com/dataset/sts-2014), which is itself sub-divided into restaurant and laptop reviews.

Initially, I chose a framework called PyABSA (https://github.com/yangheng95/PyABSA/tree/v2). The reason I chose it is because it was the first result I came across during my search, and it looked suitable. While it was able to produce some results, ultimately I abandoned this model/repo due to a litany of technical issues and some limitations on processing longer strings (see **Challenges and Learnings** section for more info).

The resulting search led me to the InstructABSA model (https://github.com/kevinscaria/InstructABSA). Not only was this model easier to use out of the box, but its codebase was much more readable, and, to top it off, it was the current SOTA on the ABSA-specific subtasks (see https://arxiv.org/abs/2302.08624).

**Model Building:** Our aim is to fine-tune the InstructABSA base model. The InstructABSA model is a t5 encoder-decoder model (https://huggingface.co/kevinscaria/joint_tk-instruct-base-def-pos-neg-neut-combined) which is itself fine-tuned on the Tk-Instruct series of models ([allenai/tk-instruct-base-def-pos](https://huggingface.co/allenai/tk-instruct-base-def-pos)), which are themselves built upon t5 models, and so on.

Here is the general configuration for the base InstructABSA model:

```jsx
{
  "_name_or_path": "allenai/tk-instruct-base-def-pos",
  "architectures": [
    "T5ForConditionalGeneration"
  ],
  "d_ff": 2048,
  "d_kv": 64,
  "d_model": 768,
  "decoder_start_token_id": 0,
  "dense_act_fn": "gelu_new",
  "dropout_rate": 0.1,
  "eos_token_id": 1,
  "feed_forward_proj": "gated-gelu",
  "initializer_factor": 1.0,
  "is_encoder_decoder": true,
  "is_gated_act": true,
  "layer_norm_epsilon": 1e-06,
  "model_type": "t5",
  "num_decoder_layers": 12,
  "num_heads": 12,
  "num_layers": 12,
  "output_past": true,
  "pad_token_id": 0,
  "relative_attention_max_distance": 128,
  "relative_attention_num_buckets": 32,
  "tie_word_embeddings": false,
  "torch_dtype": "float32",
  "transformers_version": "4.26.0",
  "use_cache": true,
  "vocab_size": 32100
}
```

And here are the training arguments used during training:

```jsx
training_args = {
    'output_dir':model_out_path,
    'evaluation_strategy':"epoch",
    'learning_rate':5e-5,
    'lr_scheduler_type':'cosine',
    'per_device_train_batch_size':8,
    'per_device_eval_batch_size':16,
    'num_train_epochs':4,
    'weight_decay':0.01,
    'warmup_ratio':0.1,
    'save_strategy':'no',
    'load_best_model_at_end':False,
    'push_to_hub':False,
    'eval_accumulation_steps':1,
    'predict_with_generate':True,
    'use_mps_device':use_mps,
}
```

The procedure to fine-tune the model was built into its source code; essentially, all you have to do is plug in your own custom dataset (via filepath) into the appropriate slot in the Jupyter Notebook and slightly modify the code.

**Tools and Technologies:** The model uses various tools from the PyTorch library as well as several from the HuggingFace Transformers library (see https://github.com/kevinscaria/InstructABSA/blob/main/InstructABSA/utils.py). Fine-tuning the model and modifying the base code required digging into these libraries at various times to figure out how the tools actually worked under the hood. Jupyter Notebooks was the application used to run the code.

## 4. Results, Evaluation, and Deployment

**Model Performance:** Here are the base model’s performance metrics before fine-tuning on the custom dataset (evaluation performed on the test set of 127 samples - no cross validation):

```jsx
Train Precision:  0.7053320860617399
Train Recall:  0.7717502558853634
Train F1:  0.7370478983382209
Test Precision:  0.680161943319838
Test Recall:  0.7962085308056872
Test F1:  0.7336244541484715
```

And here is the fine-tuned model’s performance on the same test set (again, no cross validation - this is something that could be implemented in the future for more accurate evaluation):

```jsx
Train Precision:  0.888125613346418
Train Recall:  0.9263050153531218
Train F1:  0.906813627254509
Test Precision:  0.7851063829787234
Test Recall:  0.8661971830985915
Test F1:  0.8236607142857143
```

Here are some graphs to better visualize the differences (the bar graph has only test results):

![1.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/b4999980-fc1b-429e-b94d-ef1b66852e22/7b4d548f-1d90-4fcd-a664-529f96f3a944/1.png)

![1.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/b4999980-fc1b-429e-b94d-ef1b66852e22/eeed368c-35fc-4eab-8f57-be29e7158b8a/1.png)

**Visualizations: NOTE - these visualizations were originally created in the EDA step referenced in 2. Data - Data Preparation, using various rules-based approaches, as well the data from the PyABSA model. As noted, those tracks were abandoned in favor using the InstructABSA model. However, the visualizations are still useful as a indication of the types of analysis that could be done. Work on this project was halted after fine-tuning the InstructABSA model, but it would be a relatively simple task to use the model to run inference on the data and generate results that could then be visualized using similar graphs/charts as to what follows.**

![1.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/b4999980-fc1b-429e-b94d-ef1b66852e22/e59e134a-a163-4d36-874c-de9e89a9280b/1.png)

![1.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/b4999980-fc1b-429e-b94d-ef1b66852e22/0b414b45-a633-4ba4-9a17-913c9121cffb/1.png)

![1.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/b4999980-fc1b-429e-b94d-ef1b66852e22/64af0153-8fc4-4f6d-98d7-61ec661db0f3/1.png)

![1.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/b4999980-fc1b-429e-b94d-ef1b66852e22/c8ea4f71-00d5-401b-8ad3-3460cfe035f0/1.png)

And so on…

**Interpretation:** Currently N/A because the fine-tuned model hasn’t yet been used for inference on the dataset.

**Deployment:** I first thought of using Sagemaker for model deployment, but after sifting through the docs and some initial experimentation, ultimately decided on deploying a simple demo on HuggingFace Spaces using the Gradio library. The link is here: 

[InstructABSA Ft - a Hugging Face Space by Homeskills](https://huggingface.co/spaces/Homeskills/Instruct-ABSA-ft)

And here’s a screenshot of the demo in use on a simple review sentence:

![1.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/b4999980-fc1b-429e-b94d-ef1b66852e22/f3936ced-312d-423c-9ce9-b5bedc39079b/1.png)

## 5. Challenges and Learnings

**Challenges:** As noted, there were several challenges encountered throughout the project, some relatively minor and others major roadblocks that required days to work through. Here we’ll discuss 3 of the major challenges faced.

**Approach:** First was finding the right model/approach to this problem of ABSA. Having never done a project like this before, my only guides were Google and Chatgpt. It should be noted that at the beginning, I hadn’t even settled on ABSA as my focus - I was going into it with a general ‘sentiment analysis’ approach. But I quickly learned that this general approach would be ineffective (as discussed in **1. Project Overview - Background**). I needed something more substantial, more granular, that would provide actual, useful business insight.

That led me to attempt to build a rules-based formula wherein a predefined dictionary of aspects and a lexicon of ‘positive’ or ‘negative’ terms would be applied to a piece of review text. The code would see that an aspect was present, extract it, see various sentiment words, extract those, and then deduce whether the aspect had positive or negative sentiment attached based on the frequency of words in the relative lexicon. If you’re following, you can probably see how this fails.

For starters, it requires you to manually add aspects and positive/negative words to the respective lexicons, which is time-consuming, incomplete, and error-prone. Second, it doesn’t account for anything like misspellings, implication, or reviewer intent (Eg, if a reviewer misspells “beer” as “berr”, this system wouldn’t catch it - you’d have to add in all potential misspellings of “beer”, or else be content with missing the aspect altogether). Third, it’s ultimately ineffective anyway. As discussed earlier, a review could have positive sentiment overall but still identify particular aspects that weren’t up to par; this approach would miss that. Also, there’s the challenge of ambiguous words and not understanding context. Consider the following two reviews: “I love it” and “I didn’t love it”. The word “love” would be in either the ‘positive’ lexicon or the ‘negative’ one - it can’t be in both. Assuming “love” is ‘positive’, the second review would score a point for ‘positive’ sentiment, which is clearly wrong. One can look down the rabbit hole of writing various rules to cover all cases and see how this could quickly become unwieldy. I decided a deep learning solution was necessary to adequately solve this problem.

**PyABSA:** I found what appeared to be my solution with the PyABSA repository/model. I actually got it working OK, but there were a myriad of technical issues. For one, the most recent version doesn’t work with Google Colab, so to use Colab you’ll need to rollback to a previous version of PyABSA. A second issue involved the model not outputting results once the review got over a certain length - a feature of BERT models is that the max token length is 512 tokens. To solve this, I broke each review up in to smaller pieces, which I was initially hesitant about, thinking it might compromise the integrity of the data, but in hindsight I now believe is mostly fine.

But the biggest challenge with PyABSA came when trying to train it on my own custom dataset. After formatting my data in the format required by PyABSA (which was its own challenge in itself), the question became how to actually make it available to the model? The documentation was rather vague, but there seemed to be two options - either move it into the Datasets “folder”, or upload your custom data to the Datasets repository via a pull request.

The issue with moving it into the Datasets folder was that the folder didn’t exist… yet. After cloning the repo, the folder referenced isn’t in the group of files. For some reason, it only becomes available after running a certain script in one of the notebooks. This isn’t explained in the instructions, or anywhere that I could find. And the folder only showed up after I unwittingly ran the script in question and randomly noticed that there was a new folder available in the directory. This took several days of frustration, trial and error, and digging through the repository open issues, trying various things.

The second option was to upload my custom dataset to the Datasets repository which, in hindsight, I could have at least attempted. But since ABSA Datasets was its own separate repository (https://github.com/yangheng95/ABSADatasets), it wasn’t clear to me how uploading my data to a different repository was supposed to make it available to the model (likely just clone that repo to get access to the dataset - but that still would have run into the problem of the Datasets folder not existing).

Either way, now that I had my custom dataset available to the model, I tried to run some training to fine-tune it and… another error. At this point I decided to go looking for alternative repositories, and that’s what led me to InstructABSA (it should be noted that there were many other technical issues using this repo, these are just the major ones worth noting). NOTE: I was actually able to use PyABSA on my data to extract keywords and assign sentiments. The graphs above in **4. Results - Visualizations** were created using the PyABSA results. The Dataset issue came up when I tried to fine-tune the model to my custom data to improve performance, which I had decided I wanted to do.

**AWS Sagemaker:** After fine-tuning the InstructABSA model and running inference, I wanted to deploy it so that a user could make a request with a sample review, and the API would return the result in the expected format.

I first considered AWS Lambda, having had some exposure to that service in the past. However, the model files were too large to directly load it onto Lambda (which has a 250MB limit), so I tried to use to AWS Sagemaker instead. The main challenge that presented itself was that the boilerplate code to deploy a model from HuggingFace onto Sagemaker didn’t work for this particular model. I could open up a Sagemaker notebook and have the model url point to my model, and even learned I needed to adjust the task type. But even after doing all that, the result from a sample inference was just gibberish. Sagemaker seemed to think my model was some sort of text generation model instead of a classification model, and no tuning of the ‘task type’ setting was able to change it.

After a couple days of exploring Sagemaker, I decided it wasn’t worth the effort in this case to deal with all the technical challenges, and opted to deploy my model using HuggingFace Spaces as shown above. This provides the same functionality I was hoping to get with Sagemaker, but in a simpler, nicer-looking interface.

**Learnings:** I learned a TON about the ABSA task, and NLP in general, from this project. I learned that a rules-based approach, for anything but the most trivial of tasks, is going to quickly run into problems, and that a deep learning solution is likely the way to go. I learned that the ABSA task in its current state is a very niche problem, and even SOTA models/repositories aren’t actively maintained or used. I learned that training even a SOTA model on a custom dataset can result in big performance boosts when testing on the custom dataset. I learned to try different pathways to get to my goal, and to not get set on one process. If something isn’t working, or working as well as you want it to, try something else. 

Due to the time spent digging into Github repos, I learned a lot about how repos work, about open/closed issues, about how cloning/forking works, and about the importance of good documentation. I learned to not “put the repo on a pedestal” as my mentor said, referring to my thinking that the PyABSA Datasets repo was some sort of holy sanctuary that I couldn’t even think about submitting a pull request on.

I learned how the PyTorch and HuggingFace libraries interact. I learned that SOTA models are often just fine-tuned versions of models that are fine-tuned on other models and on down the line. I learned what the “model files” actually contain. And many other things.

## 6. Conclusion and Future Work

**Summary:** A SOTA ABSA model can be trained on a custom dataset and provide reasonable (this is a relative term) accuracy classifying aspects and associated sentiment such that it could be useful in business contexts.

In this specific case, the fine-tuned model could be fed review text, past and future, and extract aspects and assign sentiments with a good enough degree of accuracy that its results could be used to generate BI analytical graphs and tables, which could be useful to business leaders when making decisions. The same process could be done manually as well - it would take far more time, but would be nearly 100% accurate.

The reality-based business applications of a model like this for most enterprises is questionable - the majority of businesses don’t have the quantity of reviews necessary either to provide adequate training data or to justify the use of machine learning techniques. The brewery whose reviews were used in this project is one of the largest breweries/hospitality endeavors in its region, and still, the entirety of the review corpus for the relevant time period numbered only 425, which was small enough that I was able to annotate the entire thing myself over a little more than a day. For smaller enterprises, where reviews trickle in one by one over the period of days or weeks, it would be entirely unnecessary to have a machine learning solution classify the sentiment. That said, for quick, exploratory data science and business intelligence use cases, this ABSA machine learning solution might be useful.

Its real use case would likely be for a popular product selling on a popular site like Amazon. A product with tens of thousands of reviews might be a good candidate for a tool like this; the number of reviews makes it impractical to sort through them manually, and the fine-tuned accuracy could be trained to be good enough that the particular aspects driving sentiment could be clearly identified across thousands of reviews.

**Future Work:** I’ve already noted that I did not use the fine-tuned InstructABSA model to actually perform inference on the dataset - that’s something that could be done to generate more accurate data that could then be used to create relevant business insights.

Additionally, both the base InstructABSA and the fine-tuned version were only tested (and in the case of the fine-tuned version, trained) on one split of the data. A more robust evaluation could be done by utilizing cross-validation and using alternate subsets of data.

I also intend to create a YT video that explores some of the uses for ABSA, specifically for the InstructABSA model, how to manipulate the data it generates, etc. Basically, create some sort of tutorial or guide that goes over a lot of the challenges I faced when building this project so that the lessons I learned might be of some help to others who are attempting something similar. The YT/internet space is empty in regards to this specific model/use case beyond anything other than simple demos.
