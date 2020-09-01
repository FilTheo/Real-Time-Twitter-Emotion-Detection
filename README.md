# Real Time Twitter Emotion Detection


**This project was worked with Chris Norrie(b19chrno@his.student.se)**


This project has attempted to quantify public opinion about topics or events based on four discrete classes of emotions: happiness, anger, sadness, and fear. These classes, which are considered to be four of the most basic human emotions[3], were chosen based on prior research of Twitter data and how different emotions are represented and classified on social media[1]. 

Several LSTM models were trained to effectively classify the different tweets. Later, a streaming pipeline obtained newly arrived tweets, pre-processed and classified them and stored into an ElasticSearch Index. Finaly, results are visualized on a Kibana's Dashboard where analysts could get informed if the public feels anger or fear about a trending topic and make decisions accordingly. Different visualizations could inform analysts about the changes in people emotions during times, or how people feel about a topic on different parts of the world. 


## Hashatags
Twitter currently provides around five hundred million tweets per day on a wide variety of topics. Most of the time, these tweets are followed by a number of hashtags which include both the topic the user is writing about and some indication of their feelings on the topic. For example, "I still don’t know how @realDonaldTrump is the president... #Trump #anger" identifies both the topic (Trump, presumably the American president) and the emotion (anger) in the hashtags alone. Another example might be "going to the hospital for a #coronavirus checkup #nervous", which again, through the hashtags alone, identifies the topic (coronavirus, presumably COVID-19 at the time of writing) and the emotion (fear).

We have attempted to create a neural network trained on tweets such as the above examples that identify the emotion of the general public. This model can then be used to classify new tweets obtained via streaming and identify current public opinion (in terms of emotion) about new topics. 

## Method
Approximately 50000 tweets were queried through the Twitter API to create our main dataset. The decision to query our own dataset instead of using an existing dataset was made because Twitter restricts publicizing datasets queried by researchers and as such many existing datasets are simply not large enough for the purposes of this project - many datasets available online only contain 1000-1500 tweets. With that in mind, the steps taken from data collection to classifying streaming data are as follows.

#### Query Twitter and retrieve a dataset of tweets
The training dataset contains tweets and their class label (happiness, sadness, fear, or anger). One of the difficulties in acquiring such a dataset is automatically classifying a tweet as belonging to one of these emotional classes. Querying a large volume of tweets is fairly trivial but classifying them manually is not. The solution proposed was to only query tweets with specific hashtags that imply a class label. This methodology is supported by  studies performed by Saif Mohammed in 2012[2] which demonstrates that when Twitter users self-label their tweets as an emotion, that emotion is accurately reflected in the tweet.

For example, if a user uses a hashtag of “#happy”, it can be classified as having the label “happy” without the need to scrutinize the tweet further. This may seem obvious, but removes any ambiguity in that there is no need to analyze tweets wherein the emotional state cannot be immediately indicated through its hashtags.

 This methodology is further supported by another study performed in 2012[5] in which seven emotional states were specified and a dataset of 2.5 million tweets were each classified as indicating one of the specified emotional states. These seven emotional states are considered to be the ‘most important’ and distinct enough from the others[3], but for the sake of time constraints and model’s performance (multiclass classification problems with seven classes may be more ambiguous than with four classes), our study has pared it down to four of these states: joy, sadness, anger, and fear. 
 
Tweets were queried based not exclusively on the four words of class labels (i.e.: #joy, #sadness, #anger, #fear), but synonyms of the class labels as well (Appendix A). Synonyms were retrieved based on the most commonly used words for each class label, supported by studies[3][1] as to which words most strongly correlate to a specific human emotion. Once a list of commonly used hashtags for each emotion was defined, the tweets were queried using Tweepy and the Twitter developer API. Only tweets containing one of the hashtags on the list of synonyms were queried, which were then automatically classified as indicating a specific emotion corresponding to the synonym (such as ‘#scared’ or ‘#nervous’ both being automatically classified as ‘fear’).

#### Filtering the Dataset

After querying two weeks worth of tweets, the dataset was then filtered based on a set of heuristics proposed by Wang, Chen, and Thirunarayan[5]. The heuristics used are as follows.

1. Removing tweets with an excessive number (in this case, five or more) of hashtags as these might be bots or generally be less indicative of a specific emotion 
2. Removing tweets where the hashtags are not at the end of the tweet body as they are less likely to clearly indicate the writer's emotion[1]. 
3. Removing tweets with three or less words. These tweets generally do not indicate an emotional state of any kind.
4. Removing tweets containing non-English characters, as the trained model can only process English language tweets.
5. Removing retweets as they may have already been queried.
6. Removing replies as determining emotion depends on the context of the tweet being replied to. 

**To ensure the size of each emotion class is roughly the same and since most tweets were classified as happy, each other category was supplemented with other previously mined datasets used for competitions on similar tasks[21][22]**

The size of each class:

| **Class** | Anger | Fear | Happiness | Sadness |
------------|-------|------|-----------|---------|
| **Number of Tweets** | 11181 | 9536 | 14296 | 13802 |

#### Pre-Processing Tweets

Pre-processing the tweets before feeding them into the NN for training is an extremely important task as it greatly affects the performance of the model. 

The steps applied to prepare the tweets, were done using the ekphrasis library[11], a github dictionary[19], and emoji library[20].

* Replacing URL’s with “<url>” 
* Replacing usernames with “<username>”
* Spelling correction
* Stemming:
 * Converted contractions to their complete form (ie: “I’ve” to “I have”, “could’ve” to “could have”)
 * Replacing elongated words with their grammatically correct form (ie: “maaaaaad” to “mad”)
 * Removed hashtag symbol (ie: “#example” to “example”)
 * Segmented the words produced after a hashtag in the step before, by splitting them into actual words(ie: “#nojobinbiology” to “nojobsinbiology” to “no jobs in biology” )
 * Transforming emojis and emoticons into meaningful phrases (ie: “:)” to “smiley_face”). 

* Replacing slang words and commonly used internet abbreviations with their grammatically correct form (ie: “gr8” to “great”, “lol” to “laughing out loud”)
* Removing hashtags that indicated an emotion (on the list of synonyms in appendix A) so the model does not just learn the hashtags and is instead forced to learn based on the tweet content

 
 Some examples of pre-processed tweets and their respective class : 
 
![Χωρίς τίτλο](https://user-images.githubusercontent.com/70523417/91849741-30097680-ec65-11ea-8309-ad9dfd5bca37.png)

#### Canditate Models : 
The models were evaluated using accuracy and F1 as the evaluation metrics. Ideally, n-fold cross validation would have been applied but due to the excessive training time, this proved to be unrealistic

As word embeddings are necessary for LSTM models, GloVe was imported pre-trained for the word embeddings layer[12], reducing the training time of our models and increasing model accuracy. GloVe has previously been trained on millions of tweets, so it should provide a strong basis for word embeddings. Multiple approaches were considered, such as training embeddings based on our dataset, using a pre-trained embedding layer like GloVe[12], or training an embedding layer on a pre-existing Twitter dataset. We concluded that using GloVe was the best choice.

Despite initial promising results, the first model trained (single layer LSTM) overfitted on the training set after the 15th epoch and the accuracy on validation data was not as high as we expected. As a result, we added a second LSTM layer with fewer nodes and increased the dropout rate to prevent the model from overfitting.

Next, we constructed more complex architectures by training models with a bi-LSTM[13] layer and later with 2 bi-LSTM layers. Finally, we added an attention mechanism, similar to other LSTM models[11][15], on each of the trained models. In using the attention mechanism, we wanted to test our assumption that aggregating the results of every hidden state on the LSTM layer (based on their importance) would give us better results

**The Performance of the different models:**
![Χωρίς τίτλο1](https://user-images.githubusercontent.com/70523417/91850206-d2295e80-ec65-11ea-8ca5-b01769348627.png)
![Χωρίς τίτλο2](https://user-images.githubusercontent.com/70523417/91850260-e9684c00-ec65-11ea-8cc5-48f16ae265ce.png)

To further understand how each model performed and identify which classes our model failed to distinguish, we calculated the confusion matrices of each model and we observed how models classified tweets from every class.
![Χωρίς τίτλο3](https://user-images.githubusercontent.com/70523417/91850338-0e5cbf00-ec66-11ea-9b41-b6cc47913993.png)

As the dataset is relatively small, it is also important to check for over- or underfitting of models in order to find the perfect number of training epochs : 
**As the two-layers LSTMs produced better results we are only showing their results**
![Χωρίς τίτλο4](https://user-images.githubusercontent.com/70523417/91850618-80cd9f00-ec66-11ea-8be4-ccba6736f436.png)

#### Model Selection:
After considering the performance of each individual model, we decided to pick **2-layer bi-LSTM without attention** for streaming. 

To facilitate the model’s integration with the streaming pipeline, hyperparameters like number of nodes and dropout rate on each layer were optimized. In order to achieve that, we used k-fold cross validation to fine-tune one hyperparameter at the time by trying different values, while keeping the other parameters unchanged. For each hyperparameter, we picked the value that gave the best F1 score while also preventing the model from overfitting.

Additionally, both bi-LSTM layers have dropout and recurrent dropout set to 0.5, to prevent overfitting.

#### Streaming Pipeline
A streaming pipeline was created using Tweepy. A Tweets Stream Listener object, tracking tweets containing one or more hashtags, given by the user, was created. In addition, a preprocessing pipeline which included cleaning new tweets the same way we cleaned our training dataset was defined. Afterwards, they were fed into the model and their emotion was classified into one of the 4 classes. As a proof-of-concept exercise, geographic data and tweet's timestamp was also recorded for integration with a visualizer.

The architecture of the streaming pipeline:
![Χωρίς τίτλο5](https://user-images.githubusercontent.com/70523417/91851134-56c8ac80-ec67-11ea-972a-43e795cedbbd.png)

#### Results
A dashboard was created on Kibana, so we could further analyze the values we have generated from our streaming pipeline. The dashboard contains three different visualizations: 

* A map to compare emotions of different users based on their geolocation and find patterns between countries, 
* A line plot which analyzes how the total number of tweets regarding each emotion varies during time
* A pie chart to compare the total counts for each emotion. 

To demonstrate our results, we initialized our streaming pipeline and got it to track tweets containing the “#coronavirus” keyword. Streaming data was collected for 8 hours and the model classified over 100000 tweets

![Χωρίς τίτλο6](https://user-images.githubusercontent.com/70523417/91851315-9db6a200-ec67-11ea-89a9-986d2611f544.png)

**Important Note** Results are from end of March

An obvious first observation is that over half of the total tweets were classified as fear. Considering the obvious implication that the 2020 COVID-19 situation is fearful, this makes sense intuitively. Furthermore, a pattern around Europe seems to take place, most (but not all) tweets are either fear or sadness (which again makes sense intuitively. In addition, the regional percentage of happy tweets is very low. However, further analyzing distinct regions within the UK, a different pattern emerges.

People’s feelings (based on emotional state) vary within the UK, more so than the rest of Europe. The percentage of angry tweets is much higher than the rest of Europe. This could be a result of the public perception that measures taken to prevent the spread of COVID-19 were ineffective or late, or in general because of the public behaviour concerning the spread of the virus. 

Our Kibana dashboard allows us to focus on specific regions or specific ranges of time. For example, if an optimistic announcement was made, we create separate visualizations for tweets before and after the announcement, and observe if the percentage of happy tweets is increasing, which could imply that the announcement was effective. In general, many different queries could be made, and many patterns could be discovered by analysts - again, this work serves as a proof of concept, and there is much still to be explored.
 
 
#### Discussion

Comparing our model against others that have been previously researched demonstrated promising results. However, checking our confusion matrix led to some interesting conclusions that muddy the waters of this research a little further. Manually looking at the dataset and the automatically classified labels, led to the notion that some of these tweets may have been initially classified incorrectly.

These were mostly within the negative emotion categories. It was quite common for a tweet that appears intuitively angry to be classified as sadness. Additionally, many tweets could have belonged to more than one category (ie: a tweet can imply both anger and sadness). Our model only accounted for one of the four class labels for each tweet. Nevertheless, our model was able to correctly classify happy tweets with 87% accuracy. Some misclassifications might have been the result of sarcasm in tweets, users straying from the defined boundaries (ie: a sad tweet may have the hashtag of ‘#joy’, there could be many reasons for this), and tweets created through bots of advertisers not being filtered.

The results imply that the trained models can successfully classify ‘happy’ tweets. However, the models fail to consistently classify the ‘negative’ classes (‘anger’, ‘fear’, ‘sadness’). This might be explained by the implication that ‘happy’ is distinct from ‘anger’, ‘fear’, and ‘sadness’ due to happiness being quantified as a positive emotion and the other three are quantified as a negative emotion. Therefore, the negative classes have similarities with each other while happiness does not have similarities with any other class. 

In looking at how our model classified some of the ambiguously classified tweets, we were actually quite surprised - the model was often able to predict the emotional class that intuition might suggest a tweet belongs to, despite the tweet having been classified in the data querying step as something else. Below are a few examples of ambiguous classifications.
![Χωρίς τίτλο7](https://user-images.githubusercontent.com/70523417/91851849-92b04180-ec68-11ea-99a5-b109833c6a27.png)

In addition, as mentioned before, the training dataset was only 50000 tweets taken from two week’s time. Due to the time frame, it’s possible that the dataset may have been biased. For example, trending topics that week might have been more positive than they are normally, or people were more likely to use hashtags such as ‘anger’ more than previous weeks. Given the time constraints, this was impossible to investigate


## References
[1] Choudhury, Munmun D., Counts, Scott (2012) Not All Moods are Created Equal! Exploring Human Emotional States in Social Media. [WWW Document] Retrieved March 5, 2020, from https://www.researchgate.net/publication/255564129_Not_All_Moods_are_Created_Equal_Exploring_Human_Emotional_States_in_Social_Media

[2] Mohammed, Saif M. (2012) #Emotional Tweets. [WWW Document] Retrieved March 5, 2020, from https://www.aclweb.org/anthology/S12-1033.pdf

[3] Shaver, Philip (1987) Emotion Knowledge: Further Exploration of a Prototype Approach. Journal of Personality and Social Psychology

[4] Baziotis, Christos, Athanasiou, Nikos, Chronopoulou, Alexandra, Kolovou, Athanasia, Paraskevopoulos, Georgios, Ellinas, Nikolaos, Narayanan, Shrikanth, Potamianos, Alexandros (2018) NTUA-SLP at SemEval-2018 Task 1: Predicting Affective Content inTweets with Deep Attentive RNNs and Transfer Learning. Retrieved March 5, 2020, from https://www.aclweb.org/anthology/S18-1037.pdf

[5] Wang, W., Chen, L., Thirunarayan, K., Sheth, A., (2012). Harnessing Twitter “Big Data” for Automatic Emotion Identification. Retrieved March 6, 2020, from https://www.researchgate.net/publication/258762213_Harnessing_Twitter_'Big_Data'_for_Automatic_Emotion_Identification

[6] Mudinas, Andrius & Zhang, Dell & Levene, Mark. (2012). Combining lexicon and learning based approaches for concept-level sentiment analysis. 10.1145/2346676.2346681. 

[7] Elvis (2018). Deep Learning for NLP: An Overview of Recent Trends [WWW Document]. Medium. URL https://medium.com/dair-ai/deep-learning-for-nlp-an-overview-of-recent-trends-d0d8f40a776d (accessed 3.5.20).

[8] Mittal, A., (2019). Understanding RNN and LSTM [WWW Document]. Medium. Retrieved March 6, 2020, from https://towardsdatascience.com/understanding-rnn-and-lstm-f7cdf6dfc14e

[9] Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., Sutskever, I., (2019). Language Models are Unsupervised Multitask Learners.

[10] Felbo, B., Mislove, A., Søgaard, A., Rahwan, I., Lehmann, S., (2017). Using millions of emoji occurrences to learn any-domain representations for detecting sentiment, emotion and sarcasm. EMNLP. Retrieved March 6, 2020, from https://doi.org/10.18653/v1/D17-1169

[11] Baziotis, C., Pelekis, N., Doulkeridis, C., 2017. DataStories at SemEval-2017 Task 4: Deep LSTM with Attention for Message-level and Topic-based Sentiment Analysis, in: Proceedings of the 11th International Workshop on Semantic Evaluation (SemEval-2017). Presented at the SemEval 2017, Association for Computational Linguistics, Vancouver, Canada, pp. 747–754. https://doi.org/10.18653/v1/S17-2126

[12] Jeffrey Pennington, Richard Socher, Christopher D. Manning (2014) GloVe: Global Vectors for Word Representation. https://nlp.stanford.edu/projects/glove/

[13] Kingma, D.P., Ba, J., 2017. Adam: A Method for Stochastic Optimization. arXiv:1412.6980 [cs].

[14] Bi-LSTM - Raghav Aggarwal - Medium [WWW Document], n.d. URL https://medium.com/@raghavaggarwal0089/bi-lstm-bc3d68da8bd0 (accessed 3.12.20).

[15] Yang, Z., Yang, D., Dyer, C., He, X., Smola, A., Hovy, E., 2016. Hierarchical Attention Networks for Document Classification, in: Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies. Presented at the Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Association for Computational Linguistics, San Diego, California, pp. 1480–1489. https://doi.org/10.18653/v1/N16-1174

[16] Colneric, N., Demsar, J., 2019. Emotion Recognition on Twitter: Comparative Study and Training a Unison Model. IEEE Trans. Affective Comput. 1–1. https://doi.org/10.1109/TAFFC.2018.2807817

[17] Mohammad, S., Bravo-Marquez, F., Salameh, M., Kiritchenko, S., 2018. SemEval-2018 Task 1: Affect in Tweets, in: Proceedings of The 12th International Workshop on Semantic Evaluation. Presented at the Proceedings of The 12th International Workshop on Semantic Evaluation, Association for Computational Linguistics, New Orleans, Louisiana, pp. 1–17. https://doi.org/10.18653/v1/S18-1001

[18] Marc Brysbaert, Michaël Stevens, Paweł Mandera, Emmanuel Keuleers. How Many Words Do We Know? Practical Estimates of Vocabulary Size Dependent on Word Definition, the Degree of Language Input and the Participant’s Age. Frontiers in Psychology, 2016; 7 DOI: 10.3389/fpsyg.2016.01116

[19] Charles Malafosse, 2019, FastText-sentiment-analysis-for-tweets , GitHub repository, https://github.com/charlesmalafosse/FastText-sentiment-analysis-for-tweets/blob/master/betsentiment_sentiment_analysis_fasttext.py                

[20] Kyokomi, 2019, Emoji, GitHub repository , https://github.com/kyokomi/emoji

[21] Saif M. Mohammad, Felipe Bravo-Marquez, Mohammad Salameh, Svetlana Kiritchenko. Semeval-2018 Task 1: Affect in Tweets.. Proceedings of the International Workshop on Semantic Evaluation (SemEval-2018), New Orleans, LA, USA, June 2018.

[22] Saif M. Mohammad and Felipe Bravo-Marque,2017, WASSA-2017 Shared Task on Emotion Intensity. z. In Proceedings of the EMNLP Workshop on Computational Approaches to Subjectivity, Sentiment, and Social Media (WASSA), September 2017, Copenhagen, Denmark.

## Appendix A

List of emotion synonyms
![Χωρίς τίτλο8](https://user-images.githubusercontent.com/70523417/91852035-e3279f00-ec68-11ea-951e-0305f4057cb5.png)
