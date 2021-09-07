# Real Time Twitter Emotion Detection


**This project is worked with Chris Norrie(b19chrno@his.student.se)**

It presents a real-time emotion classification framework resulting in a dashboard providing important analytics regarding the public opinion (for specific topics)

The main idea is to build an emotion-detection framework that in real-time streams every tweet with a specific hashtag (in our case #Covid19).
First information such as the location of the user is extracted. Then tweets are classified as either happy, angry, sad, or scared 

The acquired information is loaded through elasticsearch into a Kibana's Dashboard visualizing the results
An example of its usage is given below:

![Χωρίς τίτλο6](https://user-images.githubusercontent.com/70523417/91851315-9db6a200-ec67-11ea-89a9-986d2611f544.png)

The project is split into several steps: 

**Step1:  Acquire a training data set**
A good classifier requires an adequate and balanced training sample.
In our case, we look for a balanced number of happy, sad, angry, and scared tweets.
Manually annotating thousands of tweets is not possible.

As a result, we focus on the hashtags to mine the desired tweets.
For example, a tweet ending in: #happy, #thrilled #happiness is probably a happy tweet.

Following this approach and a specific set of designed heuristics (to make sure the extracted tweets carry the desire emotion) we construct our training dataset.

**Step2: Pre-process Tweets**

Tweeter language is much different from academic articles or well-written reports. Different vocabulary, slang, and emojis are some of the most important issues.
We apply extensive pre-process of each tweet before training our learning algorithm
For details refer to either the report or the presentation

**Step3: Pick & Train a Machine Learning Model**

Next, we train several neural networks before making our selection.
Several variations of LSTMs are trained as classifiers and a performance investigation is performed before making our selection

**Step4: Design the Pipeline and the Framework**

Once training is complete we build our pipeline "listening" to tweets with a specific hashtag, preprocessing them and classifying them, before loading them to Kibana.

