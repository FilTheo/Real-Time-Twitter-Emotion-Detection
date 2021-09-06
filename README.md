# Real Time Twitter Emotion Detection


**This project was worked with Chris Norrie(b19chrno@his.student.se)**

It presents a real-time emotion classification framework resulting in an dashboard giving important analytics

The main idea is to build an emotion-detection framework which on real-time streams every tweet that has a selected hashtag (in our case #Covid19).
First the location of the user is extracted. Then tweets are classified as either happy, angry, sad, or scared 

The acquired information is loaded through elasticsearch into a Kibana's Dashboard offering various analytics.
An example of its usage is given below:

![Χωρίς τίτλο6](https://user-images.githubusercontent.com/70523417/91851315-9db6a200-ec67-11ea-89a9-986d2611f544.png)

The project is splitted into several steps: 

**Step1:   
