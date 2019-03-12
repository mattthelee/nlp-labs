******************************************************
* SemEval-2016 Task 4: Sentiment Analysis on Twitter *
*                                                    *
*                ARABIC TRAINING DATA                *
*                                                    *
* http://alt.qcri.org/semeval2017/task4/             *
* semevaltweet@googlegroups.com                      *
*                                                    *
******************************************************

TRAINING dataset for SemEval-2016 Task 4, Arabic

Version 1.0: November 15, 2016


Task organizers:

* Sara Rosenthal, IBM Watson Health Research
* Noura Farra, Columbia University
* Preslav Nakov, Qatar Computing Research Institute, HBKU
* Fabrizio Sebastiani, Qatar Computing Research Institute, HBKU


LIST OF VERSIONS

  v1.0 [2015/11/15]: initial distribution of the data


SUMMARY

This archive includes the data for the Arabic versions of the following subtask in 2017:

A: Message Polarity Classification
B: Topic-Based Message Polarity Classification, two-point scale
C: Topic-Based Message Polarity Classification, five-pont scale
D: Topic-Based Tweet quantification, two-point scale
E: Topic-Based Tweet quantification, five-point scale

=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

Summary of the subtasks:

Subtask A: Message Polarity Classification.
Given a message, classify whether the message is of positive, negative, or neutral sentiment.

Subtasks B-C: Topic-Based Message Polarity Classification.
Given a message and a topic, classify the message on
B) two-point scale: positive or negative sentiment towards that topic
C) five-point scale: sentiment conveyed by that tweet towards the topic on a five-point scale.

Subtasks D-E: Tweet quantification.
Given a set of tweets about a given topic, estimate the distribution (i.e., %) of the tweets across
D) two-point scale: the "Positive" and "Negative" classes
E) five-point scale: the five classes of a five-point scale.

=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


NOTES

1. Please note that by downloading the Twitter data you agree to abide by the Twitter terms of service (https://twitter.com/tos), and in particular you agree not to redistribute the data and to delete tweets that are marked deleted in the future.

2. The distribution consists of a set of Twitter status IDs with annotations for Subtasks A, B, C, D, and E: topic polarity and trends toward a topic. You should use the downloading script to obtain the corresponding tweets: 
	
	https://github.com/seirasto/twitter_download

3. The "neutral" label in the annotations stands for objective_OR_neutral.

4. We do NOT provide a default split of the data into training, development and development-time testing datasets, and thus participants are free to use this data in any way they find useful when training and tuning their systems, e.g., use some train/dev/devtest split, perform cross-validation, train on all the data, etc.


FILES

README_Arabic.txt -- this file
SemEval2017-task4-train.subtask-A.arabic.txt -- training data for subtask A with gold annotations
SemEval2017-task4-train.subtask-BD.arabic.txt -- training data for subtasks B and D with gold annotations
SemEval2017-task4-train.subtask-CE.arabic.txt -- training data for subtasks C and E with gold annotations


DATA FORMAT FOR THE *GOLD* FILES


-----------------------SUBTASK A-----------------------------------------

The format for the training/dev/devtest *gold* file is as follows:

	id<TAB>label

where "label" can be 'positive', 'neutral' or 'negative'.


-----------------------SUBTASKS B,D--------------------------------------

The format for the training/dev/devtest *gold* file is as follows:

	id<TAB>topic<TAB>label

where "label" can be 'positive' or 'negative' (note: no 'neutral'!).

-----------------------SUBTASKS C,E--------------------------------------

The format for the training/dev/devtest *gold* file is as follows:

	id<TAB>topic<TAB>label

where "label" can be -2, -1, 0, 1, or 2,
corresponding to "strongly negative", "negative", "negative or neutral", "positive", and "strongly positive".


LICENSE

The accompanying datasets are released under a Creative Commons Attribution 3.0 Unported License (http://creativecommons.org/licenses/by/3.0/).


CONTACT:

E-mail: semevaltweet@googlegroups.com



REFERENCES:

Preslav Nakov, Sara Rosenthal, Svetlana Kiritchenko, Saif M. Mohammad, Zornitsa Kozareva, Alan Ritter, Veselin Stoyanov, Xiaodan Zhu. Developing a successful SemEval task in sentiment analysis of Twitter and other social media texts. Language Resources and Evaluation 50(1): 35-65 (2016).

Preslav Nakov, Alan Ritter, Sara Rosenthal, Fabrizio Sebastiani, and Veselin Stoyanov. SemEval-2016 Task 4: Sentiment Analysis in Twitter. In Proceedings of the 10th International Workshop on Semantic Evaluation (SemEval'2016), June 16-17, 2016, San Diego, California, USA.

Sara Rosenthal, Preslav Nakov, Svetlana Kiritchenko, Saif M Mohammad, Alan Ritter, and Veselin Stoyanov. SemEval-2015 Task 10: Sentiment Analysis in Twitter. In Proceedings of the 9th International Workshop on Semantic Evaluation (SemEval'2015), pp.451-463, June 4-5, 2016, Denver, Colorado, USA.

Sara Rosenthal, Preslav Nakov, Alan Ritter, Veselin Stoyanov. SemEval-2014 Task 9: Sentiment Analysis in Twitter. In Proceedings of International Workshop on Semantic Evaluation (SemEval’14), pp.73-80, August 23-24, 2014, Dublin, Ireland.

Preslav Nakov, Sara Rosenthal, Zornitsa Kozareva, Veselin Stoyanov, Alan Ritter, Theresa Wilson. SemEval-2013 Task 2: Sentiment Analysis in Twitter. In Proceedings of the Second Joint Conference on Lexical and Computational Semantics (*SEM'13), Volume 2: Proceedings of the Seventh International Workshop on Semantic Evaluation (SemEval'2013). pp. 312-320, June 17-19, 2013, Atlanta, Georgia, USA.
