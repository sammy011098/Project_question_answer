# Project_question_answer

>> ![image](https://user-images.githubusercontent.com/114941577/230344219-2b327811-ed21-4c01-a3e3-8b8cfe217acb.png)
In this figure we can see that user input question is ”how pandemic had made an impact in hiring”. First algorithm convert user input and dataset question in vector form. Python library perform the task.After that cosine values are generated between the vector form of question entered by user and question present in dataset.This is repeated for all question present in dataset. We can see here cosine values of all the questions were generated. Maximum cosine value goes to the question ”covid affected in hiring software engineers” is 0.43751459214249794

>> ![image](https://user-images.githubusercontent.com/114941577/230344458-aa5f556c-a391-47fb-aa32-11a16119b6a7.png)
  In this figure we can see that user input question is ”daily chores of a data scientist”. First algorithm convert user input and dataset ques-tion in vector form. Python library perform the task.After that cosine values are generated between the vector form of question entered by user and question present in dataset.This is repeated for all question present in dataset. We can see here cosine values of all the questions were generated. Maximum cosine value goes to the question ”Inter-view preparation for data scientist job” is 0.7291445795460926.

But the output fetched in this result is the wrong output. We al-ready have a question in the dataset where the rounds of interviews are asked. Hence, the correct output is not fetched. But let us see how much difference it is actually making in the correct output. As we can see the second most cosine score 0.7206899685191225 which is next to the fetched output and this second most is the exact output
and the correct answer whose corresponding question is ”rounds in data science interview” and hence we can say word level embedding is not the best solution for this but if we can use Long Short Term Memory we can bring out the more accurate and precise results.

>> ![image](https://user-images.githubusercontent.com/114941577/230344641-7cef46bb-3b25-4163-a43d-79715b05b2ff.png)
In this figure we can see that user input question is ”how many rounds of interviews are conducted for a data scientist job”. First algorithm
convert user input and dataset question in vector form. Python li-brary perform the task.After that cosine values are generated between the vector form of question entered by user and question present in dataset.This is repeated for all question present in dataset. We can see here cosine values of all the questions were generated. Maximum cosine value goes to the question ”Interview preparation for a data scientist job ” is 0.43751459214249794.
