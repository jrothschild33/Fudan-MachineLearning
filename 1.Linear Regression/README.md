## Homework  1： PM  2.5  prediction
#### Task
The datasets  are real observations downloaded from the website of the Central Meteorological Administration. Please use linear regression to predict the PM2.5 value.
#### Dataset
* train.csv：The  observations  of the first 20 days of each month  are  used  for  training.
* test.csv：The  observations  of 9 consecutive hours are taken from the remaining 10 days of each  month. All observations  for the first 8 hours are considered as features, and PM2.5 at the 9th hour is used as the answer. A total of 240 unique  samples  were taken  out  for  testing. Please predict the PM2.5 of these 240  samples according to features.
#### Submitted  files:
* Source  code  (python)
* submission.csv:  You  are  required  to  submit  your  results  according  to  the  format  of  sampleSubmission.csv.
* Report.pdf:  Please  briefly  describe  your  method  and  configurations  for  running  your  code  with  1-2  pages.
* Evaluation:  We  will  evaluate  your  results  with  the  RMSE  (root  mean  square  error)  metric.  You  can  sample  some  data  from  the  training.csv  for  testing  offline.
