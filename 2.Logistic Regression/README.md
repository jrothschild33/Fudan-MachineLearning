## Homework  2： Classification
#### Task
Please  implement  two  models  (probabilistic  generative  model  &&  logistic  regression  model)  to  predict  whether  a  person  can  make  over  50k  a  year  according  to  the  personal  information.

#### Dataset
train.csv  and  test.csv  have   32561 and  16281  samples  respectively.

The  attributions  are:  age, workclass, fnlwgt, education, education num, marital-status, occupation  relationship, race, sex, capital-gain, capital-loss, hours-per-week,  native-country, make over 50K a year or not.

#### Attribute  descriptions  
* The  symbol  "?”  denotes  “unsure”
* age: continuous.
* workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
* fnlwgt: continuous.  The number of people the census takers believe that observation represents.
* education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
* education-num: continuous.
* marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
* occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
* relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
* race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
* sex: Female, Male.
* capital-gain: continuous.
* capital-loss: continuous.
* hours-per-week: continuous.
* native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.

#### Submitted  files
* Source  code  (python):  It  should  include  at  least  two  programming  files,  namely  generative.py  and  logistic.py.  Please  implement  the  two  models  by  yourself.
* submission.csv:  You  are  required  to  submit  your  results  according  to  the  format  of  sampleSubmission.csv.  The  first  row  is  (id,  label),  where  id  represents  the  sample  order  in  the  test  set.  (label  =0  means  “<= 50K” 、 label = 1 means  “ >50K ”)
* Report.pdf:  Please  describe  the  configurations  for  running  your  code,  and  answer  the  following  questions  in  your  report  with  no  more  than  2  pages:
1. Please compare the accuracy of your implemented generative model and logistic regression. Which one is better?
2. Please implement input feature normalization and discuss its impact on the accuracy of your model.
3. Please implement regularization of logistic regression and discuss its impact on the accuracy of your model. (For regularization, please refer to  “Lecture  1.1-Regression”)
4. Please discuss which attribute you think has the most impact on the results.

