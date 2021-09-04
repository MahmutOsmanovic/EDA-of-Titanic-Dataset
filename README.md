### Exploraty Data Analysis of the famous titanic dataset.

## General Competition Description
>The goal in this competition is to take an image of a handwritten single digit, and determine what >that digit is.
>
>The data for this competition were taken from the MNIST dataset. The MNIST ("Modified National >Institute of Standards and Technology") dataset is a classic within the Machine Learning community >that has been extensively studied. 

## Contents
# Part 1: EDA
- 1. Feature analysis
- 2. Scanning for general relationships and trends on multiple features

# Part 2: Cleaning the Data && Feature Engineering
- 1. Adding new features
- 2. Removing redundant features
- 3. Converting features into suitable form for modeling.

# Part 3: Predictive Modeling
- 1. Running Basic Algorithms
- 2. Cross Validation
- 3. Ensembling (combine simpler models to create a new single more powerful one)
- 4. Important Features Extraction

**How many Survived?**

<img src="images/Figure%202021-09-04%20111407%20(0).png" width="1024">

It is evident that not many passengers survived the accident.

Out of 891 passengers in training set, only around 350 survived i.e Only 38.4% of the total training set survived the crash. I need to dig down more to get better insights from the data and see which categories of the passengers did survive and who didn't.

I will try to check the survival rate by using the different features of the dataset. Some of the features being Sex, Port Of Embarcation, Age,etc.

First let us understand the different types of features.

**Analysing The Features**

<img src="images/Figure%202021-09-04%20111407%20(1).png" width="1024">

This looks interesting. The number of men on the ship is lot more than the number of women. Still the number of women saved is almost twice the number of males saved. *The survival rates for a women on the ship is around 75% while that for men in around 18-19%.*

This looks to be a very important feature for modeling. But is it the best?? Lets check other features.

<img src="images/Figure%202021-09-04%20111407%20(2).png" width="1024">

People say Money Can't Buy Everything. But I can clearly see that Passenegers Of Pclass 1 were given a very high priority while rescue. Even though the the number of Passengers in Pclass 3 were a lot higher, still the number of survival from them is very low, somewhere around 25%.

For Pclass 1 %survived is around 63% while for Pclass2 is around 48%. So money and status matters. Such a materialistic world.

Lets Dive in little bit more and check for other interesting observations. Lets check survival rate with Sex and Pclass Together.

<img src="images/Figure%202021-09-04%20111407%20(3).png" width="1024">

I use FactorPlot in this case, because they make the seperation of categorical values easy.

Looking at the *CrossTab* and the *FactorPlot*, I can easily infer that survival for Women from Pclass1 is about *95-96%*, as only 3 out of 94 Women from Pclass1 died.

It is evident that irrespective of Pclass, Women were given first priority while rescue. Even Men from Pclass1 have a very low survival rate.

Looks like Pclass is also an important feature. Lets analyse other features.

<img src="images/Figure%202021-09-04%20111407%20(4).png" width="1024">

**Observations:**
- 1) The number of children increases with Pclass and the survival rate for passenegers below Age 10(i.e children) looks to be good irrespective of the Pclass.

- 2) Survival chances for Passenegers aged 20-50 from Pclass1 is high and is even better for Women.

- 3) For males, the survival chances decreases with an increase in age.

As we had seen earlier, the Age feature has *177 null values*. To replace these NaN values, we can assign them the mean age of the dataset.

But the problem is, there were many people with many different ages. We just cant assign a 4 year kid with the mean age that is 29 years. Is there any way to find out what age-band does the passenger lie??

Bingo!!!!, we can check the Name feature. Looking upon the feature, we can see that the names have a salutation like Mr or Mrs. Thus we can assign the mean values of Mr and Mrs to the respective groups.

<img src="images/Figure%202021-09-04%20111407%20(5).png" width="1024">

**Observations:**
- 1) The Toddlers(age<5) were saved in large numbers(The Women and Child First Policy).

- 2) The oldest Passenger was saved(80 years).

- 3) Maximum number of deaths were in the age group of 30-40.

<img src="images/Figure%202021-09-04%20111407%20(6).png" width="1024">

The Women and Child first policy thus holds true irrespective of the class.

<img src="images/Figure%202021-09-04%20111407%20(7).png" width="512">

The chances for survival for Port C is highest around 0.55 while it is lowest for S.

<img src="images/Figure%202021-09-04%20111407%20(8).png" width="1024">

**Observations:**
- 1) Maximum passenegers boarded from S. Majority of them being from Pclass3.

- 2) The Passengers from C look to be lucky as a good proportion of them survived. The reason for this maybe the rescue of all the Pclass1 and Pclass2 Passengers.

- 3) The Embark S looks to the port from where majority of the rich people boarded. Still the chances for survival is low here, that is because many passengers from Pclass3 around 81% didn't survive.

- 4) Port Q had almost 95% of the passengers were from Pclass3.

<img src="images/Figure%202021-09-04%20111407%20(9).png" width="768">

**Observations:**
- 1) The survival chances are almost 1 for women for Pclass1 and Pclass2 irrespective of the Pclass.

- 2) Port S looks to be very unlucky for Pclass3 Passenegers as the survival rate for both men and women is very low.(Money Matters)

- 3) Port Q looks looks to be unlukiest for Men, as almost all were from Pclass 3.

<img src="images/Figure%202021-09-04%20111407%20(10).png" width="1024">

**Observations:**
The barplot and factorplot shows that if a passenger is alone onboard with no siblings, he have 34.5% survival rate. The graph roughly decreases if the number of siblings increase. This makes sense. That is, if I have a family on board, I will try to save them instead of saving myself first. Surprisingly the survival for families with 5-8 members is *0%*. The reason may be Pclass??

The reason is Pclass. The crosstab shows that Person with SibSp>3 were all in Pclass3. It is imminent that all the large families in Pclass3(>3) died.

<img src="images/Figure%202021-09-04%20111407%20(11).png" width="1024">

**Observations:**
Here too the results are quite similar. Passengers with their parents onboard have greater chance of survival. It however reduces as the number goes up.

The chances of survival is good for somebody who has 1-3 parents on the ship. Being alone also proves to be fatal and the chances for survival decreases when somebody has >4 parents on the ship.

<img src="images/Figure%202021-09-04%20111407%20(12).png" width="1024">

# Observations in a Nutshell for all features:
**Sex:** The chance of survival for women is high as compared to men.

**Pclass:** There is a visible trend that being a 1st class passenger gives you better chances of survival. The survival rate for Pclass3 is very low. For women, the chance of survival from Pclass1 is almost 1 and is high too for those from Pclass2. Money Wins!!!.

**Age:** Children less than 5-10 years do have a high chance of survival. Passengers between age group 15 to 35 died a lot.

**Embarked:** This is a very interesting feature. The chances of survival at C looks to be better than even though the majority of Pclass1 passengers got up at S. Passengers at Q were all from Pclass3.

**Parch+SibSp:** Having 1-2 siblings,spouse on board or 1-3 Parents shows a greater chance of probablity rather than being alone or having a large family travelling with you.

# Correlation Between The Features

<img src="images/Figure%202021-09-04%20111407%20(13).png" width="1024">

**Interpreting The Heatmap**
The first thing to note is that only the numeric features are compared as it is obvious that we cannot correlate between alphabets or strings. Before understanding the plot, let us see what exactly correlation is.

**POSITIVE CORRELATION:** If an increase in feature A leads to increase in feature B, then they are positively correlated. A value 1 means perfect positive correlation.

**NEGATIVE CORRELATION:** If an increase in feature A leads to decrease in feature B, then they are negatively correlated. A value -1 means perfect negative correlation.

Now lets say that two features are highly or perfectly correlated, so the increase in one leads to increase in the other. This means that both the features are containing highly similar information and there is very little or no variance in information. This is known as **MultiColinearity** as both of them contains almost the same information.

So do you think we should use both of them as one of them is redundant. While making or training models, we should try to eliminate redundant features as it reduces training time and many such advantages.

Now from the above heatmap,we can see that the features are not much correlated. The highest correlation is between SibSp and Parch i.e 0.41. So we can carry on with all features.

# Part 2: Cleaning the Data && Feature Engineering

<img src="images/Figure%202021-09-04%20111407%20(14).png" width="1024">

True that..the survival rate decreases as the age increases irrespective of the Pclass.

It is visible that being alone is harmful irrespective of Sex or Pclass except for Pclass3 where the chances of females who are alone is high than those with family.

<img src="images/Figure%202021-09-04%20111407%20(16).png" width="1024">

Now the above correlation plot, we can see some positively related features. Some of them being SibSp andd Family_Size and Parch and Family_Size and some negative ones like Alone and Family_Size.

# Part 3: Predictive Modeling

<img src="images/Figure%202021-09-04%20111407%20(17).png" width="1024">