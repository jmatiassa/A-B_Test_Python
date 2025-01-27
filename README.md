## A-B_Test_Python

This project contains a full ab test analysis with statistical models in python.

The project is organized in the files below:
1. ABTestPractice1: main file with all the sub files
2. data: contains the csv used in the project
3. Jupyters: all the analysis coded in jupyter notebooks in python with;1.Preliminary Analyis, 2.Chi-square, 3.ExposureVariables
4.src: support file with a dictionary of formulas created to use during the analysis

The project is a solution to a practice exercise extracted in kaggle. All the answers are made during the 3 jupyter files.

The instructions are these:
Marketing A/B testing dataset
Marketing companies want to run successful campaigns, but the market is complex and several options can work. So normally they tun A/B tests, that is a randomized experimentation process wherein two or more versions of a variable (web page, page element, banner, etc.) are shown to different segments of people at the same time to determine which version leaves the maximum impact and drive business metrics.

The companies are interested in answering two questions:

Would the campaign be successful?
If the campaign was successful, how much of that success could be attributed to the ads?
With the second question in mind, we normally do an A/B test. The majority of the people will be exposed to ads (the experimental group). And a small portion of people (the control group) would instead see a Public Service Announcement (PSA) (or nothing) in the exact size and place the ad would normally be.

The idea of the dataset is to analyze the groups, find if the ads were successful, how much the company can make from the ads, and if the difference between the groups is statistically significant.

Data dictionary:

Index: Row index
user id: User ID (unique)
test group: If "ad" the person saw the advertisement, if "psa" they only saw the public service announcement
converted: If a person bought the product then True, else is False
total ads: Amount of ads seen by person
most ads day: Day that the person saw the biggest amount of ads
most ads hour: Hour of day that the person saw the biggest amount of ads


## Conclusions of the analysis

### Question 1: If the campaign was successful, how much of that success could be attributed to the ads?

The answer is in 2.chi_square jupyter notebook.
We can conclude that the test shows statistical significance with the two groups and the differences in the conversion rate are not due to random chance.

 There is strong evidence in the results due to a p-value inferior to 0.01 which implies that the probability of rejecting H0 if it were true is lower than 1%

In terms of business, we can accept that there is a possitive impact in the conversion rate and this is objectively explained due to the advertisement implemented. There is a increase of almost a 30% on conversion rate in front of the public service announcement psa

Based on this arguments, the exercise can be concluded as both main questions have been answered:

1. Would the campaign be successful? Yes, the campaign was successful because the Chi-square test showed a statistically significant difference in conversion rates between the ad and PSA groups, with the ad group having a higher conversion rate. The p-value (significantly lower than 0.05 and 0.01) suggests that the difference in conversion rates is unlikely to be due to random chance.
2. If the campaign was successful, how much of that success could be attributed to the ads?The success of the campaign can largely be attributed to the ads. The increase in conversion rates (around 30%) for the ad group compared to the PSA group provides a clear indication that the ads have had a positive impact on conversions. Additionally, by analyzing the relationship between total ads, most ads day, and most ads hour, you can further explore how ad exposure might influence the conversion success.

Nevertheless, knowing that there is a possitive effect with the advertising campaign, there has to be conducted an analysis in order to interprete the reasons that make the users increase their conversions after watching the adverts. The information that we have to do that is:

total ads: the total number of adds watched by the user

most ads day:Day that the person saw the biggest amount of ads


most ads hour:Hour of day that the person saw the biggest amount of ads

converted: dependent boolean variable

The study can be made implementing a logistic regression model that explain the effect of each independent variable on the conversion rate filtering only the advertisement group.


### Question 2:If the campaign was successful, how much of that success could be attributed to the ads?

The previous answer already answer this so the question can switch to something like: the campaign was successful, how could its results be improved?

The answer to this question is explained in 3.ExposureVariables file:

While the existing conversion rate is over 0.0255 the next information can help improve it:
1. Sending more than 27 adverts up to  2000 contains the largest conversion rates with 0.0835
2. Monday and Tuesday are the days that optimize the conversion rate due to the increase of almos a 20% on the conversion date in these values.
3. Adverts at 6 am and after 12 am are more effective than the average.























