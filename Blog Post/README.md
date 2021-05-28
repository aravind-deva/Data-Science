# AirBnB listings analysis
- import libraries
- read in csv files

## Preprocessing - Clean and Transform Data
- Identify numerical columns inferred as string
- Clean Data
- Transform Data Type

## Analyze city process
- With all the preprocessing functions defined, we can finally transform categorical columns and combine with numerical columns to analyse and derive insights 
- We will use pivot table and heatmap to visualize our results

## Prepare and analyze each city's data separately and together 

## Comparison of price differences between cities
- Let's narrow down to the most important column for grouping , reorder columns and use bar chart to visualize the stylized bar char

## Let us answer the following questions
- What are the key factors that drive the rental price? Also ,do we find any interesting findings?

Usually lease/tenant accommodations, the square_feet of the property is a good indicator of price but for holiday/guest hosting, the size of the property is not the most important driving factor. We see listings where 3000 square_feet properties are comparitively priced less.
'accommodates', 'bedrooms', 'beds', 'square_feet', 'bathrooms', 'guests_included', 'room_type_Entire home/apt', 'room_type_Private room', 'cancellation_policy_strict'
- For the same set of key factors, is there is a price difference between cities ?
Yes on average, the Boston city yields 50$ more rental prices compared to Seattle city . There are exceptions for higher values of 'accommodates' 
- Does review score/number of reviews impact the price or bookings?
Average review score has no/very less relationship with price. Well !It makes sense that reviews don't drive the price :-), rather they might impact the number of bookings(which can be analyzed with bookings data set )
If you had observed 'reviews_per_month' - the more the number of reviews the less the price of the property . Well it might NOT be a causal relationship. Rather guests who had negative experiences tend to give more reviews than people with positive/neutral experiences.
[Click for More info](https://medium.com/@aravind.deva/what-drives-the-rental-price-of-homes-and-rooms-for-guest-stay-496d7726d20)

#Contribute
The projects are submitted for review.However,in case you find a bug, please raise an issue.Kindly fork the repo and create a pull request to merge into the source repo.
