# UC Berkeley ML/AI project: What Drives the Price of a Car?
(Data analysis and visualization with python, pandas, dataframe, matplotlib, seaborn, sklearn, StandardScaler, PolynomialFeatures, 
OneHotEncoder, train_test_split, GridSearchCV, LinearRegression, Ridge, Lasso, RidgeCV, LassoCV, ElasticNetCV, mean_absolute_error, mean_squared_error, r2_score, SequentialFeatureSelector \
Project location: https://github.com/parthabiswas1/UCBerkeley-project-whatDrivesPriceOfCar) 

## Problem Statement

### Understand why some used cars sell at a higer price, compared to others.
There are many factors like age, manufacturer, model, condition, odometer reading, fuel type, transmission, and drive that influence a used car buyer's decision to buy a car at a particular asking price. We need to find a way to identify the most important factors and their weightages that drive used car pricing so that the dealership can better price the car for quick sell.

### Objective of this project
Identify the most important factors that detrmine used car prices so the dealership can better price their vehicles and understand consumer preferences.

## Source of data

The dataset was provided by UC Berkeley School of Engineering and UC Berkley Haas school of Management for assigment for Professional Certificate in Machine Learning and Artificial Intelligence

## Approach

- Exploration ofthe data and formulation of data cleaning strategy
- Execution of the data cleaning strategy
- Creating a Modeling Plan
- Execution of the Models
- Evaluation of the Models
- Recommendatons to Car Dealers
  
Below are a summary of observations and findings.

## Data exploration results ** vehicles ** dataset

1. Did basic inspection of data - head(), dtypes, shape, describe()

2. Calculated missing counts percentages with isnull().sum() and identified columns that can be dropped because high % of data is missing. **size** qualifies as **70%** of size is missing.

3. VIN does not contribute much to the buying decision and 37% of VIN is missing. Good candidate to be dropped.

4. Decided that 'year' (more like age), 'odometer' readings are critical to buying decision. Only 0.28% of rows with 'year' are null and 1.03% of 'odometer' are null. These are small numbers and dropping these rows will ensure that all remaining rows have these values critical for the model to predict future price.

5. Looked for data outliers.   

   a) Cars older than 1950. Many of these cars have **placeholder price of $1.**.

   b) many have fake odometer readings (1, 10, 100 etc)     
   a)and b) together is **1213 rows only**, so decided to drop them.    

   c) There were **1386 rows** with odometer reading of 500K miles or more. Decided to drop them as they can skew the data.  

   d) Incorect pricing - Less than 100 and more than $200K. These were about 8.5% of the rows. Since these also can distort the model performance, decided to drop them.

## Data cleaning execution

1. **Data imputation** - All nulls in non numeric features are replaced with 'unknown'

2. **Drop Features (columns):**

   A. **id** - as this add no value to the car value.

   B. **size** - though size matters to a car value, **70%** of the data is missing.  

   C. **region and model** - Each contains too many elements, very difficult to convert to 'one hot encoding'.   

   D. **VIN** - adds no value to car value.  

   E. **state** - Though state is important (cars that sell in Texas more many be different from what sells in California) however with 'one hot encoding' it will explode the columns, so dropping it for how.

3. **Drop Observations (rows):**

   A.  **Year, Odometer** - These are critical and influences the price of used car, and the null rows are very few (Year - 0.28% and Odometer 1.03%). Dropping them will ensure that this data is avaliable in all remaining rows and help the performance of the models.  

   B. **Outliers** -
   
      i) Drop any vehicle older than 1950 and younger than 2025   

      ii) Odometer readings in negative or greater than 500K.

      iii) Price less than 100 or more than 200K

4. **Feature Engineering** -

   A. **Ordinal Encoding** -

     i) **Condition** has meaning- Encoded "salvage": 0, "fair": 1, "good": 2, "excellent": 3, "like new": 4, "new": 5, "unknown": -1

     ii) **title_status** has a clear hierarchy : "parts only": 0, "salvage": 1, "rebuilt": 2, "lien": 3, "unknown": 3, "missing": 3, "clean": 4. (gave same weightage to lien, unkown, missing )

   B. **One-hot encoding**
     
     fuel, transmission, drive, paint_color, type, manufacturer.


   C. **Feature creation** \
     i) Created a column called **car_age**. Everyone wants to know how old is the car and the age is important to the buying decision.  \

     Drop **year**, no more needed

   D **Scaling** Normalized numerical Features odometer, car_age.

(Note: I did not do PCA because I will be doing Ridge and LASSO. Ridge handles multicollinearity (features highly corelated to each other) by shrinking correlated coefficients. LASSO adds both shrinkage and feature selection. PCA removes collinearity but basically redundant in this scenario, it is also hard to interpret the data.)

![Coffee House coupon acceptance rates](images/coffee_coupon_acceptance.png)

## Recommendations

- Create a marketing campaign targeting young adult drivers who frequently go to bars, are not economically well off, do not have kid and are employed in more urban settings.
- Make the bar coupon more attractive to low income earners by increasing the discount rate or by combining it with resturant discount offers.

## Obsevations - **Coffee House coupons**

![Coffee House coupon acceptance rates](images/coffee_coupon_acceptance.png)

- Frequent coffee house visitors (1 or more per month) accepted more coupons(64.73% - 68.24%)
- Morning coffee house visitors (10AM) have a higher coupon acceptance rate (63.43%).
- Irrespective of marital status, a high rate of acceptance(44.4% - 65.6%) among drivers who go to coffee houses in the morning (10AM).
- Highest rate of acceptance (76.9%) is by divorced drivers who go to coffee houses late in the evening (10PM)
- Drivers who have no urgent place to go in the morning (10AM) or late evening (10PM) have higher coupon acceptance rates (63.4% - 68.7%)
- Young single females divers with some college earning less than $12.5k who went to coffee houses in the afternoon accepted coffee coupons at the highest rate
- Overall **49.5%** of coffee house coupons were accepted. This shows that close to majority of the  drivers were interested in Coffee House coupons</li>

## Recommendations
- Run a targeted marketing campaign focused on young female drivers, drivers who frequented coffee houses often and drives to nowhere particular and divorcees who go to coffee houses late in the evening.

