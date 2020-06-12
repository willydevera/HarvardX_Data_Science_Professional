####################################################
#
# HarvardX Professional Certificate in Data Science
#
####################################################
#
# Harvardx: PH125.9x: Data Science: Capstone
# Capstone Project: Movielens Recommender System
# Wilfredo A. de Vera
# Date submitted: June 12, 2020

###################################################
###################################################

################## Step 1. Data Load: Create Train and Validation Sets (this section provided by HarvardX)

# Introduction
# You will use the following code to generate your datasets. Develop your algorithm using the edx set.
# For a final test of your algorithm, predict movie ratings in the validation set as if they were unknown.
# RMSE will be used to evaluate how close your predictions are to the true values in the validation set.

# Important: The validation data should NOT be used for training your algorithm and should ONLY be used for
# evaluating the RMSE of your final algorithm. You should split the edx data into separate training and test
# sets to design and test your algorithm.

# Also remember that by accessing this site, you are agreeing to the terms of the edX Honor Code. This means
# you are expected to submit your own work and can be removed from the course for substituting another
# student's work as your own.

# Create test and validation sets
################################
# Create edx set, validation set
################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

# load libraries
library(tidyverse)
library(caret)
library(data.table)

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- read.table(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                      col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
# set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
set.seed(1)
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index, ]
temp <- movielens[test_index, ]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

# remove temporary files
rm(dl, ratings, movies, test_index, temp, movielens, removed)

###################################################
################## Step 2. Data Wrangling

### Step 2.1 Fix the dates from timestamp in both edx and validation datasets
# based of http://files.grouplens.org/datasets/movielens/ml-10m-README.html, the predictor timestamp
# represent seconds since midnight Coordinated Universal Time (UTC) of January 1, 1970

library(lubridate)
# dates <- as_datetime(edx$timestamp, origin = lubridate::origin, tz = "UTC")
# year(dates)
# month(dates):     1 - 12
# day(dates)     :  1 - 31 
# weekdays(dates):  Sun; Mon; Tue; Wed; Thu; Fri; Sat 
# wday(dates):        1;   2;   3;   4;   5;   6;   7 

# fix dates for edx dataset
edx <- edx %>% mutate(dates=as_datetime(timestamp,
                                        origin = lubridate::origin,
                                        tz = "UTC")) %>%
  mutate(year_rated = as.numeric(year(dates)),
         month_rated = as.numeric(month(dates)),
         day_rated = as.numeric(day(dates)),
         weekday_rated = weekdays(dates),
         wday_rated = as.numeric(wday(dates)))

# fix dates for validation set
validation <- validation %>% mutate(dates=as_datetime(timestamp,
                                                      origin = lubridate::origin,
                                                      tz = "UTC")) %>%
  mutate(year_rated = as.numeric(year(dates)),
         month_rated = as.numeric(month(dates)),
         day_rated = as.numeric(day(dates)),
         weekday_rated = weekdays(dates),
         wday_rated = as.numeric(wday(dates)))


### Step 2.2 Split the genres in edx - split multi values in ine column into multiple rows separated by pipe |
# https://stackoverflow.com/questions/44401023/splitting-multiple-values-in-one-column-into-multiple-rows-r

# separate rows for edx based of genre
edx <- edx %>% separate_rows(genres, sep = "\\|")

# separate rows for validation based of genre
validation <- validation %>% separate_rows(genres, sep = "\\|")


### Step 2.3 Extract year released in variable title which is of the format " (yyyy)" and convert:
###          userId as numeric
###          genres as factor
###          weekday as factor
###          title as character
# and then select the following variables:
# userId, movieId, title, rating, genres,
# year_rated, month_rated, day_rated, weekday_rated, wday_rated, year_released

# for edx
edx <- edx %>% mutate(title_for_trim = str_trim(title)) %>%
  extract(col=title_for_trim,
          into=c("title_real", "year_released"),
          regex="^(.*) \\(([0-9 \\-]*)\\)$",
          remove=F) %>%
  mutate(userId = as.numeric(userId),
         year_released=as.numeric(year_released),
         title_real=as.character(title_real),
         weekday_rated=as.factor(weekday_rated),
         genres=as.factor(genres)) %>%
  mutate(title=title_real) %>%
  select(userId, movieId, title, rating, genres,
         year_rated, month_rated, day_rated, weekday_rated, wday_rated, year_released)

# for validation
validation <- validation %>% mutate(title_for_trim = str_trim(title)) %>%
  extract(col=title_for_trim,
          into=c("title_real", "year_released"),
          regex="^(.*) \\(([0-9 \\-]*)\\)$",
          remove=F) %>%
  mutate(userId = as.numeric(userId),
         year_released=as.numeric(year_released),
         title_real=as.character(title_real),
         weekday_rated=as.factor(weekday_rated),
         genres=as.factor(genres)) %>%
  mutate(title=title_real) %>%
  select(userId, movieId, title, rating, genres,
         year_rated, month_rated, day_rated, weekday_rated, wday_rated, year_released)


### Step 2.4 Check for and handle NAs in edx and validation, if any
library(mice)

# check for NAs in edx
summary(edx)
# userId         movieId                title              rating            genres             year_rated     
#Min.   :    1   Min.   :    1   Forrest Gump :  124316   Min.   :0.500   Drama    :3910127   Min.   :1995  
#1st Qu.:18140   1st Qu.:  616   Toy Story    :  118950   1st Qu.:3.000   Comedy   :3540930   1st Qu.:2000  
#Median :35784   Median : 1748   Jurassic Park:  117440   Median :4.000   Action   :2560545   Median :2003  
#Mean   :35886   Mean   : 4277   True Lies    :  114115   Mean   :3.527   Thriller :2325899   Mean   :2002  
#3rd Qu.:53638   3rd Qu.: 3635   Aladdin      :  105865   3rd Qu.:4.000   Adventure:1908892   3rd Qu.:2005  
#Max.   :71567   Max.   :65133   Batman       :   98340   Max.   :5.000   Romance  :1712100   Max.   :2009  
#                                (Other)      :22692397                   (Other)  :7412930                 

#month_rated       day_rated       weekday_rated       wday_rated    year_released 
#Min.   : 1.000   Min.   : 1.00   Friday   :3195562   Min.   :1.000   Min.   :1915  
#1st Qu.: 4.000   1st Qu.: 8.00   Monday   :3651489   1st Qu.:2.000   1st Qu.:1987  
#Median : 7.000   Median :16.00   Saturday :2909444   Median :4.000   Median :1995  
#Mean   : 6.789   Mean   :15.61   Sunday   :3176216   Mean   :3.905   Mean   :1990  
#3rd Qu.:10.000   3rd Qu.:23.00   Thursday :3225293   3rd Qu.:6.000   3rd Qu.:1998  
#Max.   :12.000   Max.   :31.00   Tuesday  :3732433   Max.   :7.000   Max.   :2008  
#                                 Wednesday:3480986             

md.pattern(edx)

# check for NAs in validation
summary(validation)
# userId         movieId                title             rating            genres            year_rated     
#Min.   :    1   Min.   :    1   Forrest Gump :  13512   Min.   :0.500   Drama    :434071   Min.   :1995  
#1st Qu.:18137   1st Qu.:  611   Toy Story    :  13295   1st Qu.:3.000   Comedy   :393138   1st Qu.:2000  
#Median :35828   Median : 1734   Jurassic Park:  13084   Median :4.000   Action   :284804   Median :2003  
#Mean   :35899   Mean   : 4270   True Lies    :  12790   Mean   :3.526   Thriller :258536   Mean   :2002  
#3rd Qu.:53650   3rd Qu.: 3635   Aladdin      :  11790   3rd Qu.:4.000   Adventure:212182   3rd Qu.:2005  
#Max.   :71567   Max.   :65133   Batman       :  11052   Max.   :5.000   Romance  :189783   Max.   :2009  
#                                (Other)      :2520248                   (Other)  :823257                 

# month_rated       day_rated       weekday_rated      wday_rated    year_released 
#Min.   : 1.000   Min.   : 1.00   Friday   :355986   Min.   :1.000   Min.   :1915  
#1st Qu.: 4.000   1st Qu.: 8.00   Monday   :406935   1st Qu.:2.000   1st Qu.:1987  
#Median : 7.000   Median :16.00   Saturday :322619   Median :4.000   Median :1995  
#Mean   : 6.783   Mean   :15.61   Sunday   :352723   Mean   :3.905   Mean   :1990  
#3rd Qu.:10.000   3rd Qu.:23.00   Thursday :358516   3rd Qu.:6.000   3rd Qu.:1998  
#Max.   :12.000   Max.   :31.00   Tuesday  :413580   Max.   :7.000   Max.   :2008  
#                                 Wednesday:385412                 

md.pattern(validation)

## apparently there are neither NAs in edx nor in validation sets
anyNA(edx)
# FALSE
anyNA(validation)
# FALSE


## Step 2.5 Explore unique values for vectors other than dates in dataset

sort(unique(edx$rating))
# 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0

length(unique(edx$userId)) # 69878

length(unique(edx$movieId)) # 10677

length(unique(edx$title)) # 10407
# Note the difference of 270 unique titles as compared with unique movieIds

length(unique(edx$genres)) # 20

unique(edx$genres)

nrow(edx[(edx$genres == "(no genres listed)"), ])
# 7

nrow(validation[(validation$genres == "(no genres listed)"), ])
# 0

# note that there were 7 obs of "no genres listed" in edx - these are equivalent to NAs
# since we have over 23 million obs in edx, there's no harm removing these 7 obs.
# as well, fortunately there were 0 obs of "no genres listed" in validation set


## Step 2.6 Remove the records in edx and validation datasets where genres=="(no genres listed)"

edx <- edx %>% filter(!genres=="(no genres listed)") %>% droplevels()

validation <- validation %>% filter(!genres=="(no genres listed)") %>% droplevels()


###################################################
################## Step 3. Exploratory Data Analysis (EDA)

### Step 3.1 Generate summary statistics

# install xda
# reference: https://github.com/ujjwalkarn/xda
# library(devtools)
# install_github("ujjwalkarn/xda")

library(xda)
numSummary(edx)
#                     n     mean       sd   max    min   range nunique nzeros   iqr lowerbound upperbound noutlier
#userId        23371416 35885.68 20588.42 71567    1.0 71566.0   69878      0 35498   -35107.0   106885.0        0
#movieId       23371416  4277.29  9331.20 65133    1.0 65132.0   10676      0  3019    -3912.5     8163.5  1619908
#rating        23371416     3.53     1.05     5    0.5     4.5      10      0     1        1.5        5.5  1060268
#year_rated    23371416  2002.28     3.75  2009 1995.0    14.0      15      0     5     1992.5     2012.5        0
#month_rated   23371416     6.79     3.53    12    1.0    11.0      12      0     6       -5.0       19.0        0
#day_rated     23371416    15.61     8.80    31    1.0    30.0      31      0    15      -14.5       45.5        0
#wday_rated    23371416     3.91     1.95     7    1.0     6.0       7      0     4       -4.0       12.0        0
#year_released 23371416  1990.43    13.61  2008 1915.0    93.0      94      0    11     1970.5     2014.5  2008273

#              kurtosis skewness  mode miss miss%   1%     5%   25%   50%   75%   95%   99%
#userId         -1.1936  0.00747 59269    0     0  762 3798.0 18140 35784 53638 68087 70904
#movieId        17.7460  4.22158   356    0     0   10  107.0   616  1748  3635  8984 53129
#rating          0.0405 -0.60245     4    0     0    1    1.5     3     4     4     5     5
#year_rated     -1.1256 -0.15637  2000    0     0 1996 1996.0  2000  2003  2005  2008  2008
#month_rated    -1.2470 -0.09512    11    0     0    1    1.0     4     7    10    12    12
#day_rated      -1.1884  0.01544    20    0     0    1    2.0     8    16    23    29    31
#wday_rated     -1.1965  0.08068     3    0     0    1    1.0     2     4     6     7     7
#year_released   4.4115 -2.00068  1995    0     0 1939 1960.0  1987  1995  1998  2004  2007

charSummary(edx)
#                     n miss miss% unique
#title         23371416    0     0  10406
#genres        23371416    0     0     19
#weekday_rated 23371416    0     0      7

numSummary(validation)
#                    n     mean       sd   max    min   range nunique nzeros   iqr lowerbound upperbound noutlier
#userId        2595763 35899.44 20585.26 71567    1.0 71566.0   68534      0 35513   -35132.5   106919.5        0
#movieId       2595763  4269.59  9307.18 65133    1.0 65132.0    9802      0  3024    -3925.0     8171.0   179635
#rating        2595763     3.53     1.05     5    0.5     4.5      10      0     1        1.5        5.5   118061
#year_rated    2595763  2002.28     3.74  2009 1995.0    14.0      15      0     5     1992.5     2012.5        0
#month_rated   2595763     6.78     3.53    12    1.0    11.0      12      0     6       -5.0       19.0        0
#day_rated     2595763    15.61     8.79    31    1.0    30.0      31      0    15      -14.5       45.5        0
#wday_rated    2595763     3.90     1.95     7    1.0     6.0       7      0     4       -4.0       12.0        0
#year_released 2595763  1990.41    13.63  2008 1915.0    93.0      94      0    11     1970.5     2014.5   223978

#              kurtosis skewness  mode miss miss%   1%     5%   25%   50%   75%   95%   99%
#userId          -1.193  0.00572 59269    0     0  782 3795.0 18137 35828 53650 68085 70905
#movieId         17.788  4.22492   356    0     0   10  107.0   611  1734  3635  8984 53125
#rating           0.043 -0.60365     4    0     0    1    1.5     3     4     4     5     5
#year_rated      -1.125 -0.15585  2000    0     0 1996 1996.0  2000  2003  2005  2008  2008
#month_rated     -1.246 -0.09499    11    0     0    1    1.0     4     7    10    12    12
#day_rated       -1.186  0.01462    11    0     0    1    2.0     8    16    23    29    31
#wday_rated      -1.198  0.08042     3    0     0    1    1.0     2     4     6     7     7
#year_released    4.394 -1.99850  1995    0     0 1939 1959.0  1987  1995  1998  2004  2007

charSummary(validation)
#                    n miss miss% unique
#title         2595763    0     0   9550
#genres        2595763    0     0     19
#weekday_rated 2595763    0     0      7


### Step 3.2 Visualization
# since rating is the dependent variable, we will visualize it along with the independent variables

## 3.2.1 Distribution of edx ratings 

table(edx$rating)
#    0.5       1     1.5       2     2.5       3     3.5       4     4.5       5 
# 215932  844336  276711 1794242  874289 5467061 2110688 6730401 1418246 3639510

edx %>%
  ggplot(aes(rating)) + 
  geom_histogram(bins = 20, color = "black") +
  xlab('Ratings') +
  coord_trans(y = "sqrt") +
  ggtitle("Distribution of edx Ratings")


## 3.2.2 Distribution of edx ratings by genre

table(Genres=edx$genres, Ratings=edx$rating)
#                        Ratings
# Genres                0.5       1     1.5       2     2.5       3     3.5       4     4.5       5
#Action               27453  107752   37526  225415  109966  645535  236140  689323  140828  340607
#Adventure            18776   69632   25001  151103   76982  461053  178133  533391  113356  281465
#Animation             4564   13684    4702   28319   16637  105262   45910  140012   32106   75972
#Children              9837   35105   10403   62037   29300  185795   63318  201200   37507  103492
#Comedy               38345  155026   48554  307749  145206  867899  315541  983566  186532  492512
#Crime                 8653   35774   12103   82733   42566  278559  123147  399012  102823  242345
#Documentary           1368    2348     576    3569    2362   13957   10466   30424   10006   17990
#Drama                23282   98121   31412  238141  118964  840685  349756 1228072  270247  711447
#Fantasy               9906   34889   12680   72777   38800  210512   90887  254322   62267  138597
#Film-Noir              369    1315     432    3763    1911   16452    9064   40401   11526   33308
#Horror               12172   45612   12087   75881   31824  165640   60539  171344   32470   83916
#IMAX                   122     202      72     347     298    1241     905    2294    1106    1594
#Musical               4559   15147    4638   30451   13993  102373   36194  127955   23080   74690
#Mystery               3593   14067    4883   35594   18221  115734   55027  173517   45031  102665
#Romance              13533   54270   18110  126488   60598  414476  144214  511340   94161  274910
#Sci-Fi               15660   63415   20335  127610   58020  330249  119966  352858   70229  182841
#Thriller             19250   79591   27425  180965   89020  568911  214292  676325  137978  332142
#War                   3116   12279    4032   27350   13695   95220   41946  157922   36731  118856
#Western               1374    6107    1740   13950    5926   47508   15243   57123   10262   30161

edx %>% 
  group_by(genres) %>%
  summarize(n = n()) %>%
  arrange(desc(n))

edx %>% 
  group_by(rating, genres) %>%
  summarize(n = n()) %>%
  arrange(desc(n)) %>%
  ggplot(aes(x = factor(genres),
             y = n)) + 
  geom_boxplot() +
  geom_point() +
  xlab('Genres') +
  coord_trans(y = "sqrt") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ggtitle("Distribution of edx Ratings by Genre")

# Drama, Comedy, Action, Thriller, and Adventure top the genres


## 3.2.3 Distribution of edx ratings by year released

table(Year_Released=edx$year_released, Ratings=edx$rating)

edx %>% 
  group_by(year_released) %>%
  summarize(n = n()) %>%
  arrange(desc(n))
# There were many ratings made for movies released between 1994 and 1999, with peak at 1995.

edx %>% 
  group_by(rating, year_released) %>%
  summarize(n = n()) %>%
  arrange(desc(rating, n)) %>%
  ggplot(aes(x = factor(year_released),
             y = n)) + 
  geom_boxplot() +
  geom_point() +
  xlab('Year Released') +
  # coord_trans(y = "sqrt") +
  coord_trans(y = "log10") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ggtitle("Distribution of edx Ratings by Year Released")


## 3.2.4 Distribution of edx ratings by year rated
table(Year_Rated=edx$year_rated, Ratings=edx$rating)

edx %>% 
  group_by(year_rated) %>%
  summarize(n = n()) %>%
  arrange(desc(n))
# year_rated       n
#1       2000 2879265
#2       2005 2807323
#3       1996 2480192
#4       2008 1921944
#5       2006 1852912

edx %>% 
  group_by(rating, year_rated) %>%
  summarize(n = n()) %>%
  arrange(desc(rating, n)) %>%
  ggplot(aes(x = factor(year_rated),
             y = n)) + 
  geom_boxplot() +
  geom_point() +
  xlab('Year Rated') +
  coord_trans(y = "sqrt") +
  #coord_trans(y = "log10") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ggtitle("Distribution of edx Ratings by Year Rated")

# The year 2000 was the year when there were many ratings made.


## 3.2.5 Distribution of edx ratings by month rated
table(Month_Rated=edx$month_rated, Ratings=edx$rating)

edx %>% 
  group_by(month_rated) %>%
  summarize(n = n()) %>%
  arrange(desc(n))
#  month_rated       n
#1          11 2536721
#2          12 2334608
#3          10 2154366
# There were many ratings made between October to December, with peak at around November.

edx %>% 
  group_by(rating, month_rated) %>%
  summarize(n = n()) %>%
  arrange(desc(rating, n)) %>%
  ggplot(aes(x = factor(month_rated),
             y = n)) + 
  geom_boxplot() +
  geom_point() +
  xlab('Month Rated') +
  coord_trans(y = "sqrt") +
  ggtitle("Distribution of edx Ratings by Month Rated")



## 3.2.6 Distribution of edx ratings by day rated

table(Day_Rated=edx$day_rated, Ratings=edx$rating)

edx %>% 
  group_by(day_rated) %>%
  summarize(n = n()) %>%
  arrange(desc(n))
# day_rated      n
#1        20 858415
#2        11 855958
#3         3 849976
#4        22 841397
#5         2 815102
# The number of ratings peaked around the 20th day of the month.

edx %>% 
  group_by(rating, day_rated) %>%
  summarize(n = n()) %>%
  arrange(desc(rating, n)) %>%
  ggplot(aes(x = factor(day_rated),
             y = n)) + 
  geom_boxplot() +
  geom_point() +
  xlab('Day Rated') +
  #coord_trans(y = "sqrt") +
  coord_trans(y = "log10") +
  ggtitle("Distribution of edx Ratings by Day Rated")



## 3.2.7 Distribution of edx ratings by weekday rated
table(Weekday_Rated=factor(edx$weekday_rated,
                           c("Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday")),
      Ratings=edx$rating)

edx %>% 
  group_by(weekday_rated) %>%
  summarize(n = n()) %>%
  arrange(desc(n))
# weekday_rated       n
#1 Tuesday       3732432
#2 Monday        3651487
#3 Wednesday     3480984
#4 Thursday      3225293
#5 Friday        3195562
#6 Sunday        3176215
#7 Saturday      2909443
# Ratings peaked around Tuesday of the week.

edx %>% 
  group_by(rating, weekday_rated) %>%
  summarize(n = n()) %>%
  arrange(desc(rating, n)) %>%
  ggplot(aes(x = factor(weekday_rated,
                        c("Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday")),
             y = n)) + 
  geom_boxplot() +
  geom_point() +
  xlab('Weekday Rated') +
  coord_trans(y = "sqrt") +
  ggtitle("Distribution of edx Ratings by Weekday Rated")



## 3.2.8 Distribution of ratings of Top 1000 movieIds based of count and ratings
length(unique(edx$movieId))
# there are 10676 unique movieIds, we could perhaps get the ratings of the top 1000 movieIds to visualize

edx %>%
  dplyr::count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  ggtitle("Distribution of movieIds")

edx %>% 
  group_by(movieId, rating) %>%
  summarize(n = n()) %>%
  top_n(1000, rating) %>%
  arrange(desc(rating)) %>%
  ggplot(aes(x = factor(rating, c(seq(0.5, 5, 0.5))),
             y = n)) + 
  geom_boxplot() +
  geom_point() +
  xlab('Ratings of Top 1000 movieIds') +
  coord_trans(y = "log10") +
  ggtitle("Distribution of edx Top 1000 movieIds")

# As depicted in the histogram, edx movieId somehow follows a normal distribution, and the rating of 3 and 4
# are prevalent in top 1000 movieIds


## 3.2.9 Distribution of ratings of Top 1000 userIds based of count and rating

length(unique(edx$userId))
# there are 69878 unique userIds, we could perhaps get the top 1000 userIds to visualize

edx %>%
  dplyr::count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() +
  ggtitle("Distribution of userIds")

# As depicted in the histogram, edx userId is somehow skewed to the right

edx %>% 
  group_by(userId, rating) %>%
  summarize(n = n()) %>%
  top_n(1000, rating) %>%
  arrange(desc(rating)) %>%
  ggplot(aes(x = factor(rating, c(seq(0.5, 5, 0.5))),
             y = n)) + 
  geom_boxplot() +
  geom_point() +
  xlab('Ratings') +
  coord_trans(y = "log10") +
  ggtitle("Distribution of edx Ratings of Top 1000 userIds")

# The rating of 4 appears to be prevalent in Top 1000 movieIds


## 3.2.10 Plot of movieId vs. userId for Top 1000 movieIds rated 5

edx %>% filter(rating==5) %>%
  top_n(1000, movieId) %>%
  group_by(movieId, userId, genres) %>%
  summarize(n = n()) %>%
  arrange(desc(n)) %>%
  ggplot(aes(x = userId,
             y = movieId,
             color=factor(genres))) + 
  geom_point() +
  coord_trans(x = "log10") +
  coord_trans(y = "log10") +
  xlab('userId') +
  ylab('movieId') +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ggtitle("Scatterpot of edx movieId vs. userId for Top 1000 movieIds rated 5")

# There seems to be no obvious trend here, but it appears that movieIds ranging from 1 to 61500
# were rated 5 by most users.  



## 3.3 Check correlation

# reference: https://rafalab.github.io/dsbook/ section 33.5.5

# remove title, and rating from edx
# scale and convert remaining predictors to numeric
edx %>% select(-c(title,rating)) %>%
  mutate(genres = as.numeric(factor(genres)),
         weekday_rated = as.numeric(weekday_rated)) %>%
  as.matrix() %>%
  scale(center=TRUE, scale=TRUE) %>%
  cor()
#                     userId       movieId        genres  year_rated  month_rated    day_rated weekday_rated
#userId         1.0000000000  0.0044131400 -0.0005641342  0.01590382 -0.029052530  0.023181803  0.0197264352
#movieId        0.0044131400  1.0000000000 -0.0069251758  0.37403608 -0.006093064  0.009634426  0.0006405824
#genres        -0.0005641342 -0.0069251758  1.0000000000 -0.01397005  0.002986104 -0.001400377 -0.0022176001
#year_rated     0.0159038188  0.3740360762 -0.0139700454  1.00000000 -0.160440847  0.016555937  0.0222952807
#month_rated   -0.0290525301 -0.0060930640  0.0029861043 -0.16044085  1.000000000  0.018327125 -0.0035485096
#day_rated      0.0231818030  0.0096344260 -0.0014003772  0.01655594  0.018327125  1.000000000  0.0262582799
#weekday_rated  0.0197264352  0.0006405824 -0.0022176001  0.02229528 -0.003548510  0.026258280  1.0000000000
#wday_rated    -0.0082602565 -0.0113254040  0.0003456266 -0.02073185 -0.009066571 -0.013129841 -0.1887476207
#year_released  0.0001498285  0.2572660090 -0.0406835144  0.11006947 -0.022843381  0.008309666  0.0070822057

#                 wday_rated year_released
#userId        -0.0082602565  0.0001498285
#movieId       -0.0113254040  0.2572660090
#genres         0.0003456266 -0.0406835144
#year_rated    -0.0207318465  0.1100694666
#month_rated   -0.0090665707 -0.0228433814
#day_rated     -0.0131298412  0.0083096656
#weekday_rated -0.1887476207  0.0070822057
#wday_rated     1.0000000000 -0.0062515588
#year_released -0.0062515588  1.0000000000

# It is apparent that a very slight positive correlation exists between movieId vs. year_rated and
# year_released at 0.374036 and 0.257266, respectively.  


## 3.4 Check principal components

# reference: https://rafalab.github.io/dsbook/ section 33.5.5

edx %>% select(-c(title,rating)) %>%
  mutate(genres = as.numeric(factor(genres)),
         weekday_rated = as.numeric(weekday_rated)) %>%
  as.matrix() %>%
  scale(center=TRUE, scale=TRUE) %>%
  prcomp() %>%
  summary()

# Importance of components:
#                          PC1    PC2    PC3    PC4    PC5    PC6     PC7     PC8     PC9
#Standard deviation     1.2402 1.0925 1.0253 1.0076 1.0007 0.9810 0.92803 0.89976 0.75313
#Proportion of Variance 0.1709 0.1326 0.1168 0.1128 0.1113 0.1069 0.09569 0.08995 0.06302
#Cumulative Proportion  0.1709 0.3035 0.4203 0.5331 0.6444 0.7513 0.84702 0.93698 1.00000

# The first 7 components account for 84.702% of the variability


## 3.5 Check variable importance

# The earth package implements variable importance based on generalized cross validation (GCV),
# number of subset models the variable occurs (nsubsets) and residual sum of squares (RSS).
library(earth)
library(dplyr)

edx.scaled <- edx %>% select(userId, movieId, rating,
                             year_rated, month_rated, day_rated,
                             wday_rated, year_released) %>%
  scale(center=TRUE, scale=TRUE) %>%
  data.frame()

earth_edx <- earth(rating ~., data=edx.scaled)

plot(evimp(earth_edx, trim=FALSE))

earth_edx
# Selected 9 of 9 terms, and 3 of 7 predictors
# Termination condition: RSq changed by less than 0.001 at 9 terms
# Importance: year_released, movieId, year_rated, userId-unused, month_rated-unused, day_rated-unused, ...
# Number of terms at each degree of interaction: 1 8 (additive model)
# GCV 0.9750982    RSS 22789392    GRSq 0.02490186    RSq 0.0249032

# remove unnecessary dataframes prior to train/test split
# rm(edx.scaled)


###################################################
################## Step 4. Split edx into train_set and test_set

# split edx into 70% train_set and 30% test_set
# code trainControl if cross-validation is required 
library(caret)
set.seed(1)
#set.seed(1, sample.kind="Rounding")

test_index <- createDataPartition(y = edx$rating, times = 1,
                                  p = 0.30, list = FALSE)
train_set <- edx[-test_index,]

# from reference: https://rafalab.github.io/dsbook/  section 33.7.2, we exclude users and movies
# in the test_set that do not appear in the training_set. we'll do the same for validation
test_set <- edx[test_index,] %>%
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

validation <- validation %>%
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")


###################################################
################## Step 5. Modeling: Recommender Systems
options(digits=10)

# reference: https://rafalab.github.io/dsbook/ 
# section 33.7.3 Loss function

#  If N is the number of user-movie combinations, yu,i is the rating for movie i by user u,
#  and y^u,i is our prediction, then RMSE is defined as follows: 

#            RMSE  =  sqrt( (1/N) ∑(y^u,i − yu,i)^2 )   from u,i

RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

##### 5.1 first model: just the average
## reference: https://rafalab.github.io/dsbook/  Sectiom 33.7.4

## Yu,i = μ + ϵu,i
##    where: μ = true rating for all movies and users
##           ϵ = independent errors sampled from the same distribution centered at zero

# calculate mean rating of all movies in edx
mu_hat <- mean(train_set$rating); mu_hat
# 3.526972358

# predict RMSE on test_set
justmean_rmse_test_set <- RMSE(test_set$rating, mu_hat); justmean_rmse_test_set
# 1.051898234

# predict RMSE on validation_set
justmean_rmse_val_set <- RMSE(validation$rating, mu_hat); justmean_rmse_val_set
# 1.052557167

# The accuracy of the model is just based of the mean.
# Hence any higher value above this RMSE should be worse.

# create a results dataframe that indicates all RMSE results for both test_set and validation
results <- data.frame(method = "Just the mean",
                      RMSE_test_set = justmean_rmse_test_set,
                      RMSE_validation = justmean_rmse_val_set)

results %>% knitr::kable()
#|method        | RMSE_test_set| RMSE_validation|
#|:-------------|-------------:|---------------:|
#|Just the mean |   1.051898234|     1.052557167|


##### 5.2 movie effect model
## reference: https://rafalab.github.io/dsbook/  Sectiom 33.7.5

## Yu,i = μ + bi + ϵu,i
##    where: μ = true rating for all movies and users
##           ϵ = independent errors sampled from the same distribution centered at zero
##           bi = average ranking for movie i or bias

# compute mu and approximate bias b_i by computing the average of Yu,i - μ for each movie i on training set
mu <- mean(train_set$rating)

movie_avgs <- train_set %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))

# predict the ratings on the test_set
predicted_ratings_test <- test_set %>%
  left_join(movie_avgs, by='movieId') %>%
  mutate(rating_hat = mu + b_i) %>%
  pull(rating_hat)

# predict RMSE on test_set
movie_model_rmse_test_set <- RMSE(test_set$rating, predicted_ratings_test)
# 0.9409529753

# predict the ratings on validation set
predicted_ratings_val_set <- validation %>%
  left_join(movie_avgs, by='movieId') %>%
  mutate(rating_hat = mu + b_i) %>%
  pull(rating_hat)

# predict RMSE on validation
movie_model_rmse_val_set <- RMSE(validation$rating, predicted_ratings_val_set)
# 0.9411804404

# add rmse to results
results <-  results %>% add_row(method = "Movie effect",
                                RMSE_test_set = movie_model_rmse_test_set,
                                RMSE_validation = movie_model_rmse_val_set)
# view results
results %>% knitr::kable()
#|method        | RMSE_test_set| RMSE_validation|
#|:-------------|-------------:|---------------:|
#|Just the mean |  1.0518982335|    1.0525571670|
#|Movie effect  |  0.9409529753|    0.9411804404|


##### 5.3 movie + user effect model
## reference: https://rafalab.github.io/dsbook/  Section 33.7.6

## Yu,i = μ + bi + bu + ϵu,i
##    where: μ = true rating for all movies and users
##           ϵ = independent errors sampled from the same distribution centered at zero
##           bi = average ranking for movie i or bias
##           bu = user-specific effect 

# approximate bias b_i by computing the average of Yu,i - μ for each movie i on training set
# approximate bias b_u by computing the average of (yu,i - mu - bi) for each user on training set
movie_avgs <- train_set %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))

user_avgs <- train_set %>%
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# predict the ratings on the test_set
predicted_ratings_test <- test_set %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(rating_hat = mu + b_i + b_u) %>%
  pull(rating_hat)

# predict RMSE on test_set
movie_user_model_rmse_test_set <- RMSE(test_set$rating, predicted_ratings_test)
# 0.8577063124

# predict the ratings on validation set
predicted_ratings_val_set <- validation %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(rating_hat = mu + b_i + b_u) %>%
  pull(rating_hat)

# predict RMSE on validation
movie_user_model_rmse_val_set <- RMSE(validation$rating, predicted_ratings_val_set)
# 0.8641412163

# add the results
results <-  results %>% add_row(method = "Movie + user effect",
                                RMSE_test_set = movie_user_model_rmse_test_set,
                                RMSE_validation = movie_user_model_rmse_val_set)

results %>% knitr::kable()
#|method              | RMSE_test_set| RMSE_validation|
#|:-------------------|-------------:|---------------:|
#|Just the mean       |  1.0518982335|    1.0525571670|
#|Movie effect        |  0.9409529753|    0.9411804404|
#|Movie + user effect |  0.8577063124|    0.8641412163|


##### 5.4 movie + user + genre bias model - certain users may be biased on certain genres
##### movie + user + genre effect model

## Yu,i = μ + bi + bu + bg + ϵu,i
##    where: μ = true rating for all movies and users
##           ϵ = independent errors sampled from the same distribution centered at zero
##           bi = average ranking for movie i or bias
##           bu = user-specific effect
##           bg = genre-specific effect

# approximate bias b_i by computing the average of Yu,i - μ for each movie i on training set
# approximate bias b_u by computing the average of (yu,i - mu - bi) for each user on training set
# approximate bias b_g by computing the average of (yu,i - mu - bi - bu) for each genre on training set

movie_avgs <- train_set %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))

user_avgs <- train_set %>%
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

genres_avgs <- train_set %>%
  mutate(genres=factor(genres)) %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - mu - b_i - b_u))

# predict the ratings on the test_set
predicted_ratings_test <- test_set %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genres_avgs, by='genres') %>%
  mutate(rating_hat = mu + b_i + b_u + b_g) %>%
  pull(rating_hat)

# predict RMSE on test_set
movie_user_genres_model_rmse_test_set <- RMSE(test_set$rating, predicted_ratings_test)
# 0.8576231022

# predict the ratings on validation set
predicted_ratings_val_set <- validation %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genres_avgs, by='genres') %>%
  mutate(rating_hat = mu + b_i + b_u + b_g) %>%
  pull(rating_hat)

# predict RMSE on validation
movie_user_genres_model_rmse_val_set <- RMSE(validation$rating, predicted_ratings_val_set)
# 0.8640504507

# add the results
results <-  results %>% add_row(method = "Movie + user + genres effect",
                                RMSE_test_set = movie_user_genres_model_rmse_test_set,
                                RMSE_validation = movie_user_genres_model_rmse_val_set)

results %>% knitr::kable()
#|method                       | RMSE_test_set| RMSE_validation|
#|:----------------------------|-------------:|---------------:|
#|Just the mean                |  1.0518982335|    1.0525571670|
#|Movie effect                 |  0.9409529753|    0.9411804404|
#|Movie + user effect          |  0.8577063124|    0.8641412163|
#|Movie + user + genres effect |  0.8576231022|    0.8640504507|


##### 5.5 movie + user + genre + weekday_rated bias model
##### certain users may be biased on certain weekdays
#####     that they are rating a specific movie. For instance, a specific user who loves a specific genre
#####     might be rating a specific movie more differently on a Monday than on a Friday
##### movie + user + genre + weekday_rated effect model

## Yu,i = μ + bi + bu + bg + bd + ϵu,i
##    where: μ = true rating for all movies and users
##           ϵ = independent errors sampled from the same distribution centered at zero
##           bi = average ranking for movie i or bias
##           bu = user-specific effect
##           bg = genre-specific effect
##           bd = weekday-rated-specific effect

# approximate bias b_i by computing the average of Yu,i - μ for each movie i on training set
# approximate bias b_u by computing the average of (yu,i - mu - bi) for each user on training set
# approximate bias b_g by computing the average of (yu,i - mu - bi - bu) for each genre on training set
# approximate bias b_d by computing the average of (yu,i - mu - bi - bu - bg) for each weekday on training set

movie_avgs <- train_set %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))

user_avgs <- train_set %>%
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

genres_avgs <- train_set %>%
  mutate(genres=factor(genres)) %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - mu - b_i - b_u))

weekday_avgs <- train_set %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genres_avgs, by='genres') %>%
  group_by(weekday_rated) %>%
  summarize(b_d = mean(rating - mu - b_i - b_u - b_g))

# predict the ratings on the test_set
predicted_ratings_test <- test_set %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genres_avgs, by='genres') %>%
  left_join(weekday_avgs, by='weekday_rated') %>%
  mutate(rating_hat = mu + b_i + b_u + b_g + b_d) %>%
  pull(rating_hat)

# predict RMSE on test_set
movie_user_genres_day_model_rmse_test_set <- RMSE(test_set$rating, predicted_ratings_test)
# 0.8576227528

# predict the ratings on validation set
predicted_ratings_val_set <- validation %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genres_avgs, by='genres') %>%
  left_join(weekday_avgs, by='weekday_rated') %>%
  mutate(rating_hat = mu + b_i + b_u + b_g + b_d) %>%
  pull(rating_hat)

# predict RMSE on validation
movie_user_genres_day_model_rmse_val_set <- RMSE(validation$rating, predicted_ratings_val_set)
#  0.8640488429

# add the results
results <-  results %>% add_row(method = "Movie + user + genres + weekday_rated effect",
                                RMSE_test_set = movie_user_genres_day_model_rmse_test_set,
                                RMSE_validation = movie_user_genres_day_model_rmse_val_set)

results %>% knitr::kable()
#|method                                       | RMSE_test_set| RMSE_validation|
#|:--------------------------------------------|-------------:|---------------:|
#|Just the mean                                |  1.0518982335|    1.0525571670|
#|Movie effect                                 |  0.9409529753|    0.9411804404|
#|Movie + user effect                          |  0.8577063124|    0.8641412163|
#|Movie + user + genres effect                 |  0.8576231022|    0.8640504507|
#|Movie + user + genres + weekday_rated effect |  0.8576227528|    0.8640488429|


##### 5.6 movie + user + genre + weekday_rated + year_released  bias model
#         certain users may be biased rating on certain years that a specific movie was released
#####     while they are rating such movie.

# The year_released, movieId, and year_rated appear to the 3 most important variables after running
# the earth algorithm to check for variable importance, 

## Yu,i = μ + bi + bu + bg + bd + yr + ϵu,i
##    where: μ = true rating for all movies and users
##           ϵ = independent errors sampled from the same distribution centered at zero
##           bi = average ranking for movie i or bias
##           bu = user-specific effect
##           bg = genre-specific effect
##           bd = weekday-rated-specific effect
##           yr = year released effect

# approximate bias b_i by computing the average of Yu,i - μ for each movie i on training set
# approximate bias b_u by computing the average of (yu,i - mu - bi) for each user on training set
# approximate bias b_g by computing the average of (yu,i - mu - bi - bu) for each genre on training set
# approximate bias b_d by computing the average of (yu,i - mu - bi - bu - bg) for each weekday on training set
# approximate bias y_r by computing the average of (yu,i - mu - bi - bu - bg - bd) for year_released on training set

mu <- mean(train_set$rating)

movie_avgs <- train_set %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))

user_avgs <- train_set %>%
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

genres_avgs <- train_set %>%
  mutate(genres=factor(genres)) %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - mu - b_i - b_u))

weekday_avgs <- train_set %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genres_avgs, by='genres') %>%
  group_by(weekday_rated) %>%
  summarize(b_d = mean(rating - mu - b_i - b_u - b_g))

# year_released
yearreleased_avgs <- train_set %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genres_avgs, by='genres') %>%
  left_join(weekday_avgs, by='weekday_rated') %>%
  group_by(year_released) %>%
  summarize(y_r = mean(rating - mu - b_i - b_u - b_g - b_d))

# predict the ratings on the test_set
predicted_ratings_test <- test_set %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genres_avgs, by='genres') %>%
  left_join(weekday_avgs, by='weekday_rated') %>%
  left_join(yearreleased_avgs, by='year_released') %>%
  mutate(rating_hat = mu + b_i + b_u + b_g + b_d + y_r) %>%
  pull(rating_hat)

# predict RMSE on test_set
movie_user_genres_day_yr_released_model_rmse_test_set <- RMSE(test_set$rating, predicted_ratings_test)
# 0.8572703631

# predict the ratings on validation set
predicted_ratings_val_set <- validation %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genres_avgs, by='genres') %>%
  left_join(weekday_avgs, by='weekday_rated') %>%
  left_join(yearreleased_avgs, by='year_released') %>%
  mutate(rating_hat = mu + b_i + b_u + b_g + b_d + y_r) %>%
  pull(rating_hat)

# predict RMSE on validation
movie_user_genres_day_yr_released_model_rmse_val_set <-
  RMSE(validation$rating, predicted_ratings_val_set)
# 0.8636784216

# add the results
results <-  results %>% add_row(method = "Movie + user + genres + weekday_rated + year_released effect",
                                RMSE_test_set = movie_user_genres_day_yr_released_model_rmse_test_set,
                                RMSE_validation = movie_user_genres_day_yr_released_model_rmse_val_set)

results %>% knitr::kable()
#|method                                                                    | RMSE_test_set| RMSE_validation|
#|:-------------------------------------------------------------------------|-------------:|---------------:|
#|Just the mean                                                             |  1.0518982335|    1.0525571670|
#|Movie effect                                                              |  0.9409529753|    0.9411804404|
#|Movie + user effect                                                       |  0.8577063124|    0.8641412163|
#|Movie + user + genres effect                                              |  0.8576231022|    0.8640504507|
#|Movie + user + genres + weekday_rated effect                              |  0.8576227528|    0.8640488429|
#|Movie + user + genres + weekday_rated + year_released effect              |  0.8572703631|    0.8636784216|


###########################################

##### 5.7 movie + user + genre + weekday_rated + year_released + year_rated  bias model
#         certain users may be biased rating on certain years that a specific movie was released
#####     while they are rating such movie.

# The year_released, movieId, and year_rated appear to the 3 most important variables after running
# the earth algorithm to check for variable importance, 

## Yu,i = μ + bi + bu + bg + bd + yr + ya + ϵu,i
##    where: μ = true rating for all movies and users
##           ϵ = independent errors sampled from the same distribution centered at zero
##           bi = average ranking for movie i or bias
##           bu = user-specific effect
##           bg = genre-specific effect
##           bd = weekday-rated-specific effect
##           yr = year released effect
##           ya = year rated effect

# approximate bias b_i by computing the average of Yu,i - μ for each movie i on training set
# approximate bias b_u by computing the average of (yu,i - mu - bi) for each user on training set
# approximate bias b_g by computing the average of (yu,i - mu - bi - bu) for each genre on training set
# approximate bias b_d by computing the average of (yu,i - mu - bi - bu - bg) for each weekday on training set
# approximate bias y_r by computing the average of (yu,i - mu - bi - bu - bg - bd) for year_released on training set
# approximate bias y_a by computing the average of (yu,i - mu - bi - bu - bg - bd - yr) for year_rated on training set

mu <- mean(train_set$rating)

movie_avgs <- train_set %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))

user_avgs <- train_set %>%
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

genres_avgs <- train_set %>%
  mutate(genres=factor(genres)) %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - mu - b_i - b_u))

weekday_avgs <- train_set %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genres_avgs, by='genres') %>%
  group_by(weekday_rated) %>%
  summarize(b_d = mean(rating - mu - b_i - b_u - b_g))

# year_released
yearreleased_avgs <- train_set %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genres_avgs, by='genres') %>%
  left_join(weekday_avgs, by='weekday_rated') %>%
  group_by(year_released) %>%
  summarize(y_r = mean(rating - mu - b_i - b_u - b_g - b_d))

# year_rated
yearrated_avgs <- train_set %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genres_avgs, by='genres') %>%
  left_join(weekday_avgs, by='weekday_rated') %>%
  left_join(yearreleased_avgs, by='year_released') %>%
  group_by(year_rated) %>%
  summarize(y_a = mean(rating - mu - b_i - b_u - b_g - b_d - y_r))

# predict the ratings on the test_set
predicted_ratings_test <- test_set %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genres_avgs, by='genres') %>%
  left_join(weekday_avgs, by='weekday_rated') %>%
  left_join(yearreleased_avgs, by='year_released') %>%
  left_join(yearrated_avgs, by='year_rated') %>%
  mutate(rating_hat = mu + b_i + b_u + b_g + b_d + y_r + y_a) %>%
  pull(rating_hat)

# predict RMSE on test_set
movie_user_genres_day_yr_rated_released_model_rmse_test_set <- RMSE(test_set$rating, predicted_ratings_test)
# 0.8571846922

# predict the ratings on validation set
predicted_ratings_val_set <- validation %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genres_avgs, by='genres') %>%
  left_join(weekday_avgs, by='weekday_rated') %>%
  left_join(yearreleased_avgs, by='year_released') %>%
  left_join(yearrated_avgs, by='year_rated') %>%
  mutate(rating_hat = mu + b_i + b_u + b_g + b_d + y_r + y_a) %>%
  pull(rating_hat)

# predict RMSE on validation
movie_user_genres_day_yr_rated_released_model_rmse_val_set <-
  RMSE(validation$rating, predicted_ratings_val_set)
# 0.8635987481

# add the results
results <-  results %>% add_row(method = "Movie + user + genres + weekday_rated + year_released + year_rated effect",
                                RMSE_test_set = movie_user_genres_day_yr_rated_released_model_rmse_test_set,
                                RMSE_validation = movie_user_genres_day_yr_rated_released_model_rmse_val_set)

results %>% knitr::kable()
#|method                                                                    | RMSE_test_set| RMSE_validation|
#|:-------------------------------------------------------------------------|-------------:|---------------:|
#|Just the mean                                                             |  1.0518982335|    1.0525571670|
#|Movie effect                                                              |  0.9409529753|    0.9411804404|
#|Movie + user effect                                                       |  0.8577063124|    0.8641412163|
#|Movie + user + genres effect                                              |  0.8576231022|    0.8640504507|
#|Movie + user + genres + weekday_rated effect                              |  0.8576227528|    0.8640488429|
#|Movie + user + genres + weekday_rated + year_released effect              |  0.8572703631|    0.8636784216|
#|Movie + user + genres + weekday_rated + year_released + year_rated effect |  0.8571846922|    0.8635987481|



###########################################

##### 5.8 Regularized models
## reference: https://rafalab.github.io/dsbook/  Sectiom 33.9

# Regularization constrains the total variability of the effect sizes by penalizing large
# estimates that come from small sample sizes.

# ## Yu,i = μ + bi + ϵu,i 

# The values of b that minimize this equation are given by:
#    b_i(λ) = (1 / (λ + ni)) * ∑(Yu,i − μ)   , with u = 1 to ni,

#       where: ni is a number of ratings b for movie i.
#              λ is a tuning parameter, so use cross-validation to choose it

###########################################

#### 5.8.1 Regularized movie model
# Yu,i = μ + bi + ϵu,i 
# b_i(λ) = (1 / (λ + ni)) * ∑(Yu,i − μ)   , with u = 1 to ni for movie i

# define a sequence of lambdas
lambdas <- seq(0, 50, 0.25)

# predict rating and RMSE on test_set and determine optimal lambda value
rmses <- sapply(lambdas, function(lambda){
  
  # calculate mu
  mu <- mean(train_set$rating)
  
  # calculate regularized movie b_i
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = (1/(lambda + n())) * sum(rating - mu))
  
  # predict the ratings on test set
  predicted_ratings_test <- test_set %>%
    left_join(b_i, by='movieId') %>%
    mutate(rating_hat = mu + b_i) %>%
    pull(rating_hat)
  
  # compute the RMSE
  return(RMSE(test_set$rating, predicted_ratings_test))
})


# plot the RMSEs and lambdas
qplot(lambdas, rmses) + ggtitle("Regularized Movie: Plot of lambdas vs. RMSEs")

# determine which lambda with least rmse
lambdas[which.min(rmses)]
# 1.75

# therefore optimal λ for the regularized movie model = 1.75


# predict the ratings on the test_set at λ = 1.75
lambda <- 1.75

# calculate mu
mu <- mean(train_set$rating)

# calculate regularized movie b_i
b_i_reg <- test_set %>%
  group_by(movieId) %>%
  summarize(b_i = (1/(lambda + n())) * sum(rating - mu))

# predict the ratings on test set
predicted_ratings_test <- test_set %>%
  left_join(b_i_reg, by='movieId') %>%
  mutate(rating_hat = mu + b_i) %>%
  pull(rating_hat)

# predict RMSE on test_set
reg_movie_model_rmse_test_set <- RMSE(test_set$rating, predicted_ratings_test)
# 0.9399571015


# predict the ratings on validation at λ = 1.75
lambda <- 1.75

# calculate mu
mu <- mean(train_set$rating)

# calculate regularized movie b_i
b_i_reg <- validation %>%
  group_by(movieId) %>%
  summarize(b_i = (1/(lambda + n())) * sum(rating - mu))

# predict the ratings on validation
predicted_ratings_val_set <- validation %>%
  left_join(b_i_reg, by='movieId') %>%
  mutate(rating_hat = mu + b_i) %>%
  pull(rating_hat)

# predict RMSE on validation
reg_movie_model_rmse_val_set <- RMSE(validation$rating, predicted_ratings_val_set)
# 0.9369663126

# add the results
results <-  results %>% add_row(method = "Regularized movie model",
                                RMSE_test_set = reg_movie_model_rmse_test_set,
                                RMSE_validation = reg_movie_model_rmse_val_set)

results %>% knitr::kable()
#|method                                                                    | RMSE_test_set| RMSE_validation|
#|:-------------------------------------------------------------------------|-------------:|---------------:|
#|Just the mean                                                             |  1.0518982335|    1.0525571670|
#|Movie effect                                                              |  0.9409529753|    0.9411804404|
#|Movie + user effect                                                       |  0.8577063124|    0.8641412163|
#|Movie + user + genres effect                                              |  0.8576231022|    0.8640504507|
#|Movie + user + genres + weekday_rated effect                              |  0.8576227528|    0.8640488429|
#|Movie + user + genres + weekday_rated + year_released effect              |  0.8572703631|    0.8636784216|
#|Movie + user + genres + weekday_rated + year_released + year_rated effect |  0.8571846922|    0.8635987481|
#|Regularized movie model                                                   |  0.9399571015|    0.9369663126|


###########################################

#### 5.8.2 Regularized movie + user model
# Yu,i = μ + bi + bu + ϵu,i  
# b_i(λ) = (1 / (λ + ni)) * ∑(Yu,i − μ)   , with u = 1 to ni for movie i
# b_u(λ) = (1 / (λ + ni)) * ∑(Yu,i − μ)   , with u = 1 to ni for user i

# define a sequence of lambdas
lambdas <- seq(0, 50, 0.25)

# predict rating and RMSE on test_set and determine optimal lambda value
rmses <- sapply(lambdas, function(lambda){
  
  # calculate mu
  mu <- mean(train_set$rating)
  
  # calculate regularized movie b_i
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = (1/(lambda + n())) * sum(rating - mu))
  
  # calculate regularized user b_u
  b_u <- train_set %>%
    group_by(userId) %>%
    summarize(b_u = (1/(lambda + n())) * sum(rating - mu))
  
  # predict the ratings on test set
  predicted_ratings_test <- test_set %>%
    left_join(b_i, by='movieId') %>%
    left_join(b_u, by='userId') %>%
    mutate(rating_hat = mu + b_i + b_u) %>%
    pull(rating_hat)
  
  # compute the RMSE
  return(RMSE(predicted_ratings_test, test_set$rating))
})


# plot the RMSEs and lambdas
qplot(lambdas, rmses) + ggtitle("Regularized Movie + User: Plot of lambdas vs. RMSEs")

# determine which lambda with least rmse
lambdas[which.min(rmses)]
# 27.5

# therefore optimal λ for the regularized movie + user model= 27.5


# predict the ratings on the test_set at λ = 27.5
lambda <- 27.5

# calculate mu
mu <- mean(train_set$rating)

# calculate regularized movie b_i
b_i_reg <- test_set %>%
  group_by(movieId) %>%
  summarize(b_i = (1/(lambda + n())) * sum(rating - mu))

# calculate regularized user b_u
b_u_reg <- test_set %>%
  group_by(userId) %>%
  summarize(b_u = (1/(lambda + n())) * sum(rating - mu))

# predict the ratings on test set
predicted_ratings_test <- test_set %>%
  left_join(b_i_reg, by='movieId') %>%
  left_join(b_u_reg, by='userId') %>%
  mutate(rating_hat = mu + b_i + b_u) %>%
  pull(rating_hat)

# predict RMSE on test_set
reg_movie_user_model_rmse_test_set <- RMSE(test_set$rating, predicted_ratings_test)
# 0.8692830213


# predict the ratings on validation at λ = 27.5
lambda <- 27.5

# calculate mu
mu <- mean(train_set$rating)

# calculate regularized movie b_i
b_i_reg <- validation %>%
  group_by(movieId) %>%
  summarize(b_i = (1/(lambda + n())) * sum(rating - mu))

# calculate regularized user b_u
b_u_reg <- validation %>%
  group_by(userId) %>%
  summarize(b_u = (1/(lambda + n())) * sum(rating - mu))

# predict the ratings on validation
predicted_ratings_val_set <- validation %>%
  left_join(b_i_reg, by='movieId') %>%
  left_join(b_u_reg, by='userId') %>%
  mutate(rating_hat = mu + b_i + b_u) %>%
  pull(rating_hat)

# predict RMSE on validation
reg_movie_user_model_rmse_val_set <- RMSE(validation$rating, predicted_ratings_val_set)
# 0.8571358019

# add the results
results <-  results %>% add_row(method = "Regularized movie + user model",
                                RMSE_test_set = reg_movie_user_model_rmse_test_set,
                                RMSE_validation = reg_movie_user_model_rmse_val_set)

results %>% knitr::kable()
#|method                                                                    | RMSE_test_set| RMSE_validation|
#|:-------------------------------------------------------------------------|-------------:|---------------:|
#|Just the mean                                                             |  1.0518982335|    1.0525571670|
#|Movie effect                                                              |  0.9409529753|    0.9411804404|
#|Movie + user effect                                                       |  0.8577063124|    0.8641412163|
#|Movie + user + genres effect                                              |  0.8576231022|    0.8640504507|
#|Movie + user + genres + weekday_rated effect                              |  0.8576227528|    0.8640488429|
#|Movie + user + genres + weekday_rated + year_released effect              |  0.8572703631|    0.8636784216|
#|Movie + user + genres + weekday_rated + year_released + year_rated effect |  0.8571846922|    0.8635987481|
#|Regularized movie model                                                   |  0.9399571015|    0.9369663126|
#|Regularized movie + user model                                            |  0.8692830213|    0.8571358019|


###########################################

#### 5.8.3 Regularized movie + user + genres model
# Yu,i = μ + bi + bu + bg + ϵu,i 
# b_i(λ) = (1 / (λ + ni)) * ∑(Yu,i − μ)   , with u = 1 to ni for movie i
# b_u(λ) = (1 / (λ + ni)) * ∑(Yu,i − μ)   , with u = 1 to ni for user i
# b_g(λ) = (1 / (λ + ni)) * ∑(Yu,i − μ)   , with u = 1 to ni for genre i

# define a sequence of lambdas
lambdas <- seq(0, 50, 0.25)

# predict rating and RMSE on test_set and determine optimal lambda value
rmses <- sapply(lambdas, function(lambda){
  
  # calculate mu
  mu <- mean(train_set$rating)
  
  # calculate regularized movie b_i
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = (1/(lambda + n())) * sum(rating - mu))
  
  # calculate regularized user b_u
  b_u <- train_set %>%
    group_by(userId) %>%
    summarize(b_u = (1/(lambda + n())) * sum(rating - mu))
  
  # calculate regularized genres b_g
  b_g <- train_set %>%
    group_by(genres) %>%
    summarize(b_g = (1/(lambda + n())) * sum(rating - mu))
  
  # predict the ratings on test set
  predicted_ratings_test <- test_set %>%
    left_join(b_i, by='movieId') %>%
    left_join(b_u, by='userId') %>%
    left_join(b_g, by='genres') %>%
    mutate(rating_hat = mu + b_i + b_u + b_g) %>%
    pull(rating_hat)
  
  # compute the RMSE
  return(RMSE(predicted_ratings_test, test_set$rating))
})


# plot the RMSEs and lambdas
qplot(lambdas, rmses) + ggtitle("Regularized Movie + User + Genres: Plot of lambdas vs. RMSEs")

# determine which lambda with least rmse
lambdas[which.min(rmses)]
# 33.25

# therefore optimal λ for the regularized movie + user + genres model  = 33.25


# predict the ratings on the test_set at λ = 33.25
lambda <- 33.25

# calculate mu
mu <- mean(train_set$rating)

# calculate regularized movie b_i
b_i_reg <- test_set %>%
  group_by(movieId) %>%
  summarize(b_i = (1/(lambda + n())) * sum(rating - mu))

# calculate regularized user b_u
b_u_reg <- test_set %>%
  group_by(userId) %>%
  summarize(b_u = (1/(lambda + n())) * sum(rating - mu))

# calculate regularized genres b_g
b_g_reg <- test_set %>%
  group_by(genres) %>%
  summarize(b_g = (1/(lambda + n())) * sum(rating - mu))

# predict the ratings on test set
predicted_ratings_test <- test_set %>%
  left_join(b_i_reg, by='movieId') %>%
  left_join(b_u_reg, by='userId') %>%
  left_join(b_g_reg, by='genres') %>%
  mutate(rating_hat = mu + b_i + b_u + b_g) %>%
  pull(rating_hat)

# predict RMSE on test_set
reg_movie_user_genres_model_rmse_test_set <- RMSE(test_set$rating, predicted_ratings_test)
# 0.8788499509


# predict the ratings on validation at λ = 33.25
lambda <- 33.25

# calculate mu
mu <- mean(train_set$rating)

# calculate regularized movie b_i
b_i_reg <- validation %>%
  group_by(movieId) %>%
  summarize(b_i = (1/(lambda + n())) * sum(rating - mu))

# calculate regularized user b_u
b_u_reg <- validation %>%
  group_by(userId) %>%
  summarize(b_u = (1/(lambda + n())) * sum(rating - mu))

# calculate regularized genres b_g
b_g_reg <- validation %>%
  group_by(genres) %>%
  summarize(b_g = (1/(lambda + n())) * sum(rating - mu))

# predict the ratings on validation
predicted_ratings_val_set <- validation %>%
  left_join(b_i_reg, by='movieId') %>%
  left_join(b_u_reg, by='userId') %>%
  left_join(b_g_reg, by='genres') %>%
  mutate(rating_hat = mu + b_i + b_u + b_g) %>%
  pull(rating_hat)

# predict RMSE on validation
reg_movie_user_genres_model_rmse_val_set <- RMSE(validation$rating, predicted_ratings_val_set)
# 0.8684473123

# add the results
results <-  results %>% add_row(method = "Regularized movie + user + genres model",
                                RMSE_test_set = reg_movie_user_genres_model_rmse_test_set,
                                RMSE_validation = reg_movie_user_genres_model_rmse_val_set)

results %>% knitr::kable()
#|method                                                                    | RMSE_test_set| RMSE_validation|
#|:-------------------------------------------------------------------------|-------------:|---------------:|
#|Just the mean                                                             |  1.0518982335|    1.0525571670|
#|Movie effect                                                              |  0.9409529753|    0.9411804404|
#|Movie + user effect                                                       |  0.8577063124|    0.8641412163|
#|Movie + user + genres effect                                              |  0.8576231022|    0.8640504507|
#|Movie + user + genres + weekday_rated effect                              |  0.8576227528|    0.8640488429|
#|Movie + user + genres + weekday_rated + year_released effect              |  0.8572703631|    0.8636784216|
#|Movie + user + genres + weekday_rated + year_released + year_rated effect |  0.8571846922|    0.8635987481|
#|Regularized movie model                                                   |  0.9399571015|    0.9369663126|
#|Regularized movie + user model                                            |  0.8692830213|    0.8571358019|
#|Regularized movie + user + genres model                                   |  0.8788499509|    0.8684473123|



###########################################

#### 5.8.4 Regularized movie + user + genres + weekday_rated model
# Yu,i = μ + bi + bu + bg + bd + ϵu,i 
# b_i(λ) = (1 / (λ + ni)) * ∑(Yu,i − μ)   , with u = 1 to ni for movie i
# b_u(λ) = (1 / (λ + ni)) * ∑(Yu,i − μ)   , with u = 1 to ni for user i
# b_g(λ) = (1 / (λ + ni)) * ∑(Yu,i − μ)   , with u = 1 to ni for genre i
# b_d(λ) = (1 / (λ + ni)) * ∑(Yu,i − μ)   , with u = 1 to ni for weekday_rated i

# define a sequence of lambdas
lambdas <- seq(0, 70, 0.25)

# predict rating and RMSE on test_set and determine optimal lambda value
rmses <- sapply(lambdas, function(lambda){
  
  # calculate mu
  mu <- mean(train_set$rating)
  
  # calculate regularized movie b_i
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = (1/(lambda + n())) * sum(rating - mu))
  
  # calculate regularized user b_u
  b_u <- train_set %>%
    group_by(userId) %>%
    summarize(b_u = (1/(lambda + n())) * sum(rating - mu))
  
  # calculate regularized genres b_g
  b_g <- train_set %>%
    group_by(genres) %>%
    summarize(b_g = (1/(lambda + n())) * sum(rating - mu))
  
  # calculate regularized weekday_rated b_d
  b_d <- train_set %>%
    group_by(weekday_rated) %>%
    summarize(b_d = (1/(lambda + n())) * sum(rating - mu))
  
  # predict the ratings on test set
  predicted_ratings_test <- test_set %>%
    left_join(b_i, by='movieId') %>%
    left_join(b_u, by='userId') %>%
    left_join(b_g, by='genres') %>%
    left_join(b_d, by='weekday_rated') %>%
    mutate(rating_hat = mu + b_i + b_u + b_g + b_d) %>%
    pull(rating_hat)
  
  # compute the RMSE
  return(RMSE(predicted_ratings_test, test_set$rating))
})


# plot the RMSEs and lambdas
qplot(lambdas, rmses) +
  ggtitle("Regularized Movie + User + Genres + Weekday_Rated: Plot of lambdas vs. RMSEs")

# determine which lambda with least rmse
lambdas[which.min(rmses)]
# 33.25

# therefore optimal λ for the regularized movie + user + genres + weekday_rated model  = 33.25


# predict the ratings on the test_set at λ = 33.25
lambda <- 33.25

# calculate mu
mu <- mean(train_set$rating)

# calculate regularized movie b_i
b_i_reg <- test_set %>%
  group_by(movieId) %>%
  summarize(b_i = (1/(lambda + n())) * sum(rating - mu))

# calculate regularized user b_u
b_u_reg <- test_set %>%
  group_by(userId) %>%
  summarize(b_u = (1/(lambda + n())) * sum(rating - mu))

# calculate regularized genres b_g
b_g_reg <- test_set %>%
  group_by(genres) %>%
  summarize(b_g = (1/(lambda + n())) * sum(rating - mu))

# calculate regularized weekday_rated b_d
b_d_reg <- test_set %>%
  group_by(weekday_rated) %>%
  summarize(b_d = (1/(lambda + n())) * sum(rating - mu))

# predict the ratings on test set
predicted_ratings_test <- test_set %>%
  left_join(b_i_reg, by='movieId') %>%
  left_join(b_u_reg, by='userId') %>%
  left_join(b_g_reg, by='genres') %>%
  left_join(b_d_reg, by='weekday_rated') %>%
  mutate(rating_hat = mu + b_i + b_u + b_g + b_d) %>%
  pull(rating_hat)

# predict RMSE on test_set
reg_movie_user_genres_day_model_rmse_test_set <- RMSE(test_set$rating, predicted_ratings_test)
# 0.8788909578


# predict the ratings on validation at λ = 33.25
lambda <- 33.25

# calculate mu
mu <- mean(train_set$rating)

# calculate regularized movie b_i
b_i_reg <- validation %>%
  group_by(movieId) %>%
  summarize(b_i = (1/(lambda + n())) * sum(rating - mu))

# calculate regularized user b_u
b_u_reg <- validation %>%
  group_by(userId) %>%
  summarize(b_u = (1/(lambda + n())) * sum(rating - mu))

# calculate regularized genres b_g
b_g_reg <- validation %>%
  group_by(genres) %>%
  summarize(b_g = (1/(lambda + n())) * sum(rating - mu))

# calculate regularized weekday_rated b_d
b_d_reg <- validation %>%
  group_by(weekday_rated) %>%
  summarize(b_d = (1/(lambda + n())) * sum(rating - mu))


# predict the ratings on validation
predicted_ratings_val_set <- validation %>%
  left_join(b_i_reg, by='movieId') %>%
  left_join(b_u_reg, by='userId') %>%
  left_join(b_g_reg, by='genres') %>%
  left_join(b_d_reg, by='weekday_rated') %>%
  mutate(rating_hat = mu + b_i + b_u + b_g + b_d) %>%
  pull(rating_hat)


# predict RMSE on validation
reg_movie_user_genres_day_model_rmse_val_set <- RMSE(validation$rating, predicted_ratings_val_set)
# 0.8684984877

# add the results
results <-  results %>% add_row(method = "Regularized movie + user + genres + weekday_rated model",
                                RMSE_test_set = reg_movie_user_genres_day_model_rmse_test_set,
                                RMSE_validation = reg_movie_user_genres_day_model_rmse_val_set)

results %>% knitr::kable()
#|method                                                                    | RMSE_test_set| RMSE_validation|
#|:-------------------------------------------------------------------------|-------------:|---------------:|
#|Just the mean                                                             |  1.0518982335|    1.0525571670|
#|Movie effect                                                              |  0.9409529753|    0.9411804404|
#|Movie + user effect                                                       |  0.8577063124|    0.8641412163|
#|Movie + user + genres effect                                              |  0.8576231022|    0.8640504507|
#|Movie + user + genres + weekday_rated effect                              |  0.8576227528|    0.8640488429|
#|Movie + user + genres + weekday_rated + year_released effect              |  0.8572703631|    0.8636784216|
#|Movie + user + genres + weekday_rated + year_released + year_rated effect |  0.8571846922|    0.8635987481|
#|Regularized movie model                                                   |  0.9399571015|    0.9369663126|
#|Regularized movie + user model                                            |  0.8692830213|    0.8571358019|
#|Regularized movie + user + genres model                                   |  0.8788499509|    0.8684473123|
#|Regularized movie + user + genres + weekday_rated model                   |  0.8788909578|    0.8684984877|


###############################################

#### 5.8.5 Regularized movie + user + genres + weekday_rated + year_released model
# Yu,i = μ + bi + bu + bg + bd + ϵu,i 
# b_i(λ) = (1 / (λ + ni)) * ∑(Yu,i − μ)   , with u = 1 to ni for movie i
# b_u(λ) = (1 / (λ + ni)) * ∑(Yu,i − μ)   , with u = 1 to ni for user i
# b_g(λ) = (1 / (λ + ni)) * ∑(Yu,i − μ)   , with u = 1 to ni for genre i
# b_d(λ) = (1 / (λ + ni)) * ∑(Yu,i − μ)   , with u = 1 to ni for weekday_rated i
# y_r(λ) = (1 / (λ + ni)) * ∑(Yu,i − μ)   , with u = 1 to ni for year_released i

# define a sequence of lambdas
lambdas <- seq(0, 100, 0.25)

# predict rating and RMSE on test_set and determine optimal lambda value
rmses <- sapply(lambdas, function(lambda){
  
  # calculate mu
  mu <- mean(train_set$rating)
  
  # calculate regularized movie b_i
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = (1/(lambda + n())) * sum(rating - mu))
  
  # calculate regularized user b_u
  b_u <- train_set %>%
    group_by(userId) %>%
    summarize(b_u = (1/(lambda + n())) * sum(rating - mu))
  
  # calculate regularized genres b_g
  b_g <- train_set %>%
    group_by(genres) %>%
    summarize(b_g = (1/(lambda + n())) * sum(rating - mu))
  
  # calculate regularized weekday_rated b_d
  b_d <- train_set %>%
    group_by(weekday_rated) %>%
    summarize(b_d = (1/(lambda + n())) * sum(rating - mu))
  
  # calculate regularized year_released y_r
  y_r <- train_set %>%
    group_by(year_released) %>%
    summarize(y_r = (1/(lambda + n())) * sum(rating - mu))
  
  # predict the ratings on test set
  predicted_ratings_test <- test_set %>%
    left_join(b_i, by='movieId') %>%
    left_join(b_u, by='userId') %>%
    left_join(b_g, by='genres') %>%
    left_join(b_d, by='weekday_rated') %>%
    left_join(y_r, by='year_released') %>%
    mutate(rating_hat = mu + b_i + b_u + b_g + b_d + y_r) %>%
    pull(rating_hat)
  
  # compute the RMSE
  return(RMSE(predicted_ratings_test, test_set$rating))
})


# plot the RMSEs and lambdas
qplot(lambdas, rmses) +
  ggtitle("Regularized Movie + User + Genres + Weekday_Rated + Year_Released: Plot of lambdas vs. RMSEs")

# determine which lambda with least rmse
lambdas[which.min(rmses)]
# 42.75

# therefore optimal λ for the regularized movie + user + genres + weekday_rated + year_released model  =  42.75


# predict the ratings on the test_set at λ = 42.75
lambda <- 42.75
# calculate mu
mu <- mean(train_set$rating)

# calculate regularized movie b_i
b_i_reg <- test_set %>%
  group_by(movieId) %>%
  summarize(b_i = (1/(lambda + n())) * sum(rating - mu))

# calculate regularized user b_u
b_u_reg <- test_set %>%
  group_by(userId) %>%
  summarize(b_u = (1/(lambda + n())) * sum(rating - mu))

# calculate regularized genres b_g
b_g_reg <- test_set %>%
  group_by(genres) %>%
  summarize(b_g = (1/(lambda + n())) * sum(rating - mu))

# calculate regularized weekday_rated b_d
b_d_reg <- test_set %>%
  group_by(weekday_rated) %>%
  summarize(b_d = (1/(lambda + n())) * sum(rating - mu))

# calculate regularized year_released y_r
y_r_reg <- test_set %>%
  group_by(year_released) %>%
  summarize(y_r = (1/(lambda + n())) * sum(rating - mu))

# predict the ratings on test set
predicted_ratings_test <- test_set %>%
  left_join(b_i_reg, by='movieId') %>%
  left_join(b_u_reg, by='userId') %>%
  left_join(b_g_reg, by='genres') %>%
  left_join(b_d_reg, by='weekday_rated') %>%
  left_join(y_r_reg, by='year_released') %>%
  mutate(rating_hat = mu + b_i + b_u + b_g + b_d + y_r) %>%
  pull(rating_hat)

# predict RMSE on test_set
reg_movie_user_genres_day_yr_released_model_rmse_test_set <-
  RMSE(test_set$rating, predicted_ratings_test)
# 0.8947160951


# predict the ratings on validation at λ = 42.75
lambda <- 42.75

# calculate mu
mu <- mean(train_set$rating)

# calculate regularized movie b_i
b_i_reg <- validation %>%
  group_by(movieId) %>%
  summarize(b_i = (1/(lambda + n())) * sum(rating - mu))

# calculate regularized user b_u
b_u_reg <- validation %>%
  group_by(userId) %>%
  summarize(b_u = (1/(lambda + n())) * sum(rating - mu))

# calculate regularized genres b_g
b_g_reg <- validation %>%
  group_by(genres) %>%
  summarize(b_g = (1/(lambda + n())) * sum(rating - mu))

# calculate regularized weekday_rated b_d
b_d_reg <- validation %>%
  group_by(weekday_rated) %>%
  summarize(b_d = (1/(lambda + n())) * sum(rating - mu))

# calculate regularized year_released y_r
y_r_reg <- validation %>%
  group_by(year_released) %>%
  summarize(y_r = (1/(lambda + n())) * sum(rating - mu))

# predict the ratings on validation
predicted_ratings_val_set <- validation %>%
  left_join(b_i_reg, by='movieId') %>%
  left_join(b_u_reg, by='userId') %>%
  left_join(b_g_reg, by='genres') %>%
  left_join(b_d_reg, by='weekday_rated') %>%
  left_join(y_r_reg, by='year_released') %>%
  mutate(rating_hat = mu + b_i + b_u + b_g + b_d + y_r) %>%
  pull(rating_hat)

# predict RMSE on validation
reg_movie_user_genres_day_yr_released_model_rmse_val_set <-
  RMSE(validation$rating, predicted_ratings_val_set)
#  0.8866161287

# add the results
results <-  results %>% add_row(method = "Regularized movie + user + genres + weekday_rated + year_released model",
                                RMSE_test_set = reg_movie_user_genres_day_yr_released_model_rmse_test_set,
                                RMSE_validation = reg_movie_user_genres_day_yr_released_model_rmse_val_set)

results %>% knitr::kable()
#|method                                                                      | RMSE_test_set| RMSE_validation|
#|:---------------------------------------------------------------------------|-------------:|---------------:|
#|Just the mean                                                               |  1.0518982335|    1.0525571670|
#|Movie effect                                                                |  0.9409529753|    0.9411804404|
#|Movie + user effect                                                         |  0.8577063124|    0.8641412163|
#|Movie + user + genres effect                                                |  0.8576231022|    0.8640504507|
#|Movie + user + genres + weekday_rated effect                                |  0.8576227528|    0.8640488429|
#|Movie + user + genres + weekday_rated + year_released effect                |  0.8572703631|    0.8636784216|
#|Movie + user + genres + weekday_rated + year_released + year_rated effect   |  0.8571846922|    0.8635987481|
#|Regularized movie model                                                     |  0.9399571015|    0.9369663126|
#|Regularized movie + user model                                              |  0.8692830213|    0.8571358019|
#|Regularized movie + user + genres model                                     |  0.8788499509|    0.8684473123|
#|Regularized movie + user + genres + weekday_rated model                     |  0.8788909578|    0.8684984877|
#|Regularized movie + user + genres + weekday_rated + year_released model     |  0.8947160951|    0.8866161287|


################

#### 5.8.6 Regularized movie + user + genres + weekday_rated + year_released + year_rated model
# Yu,i = μ + bi + bu + bg + bd + ϵu,i 
# b_i(λ) = (1 / (λ + ni)) * ∑(Yu,i − μ)   , with u = 1 to ni for movie i
# b_u(λ) = (1 / (λ + ni)) * ∑(Yu,i − μ)   , with u = 1 to ni for user i
# b_g(λ) = (1 / (λ + ni)) * ∑(Yu,i − μ)   , with u = 1 to ni for genre i
# b_d(λ) = (1 / (λ + ni)) * ∑(Yu,i − μ)   , with u = 1 to ni for weekday_rated i
# y_r(λ) = (1 / (λ + ni)) * ∑(Yu,i − μ)   , with u = 1 to ni for year_released i
# y_a(λ) = (1 / (λ + ni)) * ∑(Yu,i − μ)   , with u = 1 to ni for year_rated i

# define a sequence of lambdas
lambdas <- seq(0, 100, 0.25)

# predict rating and RMSE on test_set and determine optimal lambda value
rmses <- sapply(lambdas, function(lambda){
  
  # calculate mu
  mu <- mean(train_set$rating)
  
  # calculate regularized movie b_i
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = (1/(lambda + n())) * sum(rating - mu))
  
  # calculate regularized user b_u
  b_u <- train_set %>%
    group_by(userId) %>%
    summarize(b_u = (1/(lambda + n())) * sum(rating - mu))
  
  # calculate regularized genres b_g
  b_g <- train_set %>%
    group_by(genres) %>%
    summarize(b_g = (1/(lambda + n())) * sum(rating - mu))
  
  # calculate regularized weekday_rated b_d
  b_d <- train_set %>%
    group_by(weekday_rated) %>%
    summarize(b_d = (1/(lambda + n())) * sum(rating - mu))
  
  # calculate regularized year_released y_r
  y_r <- train_set %>%
    group_by(year_released) %>%
    summarize(y_r = (1/(lambda + n())) * sum(rating - mu))
  
  # calculate regularized year_rated y_a
  y_a <- train_set %>%
    group_by(year_rated) %>%
    summarize(y_a = (1/(lambda + n())) * sum(rating - mu))
  
  # predict the ratings on test set
  predicted_ratings_test <- test_set %>%
    left_join(b_i, by='movieId') %>%
    left_join(b_u, by='userId') %>%
    left_join(b_g, by='genres') %>%
    left_join(b_d, by='weekday_rated') %>%
    left_join(y_r, by='year_released') %>%
    left_join(y_a, by='year_rated') %>%
    mutate(rating_hat = mu + b_i + b_u + b_g + b_d + y_r + y_a) %>%
    pull(rating_hat)
  
  # compute the RMSE
  return(RMSE(predicted_ratings_test, test_set$rating))
})


# plot the RMSEs and lambdas
qplot(lambdas, rmses) +
  ggtitle("Regularized Movie + User + Genres + Weekday_Rated + Year_Released + Year_Rated: Plot of lambdas vs. RMSEs")

# determine which lambda with least rmse
lambdas[which.min(rmses)]
# 50.0

# therefore optimal λ for the regularized movie + user + genres + weekday_rated model  =  50.0


# predict the ratings on the test_set at λ = 50.0
lambda <- 50.0
# calculate mu
mu <- mean(train_set$rating)

# calculate regularized movie b_i
b_i_reg <- test_set %>%
  group_by(movieId) %>%
  summarize(b_i = (1/(lambda + n())) * sum(rating - mu))

# calculate regularized user b_u
b_u_reg <- test_set %>%
  group_by(userId) %>%
  summarize(b_u = (1/(lambda + n())) * sum(rating - mu))

# calculate regularized genres b_g
b_g_reg <- test_set %>%
  group_by(genres) %>%
  summarize(b_g = (1/(lambda + n())) * sum(rating - mu))

# calculate regularized weekday_rated b_d
b_d_reg <- test_set %>%
  group_by(weekday_rated) %>%
  summarize(b_d = (1/(lambda + n())) * sum(rating - mu))

# calculate regularized year_released y_r
y_r_reg <- test_set %>%
  group_by(year_released) %>%
  summarize(y_r = (1/(lambda + n())) * sum(rating - mu))

# calculate regularized year_rated y_a
y_a_reg <- test_set %>%
  group_by(year_rated) %>%
  summarize(y_a = (1/(lambda + n())) * sum(rating - mu))

# predict the ratings on test set
predicted_ratings_test <- test_set %>%
  left_join(b_i_reg, by='movieId') %>%
  left_join(b_u_reg, by='userId') %>%
  left_join(b_g_reg, by='genres') %>%
  left_join(b_d_reg, by='weekday_rated') %>%
  left_join(y_r_reg, by='year_released') %>%
  left_join(y_a_reg, by='year_rated') %>%
  mutate(rating_hat = mu + b_i + b_u + b_g + b_d + y_r + y_a) %>%
  pull(rating_hat)

# predict RMSE on test_set
reg_movie_user_genres_day_yr_rated_released_model_rmse_test_set <-
  RMSE(test_set$rating, predicted_ratings_test)
# 0.8964442194


# predict the ratings on validation at λ = 50.0
lambda <- 50.0

# calculate mu
mu <- mean(train_set$rating)

# calculate regularized movie b_i
b_i_reg <- validation %>%
  group_by(movieId) %>%
  summarize(b_i = (1/(lambda + n())) * sum(rating - mu))

# calculate regularized user b_u
b_u_reg <- validation %>%
  group_by(userId) %>%
  summarize(b_u = (1/(lambda + n())) * sum(rating - mu))

# calculate regularized genres b_g
b_g_reg <- validation %>%
  group_by(genres) %>%
  summarize(b_g = (1/(lambda + n())) * sum(rating - mu))

# calculate regularized weekday_rated b_d
b_d_reg <- validation %>%
  group_by(weekday_rated) %>%
  summarize(b_d = (1/(lambda + n())) * sum(rating - mu))

# calculate regularized year_released y_r
y_r_reg <- validation %>%
  group_by(year_released) %>%
  summarize(y_r = (1/(lambda + n())) * sum(rating - mu))

# calculate regularized year_rated y_a
y_a_reg <- validation %>%
  group_by(year_rated) %>%
  summarize(y_a = (1/(lambda + n())) * sum(rating - mu))

# predict the ratings on validation
predicted_ratings_val_set <- validation %>%
  left_join(b_i_reg, by='movieId') %>%
  left_join(b_u_reg, by='userId') %>%
  left_join(b_g_reg, by='genres') %>%
  left_join(b_d_reg, by='weekday_rated') %>%
  left_join(y_r_reg, by='year_released') %>%
  left_join(y_a_reg, by='year_rated') %>%
  mutate(rating_hat = mu + b_i + b_u + b_g + b_d + y_r + y_a) %>%
  pull(rating_hat)

# predict RMSE on validation
reg_movie_user_genres_day_yr_rated_released_model_rmse_val_set <-
  RMSE(validation$rating, predicted_ratings_val_set)
# 0.8897930635

# add the results
results <-  results %>% add_row(method = "Regularized movie + user + genres + weekday_rated + year_released + year_rated model",
                                RMSE_test_set = reg_movie_user_genres_day_yr_rated_released_model_rmse_test_set,
                                RMSE_validation = reg_movie_user_genres_day_yr_rated_released_model_rmse_val_set)

results %>% knitr::kable()
#|method                                                                               | RMSE_test_set| RMSE_validation|
#|:------------------------------------------------------------------------------------|-------------:|---------------:|
#|Just the mean                                                                        |  1.0518982335|    1.0525571670|
#|Movie effect                                                                         |  0.9409529753|    0.9411804404|
#|Movie + user effect                                                                  |  0.8577063124|    0.8641412163|
#|Movie + user + genres effect                                                         |  0.8576231022|    0.8640504507|
#|Movie + user + genres + weekday_rated effect                                         |  0.8576227528|    0.8640488429|
#|Movie + user + genres + weekday_rated + year_released effect                         |  0.8572703631|    0.8636784216|
#|Movie + user + genres + weekday_rated + year_released + year_rated effect            |  0.8571846922|    0.8635987481|
#|Regularized movie model                                                              |  0.9399571015|    0.9369663126|
#|Regularized movie + user model                                                       |  0.8692830213|    0.8571358019|
#|Regularized movie + user + genres model                                              |  0.8788499509|    0.8684473123|
#|Regularized movie + user + genres + weekday_rated model                              |  0.8788909578|    0.8684984877|
#|Regularized movie + user + genres + weekday_rated + year_released model              |  0.8947160951|    0.8866161287|
#|Regularized movie + user + genres + weekday_rated + year_released + year_rated model |  0.8964442194|    0.8897930635|




####################

## 5.9 linear model implementation

# dependent var y = rating at column {4}
# independent var x= exclude title{3}, month_rated{7}, day_rated{8}, and wday_rated{10}
# include userId, movieId, genres, year_rated, weekday_rated, year_released

# Yu,i = rating = userId + movieId + genres + weekday_rated + year_released + year_rated

library(stats)

regression_stats_lm <- lm(rating ~ userId + movieId + genres + year_rated + weekday_rated + year_released,
                          data=train_set)



# predict the ratings on the test_set
predicted_ratings_test <- predict(regression_stats_lm, test_set)

# predict RMSE on test_set
stats_lm_model_rmse_test_set <- RMSE(test_set$rating, predicted_ratings_test)
# 1.037941151


# predict the ratings on validation set
predicted_ratings_val_set <- predict(regression_stats_lm, validation)

# predict RMSE on validation
stats_lm_model_rmse_val_set <- RMSE(validation$rating, predicted_ratings_val_set)
# 1.038492217

# add the results
results <-  results %>% data.frame(method = "stats linear regression (lm) method",
                                   RMSE_test_set = stats_lm_model_rmse_test_set,
                                   RMSE_validation = stats_lm_model_rmse_val_set)


results %>% knitr::kable()
#|method                                                                               | RMSE_test_set| RMSE_validation|
#|:------------------------------------------------------------------------------------|-------------:|---------------:|
#|Just the mean                                                                        |  1.0518982335|    1.0525571670|
#|Movie effect                                                                         |  0.9409529753|    0.9411804404|
#|Movie + user effect                                                                  |  0.8577063124|    0.8641412163|
#|Movie + user + genres effect                                                         |  0.8576231022|    0.8640504507|
#|Movie + user + genres + weekday_rated effect                                         |  0.8576227528|    0.8640488429|
#|Movie + user + genres + weekday_rated + year_released effect                         |  0.8572703631|    0.8636784216|
#|Movie + user + genres + weekday_rated + year_released + year_rated effect            |  0.8571846922|    0.8635987481|
#|Regularized movie model                                                              |  0.9399571015|    0.9369663126|
#|Regularized movie + user model                                                       |  0.8692830213|    0.8571358019|
#|Regularized movie + user + genres model                                              |  0.8788499509|    0.8684473123|
#|Regularized movie + user + genres + weekday_rated model                              |  0.8788909578|    0.8684984877|
#|Regularized movie + user + genres + weekday_rated + year_released model              |  0.8947160951|    0.8866161287|
#|Regularized movie + user + genres + weekday_rated + year_released + year_rated model |  0.8964442194|    0.8897930635|
#|stats linear regression (lm) method                                                  |  1.0379411513|    1.0384922169|



###################
#### 5.10 h2o random forest implementation

# Warning! this process runs a longer while

library(h2o)

#localH2O <- h2o.init(nthreads = -1)

h2o.init()

# transfer train_set, test_set and validation to h2o instance
train.h2o <- as.h2o(train_set)
test.h2o <- as.h2o(test_set)
validation.h2o <- as.h2o(validation)

# train the h2o RF model
# dependent var y = rating at column {4}
# independent var x= exclude title{3}, month_rated{7}, day_rated{8}, and wday_rated{10}
# include userId, movieId, genres, year_rated, weekday_rated, year_released
regression_rf_h2o <- h2o.randomForest(x = -c(3,7,8,10),
                                      y = 4,
                                      training_frame = train.h2o,
                                      seed=1,
                                      nfolds = 10,
                                      ntrees = 1000,
                                      mtries = 3,
                                      max_depth = 4)

h2o.performance(regression_rf_h2o)
#MSE:  1.059092533
#RMSE:  1.029122215
#MAE:  0.8278924097
#RMSLE:  0.2708952685
#Mean Residual Deviance :  1.059092533


# predict the ratings on the test_set
predicted_ratings_test <- as.data.frame(h2o.predict(regression_rf_h2o, test.h2o))

# predict RMSE on test_set
rf_model_rmse_test_set <- RMSE(test_set$rating, predicted_ratings_test$predict)
# 1.028565994


# predict the ratings on validation set
predicted_ratings_val_set <- as.data.frame(h2o.predict(regression_rf_h2o, validation.h2o))

# predict RMSE on validation
rf_model_rmse_val_set <- RMSE(validation$rating, predicted_ratings_val_set$predict)
# 1.029081025

# add the results
results <-  results %>% add_row(method = "h2o random forest model",
                                RMSE_test_set = rf_model_rmse_test_set,
                                RMSE_validation = rf_model_rmse_val_set)

results %>% knitr::kable()
#|method                                                                               | RMSE_test_set| RMSE_validation|
#|:------------------------------------------------------------------------------------|-------------:|---------------:|
#|Just the mean                                                                        |  1.0518982335|    1.0525571670|
#|Movie effect                                                                         |  0.9409529753|    0.9411804404|
#|Movie + user effect                                                                  |  0.8577063124|    0.8641412163|
#|Movie + user + genres effect                                                         |  0.8576231022|    0.8640504507|
#|Movie + user + genres + weekday_rated effect                                         |  0.8576227528|    0.8640488429|
#|Movie + user + genres + weekday_rated + year_released effect                         |  0.8572703631|    0.8636784216|
#|Movie + user + genres + weekday_rated + year_released + year_rated effect            |  0.8571846922|    0.8635987481|
#|Regularized movie model                                                              |  0.9399571015|    0.9369663126|
#|Regularized movie + user model                                                       |  0.8692830213|    0.8571358019|
#|Regularized movie + user + genres model                                              |  0.8788499509|    0.8684473123|
#|Regularized movie + user + genres + weekday_rated model                              |  0.8788909578|    0.8684984877|
#|Regularized movie + user + genres + weekday_rated + year_released model              |  0.8947160951|    0.8866161287|
#|Regularized movie + user + genres + weekday_rated + year_released + year_rated model |  0.8964442194|    0.8897930635|
#|stats linear regression (lm) method                                                  |  1.0379411513|    1.0384922169|
#|h2o random forest model                                                              |  1.0285659940|    1.0290810245|


#h2o.shutdown()


######################
#### 5.11 h2o generalized linear model (glm) implementation

# Warning! this process runs a longer while

#library(h2o)

#localH2O <- h2o.init(nthreads = -1)

# h2o.init()

# transfer train_set, test_set and validation to h2o instance
#train.h2o <- as.h2o(train_set)
#test.h2o <- as.h2o(test_set)
#validation.h2o <- as.h2o(validation)


# train the h2o glm model
# dependent var y = rating at column {4}
# independent var x= exclude title{3}, month_rated{7}, day_rated{8}, and wday_rated{10}
# include userId, movieId, genres, year_rated, weekday_rated, year_released
regression_glm_h2o <- h2o.glm(x = -c(3,7,8,10),
                              y = 4,
                              training_frame = train.h2o,
                              seed=1,
                              nfolds = 10,
                              family="gaussian")

h2o.performance(regression_glm_h2o)
#H2ORegressionMetrics: glm
#MSE:  1.078407736
#RMSE:  1.038464124
#MAE:  0.8380170445
#RMSLE:  0.2726546496

# predict the ratings on the test_set
predicted_ratings_test <- as.data.frame(h2o.predict(regression_glm_h2o, test.h2o))

# predict RMSE on test_set
glm_model_rmse_test_set <- RMSE(test_set$rating, predicted_ratings_test$predict)
# 1.037962952


# predict the ratings on validation set
predicted_ratings_val_set <- as.data.frame(h2o.predict(regression_glm_h2o, validation.h2o))

# predict RMSE on validation
glm_model_rmse_val_set <- RMSE(validation$rating, predicted_ratings_val_set$predict)
# 1.038513907

# add the results
results <-  results %>% add_row(method = "h2o glm model",
                                RMSE_test_set = glm_model_rmse_test_set,
                                RMSE_validation = glm_model_rmse_val_set)

results %>% knitr::kable()
#|method                                                                               | RMSE_test_set| RMSE_validation|
#|:------------------------------------------------------------------------------------|-------------:|---------------:|
#|Just the mean                                                                        |  1.0518982335|    1.0525571670|
#|Movie effect                                                                         |  0.9409529753|    0.9411804404|
#|Movie + user effect                                                                  |  0.8577063124|    0.8641412163|
#|Movie + user + genres effect                                                         |  0.8576231022|    0.8640504507|
#|Movie + user + genres + weekday_rated effect                                         |  0.8576227528|    0.8640488429|
#|Movie + user + genres + weekday_rated + year_released effect                         |  0.8572703631|    0.8636784216|
#|Movie + user + genres + weekday_rated + year_released + year_rated effect            |  0.8571846922|    0.8635987481|
#|Regularized movie model                                                              |  0.9399571015|    0.9369663126|
#|Regularized movie + user model                                                       |  0.8692830213|    0.8571358019|
#|Regularized movie + user + genres model                                              |  0.8788499509|    0.8684473123|
#|Regularized movie + user + genres + weekday_rated model                              |  0.8788909578|    0.8684984877|
#|Regularized movie + user + genres + weekday_rated + year_released model              |  0.8947160951|    0.8866161287|
#|Regularized movie + user + genres + weekday_rated + year_released + year_rated model |  0.8964442194|    0.8897930635|
#|stats linear regression (lm) method                                                  |  1.0379411513|    1.0384922169|
#|h2o random forest model                                                              |  1.0285659940|    1.0290810245|
#|h2o glm model                                                                        |  1.0379629524|    1.0385139067|

#h2o.shutdown()



#####################
#### 5.12 h2o deep neural network implementation

# Warning! this process runs a longer while

#library(h2o)

#localH2O <- h2o.init(nthreads = -1)

# h2o.init()

# transfer train_set, test_set and validation to h2o instance
#train.h2o <- as.h2o(train_set)
#test.h2o <- as.h2o(test_set)
#validation.h2o <- as.h2o(validation)


# train the h2o deeplearning model
# dependent var y = rating at column {4}
# independent var x= exclude title{3}, month_rated{7}, day_rated{8}, and wday_rated{10}
# include userId, movieId, genres, year_rated, weekday_rated, year_released
regression_deeplearn_h2o <- h2o.deeplearning(x = -c(3,7,8,10),
                                             y = 4,
                                             training_frame = train.h2o,
                                             seed=1,
                                             nfolds = 10,
                                             epochs=10,
                                             hidden = c(7,3),
                                             activation="Rectifier",
                                             overwrite_with_best_model=TRUE,
                                             use_all_factor_levels = TRUE,
                                             variable_importances = TRUE,
                                             export_weights_and_biases = TRUE,
                                             verbose=TRUE)

h2o.performance(regression_deeplearn_h2o)
#H2ORegressionMetrics: deeplearning
#MSE:  1.097316472
#RMSE:  1.047528745
#MAE:  0.8379924672
#RMSLE:  0.2782778684
#Mean Residual Deviance :  1.097316472


# predict the ratings on the test_set
predicted_ratings_test <- as.data.frame(h2o.predict(regression_deeplearn_h2o, test.h2o))

# predict RMSE on test_set
deeplearn_model_rmse_test_set <- RMSE(test_set$rating, predicted_ratings_test$predict)
# 1.028453286


# predict the ratings on validation set
predicted_ratings_val_set <- as.data.frame(h2o.predict(regression_deeplearn_h2o, validation.h2o))

# predict RMSE on validation
deeplearn_model_rmse_val_set <- RMSE(validation$rating, predicted_ratings_val_set$predict)
# 1.029167872

# add the results
results <-  results %>% add_row(method = "h2o deep learning: (7,3) hidden layers",
                                RMSE_test_set = deeplearn_model_rmse_test_set,
                                RMSE_validation = deeplearn_model_rmse_val_set)

results %>% knitr::kable()
#|method                                                                               | RMSE_test_set| RMSE_validation|
#|:------------------------------------------------------------------------------------|-------------:|---------------:|
#|Just the mean                                                                        |  1.0518982335|    1.0525571670|
#|Movie effect                                                                         |  0.9409529753|    0.9411804404|
#|Movie + user effect                                                                  |  0.8577063124|    0.8641412163|
#|Movie + user + genres effect                                                         |  0.8576231022|    0.8640504507|
#|Movie + user + genres + weekday_rated effect                                         |  0.8576227528|    0.8640488429|
#|Movie + user + genres + weekday_rated + year_released effect                         |  0.8572703631|    0.8636784216|
#|Movie + user + genres + weekday_rated + year_released + year_rated effect            |  0.8571846922|    0.8635987481|
#|Regularized movie model                                                              |  0.9399571015|    0.9369663126|
#|Regularized movie + user model                                                       |  0.8692830213|    0.8571358019|
#|Regularized movie + user + genres model                                              |  0.8788499509|    0.8684473123|
#|Regularized movie + user + genres + weekday_rated model                              |  0.8788909578|    0.8684984877|
#|Regularized movie + user + genres + weekday_rated + year_released model              |  0.8947160951|    0.8866161287|
#|Regularized movie + user + genres + weekday_rated + year_released + year_rated model |  0.8964442194|    0.8897930635|
#|stats linear regression (lm) method                                                  |  1.0379411513|    1.0384922169|
#|h2o random forest model                                                              |  1.0285659940|    1.0290810245|
#|h2o glm model                                                                        |  1.0379629524|    1.0385139067|
#|h2o deep learning: (7,3) hidden layers                                               |  1.0284532860|    1.0291678719|

#h2o.shutdown()


#####################
#### 5.13 h2o gradient boosting machine (gbm) implementation

# Warning! this process runs a longer while

#library(h2o)

#localH2O <- h2o.init(nthreads = -1)

# h2o.init()

# transfer train_set, test_set and validation to h2o instance
#train.h2o <- as.h2o(train_set)
#test.h2o <- as.h2o(test_set)
#validation.h2o <- as.h2o(validation)


# train the h2o gbm model
# dependent var y = rating at column {4}
# independent var x= exclude title{3}, month_rated{7}, day_rated{8}, and wday_rated{10}
# include userId, movieId, genres, year_rated, weekday_rated, year_released
regression_gbm_h2o <- h2o.gbm(x = -c(3,7,8,10),
                              y = 4,
                              training_frame = train.h2o,
                              seed=1,
                              nfolds = 10,
                              distribution="gaussian",
                              stopping_metric = "RMSE",
                              categorical_encoding = "AUTO",
                              verbose=TRUE)

h2o.performance(regression_gbm_h2o)
#H2ORegressionMetrics: gbm
#MSE:  1.019485802
#RMSE:  1.009695896
#MAE:  0.8071450823
#RMSLE:  0.2664681261
#Mean Residual Deviance :  1.019485802


# predict the ratings on the test_set
predicted_ratings_test <- as.data.frame(h2o.predict(regression_gbm_h2o, test.h2o))

# predict RMSE on test_set
gbm_model_rmse_test_set <- RMSE(test_set$rating, predicted_ratings_test$predict)
#  1.009190132


# predict the ratings on validation set
predicted_ratings_val_set <- as.data.frame(h2o.predict(regression_gbm_h2o, validation.h2o))

# predict RMSE on validation
gbm_model_rmse_val_set <- RMSE(validation$rating, predicted_ratings_val_set$predict)
# 1.009508203

# add the results
results <-  results %>% add_row(method = "h2o gradient boosting machine (gbm)",
                                RMSE_test_set = gbm_model_rmse_test_set,
                                RMSE_validation = gbm_model_rmse_val_set)

results %>% knitr::kable()
#|method                                                                               | RMSE_test_set| RMSE_validation|
#|:------------------------------------------------------------------------------------|-------------:|---------------:|
#|Just the mean                                                                        |  1.0518982335|    1.0525571670|
#|Movie effect                                                                         |  0.9409529753|    0.9411804404|
#|Movie + user effect                                                                  |  0.8577063124|    0.8641412163|
#|Movie + user + genres effect                                                         |  0.8576231022|    0.8640504507|
#|Movie + user + genres + weekday_rated effect                                         |  0.8576227528|    0.8640488429|
#|Movie + user + genres + weekday_rated + year_released effect                         |  0.8572703631|    0.8636784216|
#|Movie + user + genres + weekday_rated + year_released + year_rated effect            |  0.8571846922|    0.8635987481|
#|Regularized movie model                                                              |  0.9399571015|    0.9369663126|
#|Regularized movie + user model                                                       |  0.8692830213|    0.8571358019|
#|Regularized movie + user + genres model                                              |  0.8788499509|    0.8684473123|
#|Regularized movie + user + genres + weekday_rated model                              |  0.8788909578|    0.8684984877|
#|Regularized movie + user + genres + weekday_rated + year_released model              |  0.8947160951|    0.8866161287|
#|Regularized movie + user + genres + weekday_rated + year_released + year_rated model |  0.8964442194|    0.8897930635|
#|stats linear regression (lm) method                                                  |  1.0379411513|    1.0384922169|
#|h2o random forest model                                                              |  1.0285659940|    1.0290810245|
#|h2o glm model                                                                        |  1.0379629524|    1.0385139067|
#|h2o deep learning: (7,3) hidden layers                                               |  1.0284532860|    1.0291678719|
#|h2o gradient boosting machine (gbm)                                                  |  1.0091901316|    1.0095082035|


h2o.shutdown()


#################################

## Grade determination based of reported RMSE

# function to determine grade from RMSE on validation
grade_rmse <- function(rmse){
  if(is.na(rmse) == TRUE){
    grade <- 0.0
  }else if(rmse > 0.9000000000){
    grade <- 5.0
  }else if(rmse >= 0.8655000000 & rmse <= 0.899990000){
    grade <- 10.0
  }else if(rmse >= 0.8650000000 & rmse <= 0.865490000){
    grade <- 15.0
  }else if(rmse >= 0.8649000000 & rmse <= 0.864990000) {
    grade <- 20.0
  }else if(rmse < 0.8649000000){
    grade <- 25.0
  }
  grade
}

results %>% select(method, RMSE_validation) %>%
  mutate(grade = sapply(RMSE_validation, grade_rmse)) %>%
  knitr::kable()


#################################

# Based of the above results, it is evident that the 'Regularized movie + user' model provides the
# least RMSE of 0.8571358019 on the validation set. This means that the variables movieId and userId
# are sufficient to predict the ratings of movies with the least acceptable RMSE.
