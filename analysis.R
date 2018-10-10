#Brian Kilburn
#CSC 546 AI
#Vehicle Calculator data analysis

#load the data
data <- read.csv('CARS.csv')

#print first three lines
head(data, n= 3L)
#print last three lines
tail(data, n= 3L)

#check for missing data
sapply(data, function(x) sum(is.na(x)))
#how many unique values
sapply(data, function(x) length(unique(x)))

#visualize missing values if any
library(Amelia)
missmap(data, main = "Missing Values vs Observed")

#replace any missing values with the average
data$Cylinders[is.na(data$Cylinders)] <- mean(data$Cylinders, na.rm=T)

#encode categorical variables as factors
is.factor(data$MSRP)

#better visualize how factor is used
contrasts(data$MSRP)