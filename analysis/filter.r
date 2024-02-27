library(readr)
library(dplyr)
library(superml)


data_t <- read.csv("../data/train.csv", sep=";")
# create a label encoder object
encoder <- LabelEncoder$new()

# fitting the data over the x vector
encoder$fit(data_t$emotion)


filter <- function(filename)
{
    data <- read.csv(paste("../data/", filename, ".csv", sep=""), sep=";")
    data$emotion <- encoder$transform(data$emotion)

    data <- data %>% select(essay, emotion) %>% rename(text = essay, labels = emotion)

    write.table(data, file = paste("../data/", filename, "-clean.csv", sep=""), sep = ";",
                quote = FALSE, row.names = FALSE)
}




filter("test")
filter("train")


