library(readr)
library(dplyr)
library(caTools)
library(superml)

data_t <- read.csv("../data/train.csv", sep=";")
# create a label encoder object
encoder <- LabelEncoder$new()

# fitting the data over the x vector
encoder$fit(data_t$emotion)


filter <- function(filename, split)
{
    data <- read.csv(paste("../data/", filename, ".csv", sep=""), sep=";")
    data$emotion <- encoder$transform(data$emotion)

    data <- data %>% select(essay, emotion) %>% rename(text = essay, labels = emotion)

    if (split < 1)
    {
        mask <- sample.split(data[,1], split)
        write.table(data[!mask, 1:2], file = paste("../data/val-clean.csv", sep=""), sep = "\t",
                    quote = FALSE, row.names = FALSE)
        data <- data[mask, 1:2]
    }

    write.table(data, file = paste("../data/", filename, "-clean.csv", sep=""), sep = "\t",
                quote = FALSE, row.names = FALSE)
}

filter("test", 1)
filter("train", 0.7)
