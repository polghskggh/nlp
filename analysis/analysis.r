library(dplyr)
library(ggplot2)
library(ggpubr)
library(superml)

data <- read.csv("../data/train.csv", sep=";")
ggplot(data, aes(x = emotion, fill = emotion)) + geom_bar() +
       theme(legend.position = "none")

encoder <- LabelEncoder$new()
encoder$fit(data$emotion)



metrics <- read.csv("../src/metrics.csv", sep=",")
metrics$epoch <- c(0:11)
dummy <- read.csv("../src/dummy.csv", sep=",")
dummy$epoch <- c(0:10)
  
f1 <- ggplot(head(metrics, 11),aes(x = epoch, y = f1)) + geom_line(color = "red") + 
  scale_x_continuous(breaks = 0:10) + labs(y = "F1 score") +
  scale_y_continuous(limits = c(0,0.7)) + 
  theme(axis.title.x=element_blank(), axis.text.x=element_blank()) +
  theme(axis.text.y=element_blank()) + geom_line(data=dummy, aes(x = epoch, y=f1), color= "blue")



acc <- ggplot(head(metrics, 11),aes(x = epoch, y = accuracy)) + geom_line(color = "red") + 
  scale_x_continuous(breaks = 0:10) + labs(y = "Accuracy") +
  scale_y_continuous(limits = c(0,0.7)) +
  theme(axis.title.x=element_blank(), axis.text.x=element_blank())


prec <- ggplot(head(metrics, 11),aes(x = epoch, y = precision)) + geom_line(color = "red") + 
  scale_x_continuous(breaks = 0:10) + labs(y = "Precision") +
  scale_y_continuous(limits = c(0,0.7))


rec <- ggplot(head(metrics, 11),aes(x = epoch, y = recall)) + geom_line(color = "red") + 
  scale_x_continuous(breaks = 0:10) + labs(y = "Recall") +
  scale_y_continuous(limits = c(0,0.7)) +
  theme(axis.text.y=element_blank())


ggarrange(acc, f1, prec, rec, nrow = 2, ncol = 2)



confusion <- read.csv("../src/confusion.csv", sep=",")
confusion <- data.frame(emotion1 = rep(encoder$inverse_transform(0:6), each = 7), 
                        emotion2 = rep(encoder$inverse_transform(0:6), times = 7), 
                        value = c(confusion$X0, confusion$X1, confusion$X2, 
                                  confusion$X3, confusion$X4, confusion$X5,
                                  confusion$X6))

ggplot(confusion, aes(x = emotion1, y = emotion2, fill = value)) + geom_tile() +  
  geom_text(aes(label = value), color = "white", size = 4) +
  theme(axis.title.y=element_blank()) +
  theme(axis.title.y=element_blank())



