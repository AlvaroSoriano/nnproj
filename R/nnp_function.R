library(NeuralNetTools)
library(neuralnet)
library(pracma)
library(nnet)
library(class)

#' A supervised projection function
#'
#' This function transforms a dataset into the activations of the neurons of the hidden layer of a neural network. This is done by training a neural network and then computing the activations of the neural network for each input pattern
#' @param x is a data.frame that contains the inputs of the dataset
#' @param y is a vector that contains the outpus of the dataset
#' @param hidden is a vector that contains the number of neurons for each hidden layer
#' @param rep is the number of iterations used for training the neural network
#' @export
#' @examples
#' data <- data.frame(A=runif(10, 5.0, 7.5), B=runif(10, 1.0, 3.5), C=runif(10, 10.0, 10.5), output=runif(10,1.0, 10.0))
#' projection <- nnp(data[-ncol(data)], datos[[ncol(data)]], hidden=2, rep=10)


nnp2 <- function(x, y, hidden, steps){
  data <- as.data.frame(cbind(as.matrix(x), output=y))
  
  formula <- paste0("output ~ ", paste0(names(x),  collapse="+"))
  
  mod <- neuralnet(as.formula(formula), data=data, hidden=hidden, err.fct = 'ce', linear.output = FALSE, rep=1, lifesign=3)
  wts <- neuralweights(mod)
  struct <- wts$struct
  
  matrix1 <- as.matrix(as.data.frame(wts$wts[1:struct[2]]))
  matrix2 <- as.matrix(as.data.frame(wts$wts[(1+struct[2]):length(wts$wts)]))
  
  #computed_output <- sigmoid(cbind(1,sigmoid(cbind(1,x) %*% matrix1)) %*% matrix2)
  #net_output <- compute(mod, x)
  
  return(function(x) sigmoid(cbind(1,as.matrix(x)) %*% matrix1) )
}


#' A supervised projection function
#'
#' This function transforms a dataset into the activations of the neurons of the hidden layer of a neural network. This is done by training a neural network and then computing the activations of the neural network for each input pattern
#' @param x is a data.frame that contains the inputs of the dataset
#' @param y is a vector that contains the outpus of the dataset
#' @param hidden is a vector that contains the number of neurons for each hidden layer
#' @param steps is the number of iterations of the net
#' @export
#' @examples
#' nnp_example()

nnp <- function(x, y, hidden, steps){
  
  mod <- nnet(x, y, size=hidden, maxit=steps, trace=FALSE)
  wts <- neuralweights(mod)
  struct <- wts$struct
  
  matrix1 <- as.matrix(as.data.frame(wts$wts[1:struct[2]]))
  matrix2 <- as.matrix(as.data.frame(wts$wts[(1+struct[2]):length(wts$wts)]))
  
  #computed_output <- sigmoid(cbind(1,sigmoid(cbind(1,x) %*% matrix1)) %*% matrix2)
  #net_output <- compute(mod, x)
  
  output <- list(projection = function(x) (cbind(1,as.matrix(x)) %*% matrix1),
                 mod = mod,
                 matrix1 = matrix1,
                 matrix2 = matrix2)
  
  return( output )
}

nnp_example <- function(){
  
  library(class)
  
  set.seed(1)
  # Doughnougt data
  data("doughnut")
  sampled <- sample(nrow(doughnut), n<-nrow(doughnut))
  #sampled <- 1:(n<-nrow(doughnout))
  
  training_index <- sample(1:n, round(0.6*n))
  
  rd <- doughnut[sampled,]
  train <- rd[training_index,]
  test <- rd[-training_index,]
  
  cl <- factor(train[[ncol(train)]])
  outputs <- knn1(train[-ncol(train)], test[-ncol(test)], cl)
  success <- sum(outputs == test[[ncol(test)]])/nrow(test)
  print(success)
  
  # Doughnougt with random attributes data
  data("doughnutRand")
  rd <- doughnutRand[sampled,]
  
  train <- rd[training_index,]
  test <- rd[-training_index,]
  
  cl <- factor(train[[ncol(train)]])
  outputs <- knn1(train[-ncol(train)], test[-ncol(test)], cl)
  success <- sum(outputs == test[[ncol(test)]])/nrow(test)
  print(success)
  
  # Doughnougt with random attributes and rotated data
  data("doughnutRandRotated")
  rd <- doughnutRandRotated[sampled,]
  
  train <- rd[training_index,]
  test <- rd[-training_index,]
  
  cl <- factor(train[[ncol(train)]])
  outputs <- knn1(train[-ncol(train)], test[-ncol(test)], cl)
  success <- mean(outputs == test[[ncol(test)]])
  print(success)
  
  
  
  fichero <- "repeticionesSigmoide.txt"
  cat("Transformation Hidden Iterations Seed knn1test\n", file=fichero)
  
  for(hidden in 1:(ncol(train))){
    # hidden <- 10
    for(steps in c(10, 100, 1000, 10000)){
      for(repetition in 1:50){
        set.seed(repetition)
        
        nnpo <- nnp(train[-ncol(train)], train[[ncol(train)]], hidden=c(hidden), steps=steps)
        projection <- nnpo$projection
        mod <- nnpo$mod
        
        outputs <- predict(mod, test[-ncol(test)])
        
        errorp <- mean((outputs-test[[ncol(test)]])^2)
        # meanTrain <- mean(train[[ncol(train)]])
        # variance <- mean((meanTrain-test[[ncol(test)]])^2)
        # R2 <- 1- errorp/variance
        
        cat(paste0("nnetc ", hidden, " ", steps, " ", repetition, " ", errorp, "\n"), file=fichero, append=TRUE)
        cat(paste0("nnetc ", hidden, " ", steps, " ", repetition, " ", errorp, "\n"))
        
        outputp <- ifelse(outputs>=0.5, TRUE, FALSE)
        successp <- mean(outputsp == test[[ncol(test)]])
        
        cat(paste0("nnet ", hidden, " ", steps, " ", repetition, " ", successp, "\n"), file=fichero, append=TRUE)
        cat(paste0("nnet ", hidden, " ", steps, " ", repetition, " ", successp, "\n"))
        
        
        projected_train <-  projection(train[-ncol(train)])
        projected_test <- projection(test[-ncol(test)])
        # plot(projected_test, col=cl)
        outputsp <- knn1(projected_train, projected_test, cl)
        successp <- mean(outputsp == test[[ncol(test)]])
        
        cat(paste0("Linear ", hidden, " ", steps, " ", repetition, " ", successp, "\n"), file=fichero, append=TRUE)
        cat(paste0("Linear ", hidden, " ", steps, " ", repetition, " ", successp, "\n"))
        
        
        projected_train <-  sigmoid(projected_train)
        projected_test <- sigmoid(projected_test)
        # plot(projected_test, col=cl)
        outputsp <- knn1(projected_train, projected_test, cl)
        successp <- mean(outputsp == test[[ncol(test)]])
        
        cat(paste0("Sigmoid ", hidden, " ", steps, " ", repetition, " ", successp, "\n"), file=fichero, append=TRUE)
        cat(paste0("Sigmoid ", hidden, " ", steps, " ", repetition, " ", successp, "\n"))
        
      }
    }
  }
}

iteratePlot <- function(x, y, hidden, steps){
  
  set.seed(3)
  fichero <- "results_seed3_2hn.txt"
  # fichero <- "results_seed1.txt"
  jump <- 20
  cat("Iteration NNtrain NNtest knntestlineal knntestsigmoid\n", file=fichero )
  
  
  hidden <- 2
  x <- train[-ncol(train)]
  y <- train[[ncol(train)]]
  
  mod <- nnet(x, y, size=hidden, maxit=1, trace=TRUE)
  
  wts <- neuralweights(mod)
  struct <- wts$struct
  matrix1 <- as.matrix(as.data.frame(wts$wts[1:struct[2]]))
  matrix2 <- as.matrix(as.data.frame(wts$wts[(1+struct[2]):length(wts$wts)]))
  projection <- function(x) (cbind(1,as.matrix(x)) %*% matrix1)
  projected_train <-  projection(train[-ncol(train)])
  plot(projected_train[,2] ~ projected_train[,1], col=as.factor(y))
  
  #  library(far)
  #  rp=orthonormalization(matrix(rnorm(10*10),10,10),norm=TRUE)
  
  #  matrix1 <- orthonormalization(matrix1, basis=FALSE, norm=FALSE)
  #  projection <- function(x) (cbind(1,as.matrix(x)) %*% matrix1)
  #  projected_train <-  projection(train[-ncol(train)])
  #  plot(projected_train[,2] ~ projected_train[,1], col=as.factor(y))
  
  #  wts$wts[1:struct[2]] <- split(matrix1, col(matrix1))
  
  
  
  for(i in 1:10000){
    
    mod <- nnet(x, y, size=hidden, maxit=jump, trace=FALSE, Wts=mod$wts)
    
    output_nn <- ifelse(predict(mod, train[-ncol(train)])>=0.5, TRUE, FALSE)
    success_nn_tr <- mean(output_nn == train[[ncol(train)]])
    
    output_nn <- ifelse(predict(mod, test[-ncol(test)])>=0.5, TRUE, FALSE)
    success_nn_te <- mean(output_nn == test[[ncol(test)]])
    
    
    wts <- neuralweights(mod)
    struct <- wts$struct
    
    matrix1 <- as.matrix(as.data.frame(wts$wts[1:struct[2]]))
    matrix2 <- as.matrix(as.data.frame(wts$wts[(1+struct[2]):length(wts$wts)]))
    
    projection <- function(x) (cbind(1,as.matrix(x)) %*% matrix1)
    
    
    
    par(mfrow=c(2,2))
    
    projected_train <-  projection(train[-ncol(train)])
    projected_test <-  projection(test[-ncol(test)])
    
    outputsp <- knn1(projected_train, projected_test, cl)
    successp_k_l <- mean(outputsp == test[[ncol(test)]])
    
    plot(projected_train[,2] ~ projected_train[,1], col=as.factor(y))
    #plot(projected_train[,3] ~ projected_train[,2], col=as.factor(y))
    #plot(projected_train[,4] ~ projected_train[,3], col=as.factor(y))
    #plot(projected_train[,5] ~ projected_train[,4], col=as.factor(y))
    
    projected_train <-  sigmoid(projected_train)
    projected_test <-  sigmoid(projected_test)
    
    outputsp <- knn1(projected_train, projected_test, cl)
    successp_k_s <- mean(outputsp == test[[ncol(test)]])
    
    
    cat(paste0(jump*i, " ", success_nn_tr, " ", success_nn_te, " ", successp_k_l, " ", successp_k_s, "\n"), file=fichero, append = TRUE )
    cat(paste0(jump*i, " ", success_nn_tr, " ", success_nn_te, " ", successp_k_l, " ", successp_k_s, "\n"))
    
    # plot(projected_train[,2] ~ projected_train[,1], col=as.factor(y))
    
    
    Sys.sleep(0.1)
  }
  
  #computed_output <- sigmoid(cbind(1,sigmoid(cbind(1,x) %*% matrix1)) %*% matrix2)
  #net_output <- compute(mod, x)
  
  
}

