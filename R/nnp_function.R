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