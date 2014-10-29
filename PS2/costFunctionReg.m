function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

fit_fun = sigmoid(X*theta);

J = mean(-y.*log(fit_fun)-(1-y).*log(1-fit_fun)) + (theta(2:end)'*theta(2:end))*lambda/(2*m);

grad = X' * (fit_fun-y) / m +  [0;lambda/m*theta(2:end)];

end
