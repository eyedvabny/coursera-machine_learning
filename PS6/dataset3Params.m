function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% Possible choices
opts = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

% temp storage matrix
params = zeros(size(opts,2)^2,3);

% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

index = 1;
for C = 1:size(opts,2)
  for sigma = 1:size(opts,2)
    params(index,1) = opts(C);
    params(index,2) = opts(sigma);
    
    model= svmTrain(X, y, opts(C), @(x1, x2) gaussianKernel(x1, x2, opts(sigma)));
    error = mean(double(svmPredict(model, Xval) ~= yval));
    
    params(index,3) = error;
    
    index = index + 1;
  end
end

% Find the optimal selection
[v,i] = min(params(:,3));

C = params(i,1);
sigma = params(i,2);
end
