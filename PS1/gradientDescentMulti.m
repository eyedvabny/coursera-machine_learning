function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
n = size(X,2); % number of features

J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    h_x = X * theta;
    
    coef = zeros(n,1);
    coef(1) = sum(h_x - y) / m;
    coef(2) = sum((h_x - y).*X(:,2)) / m;
    
    for i=1:n
        coef(i) = sum((h_x - y).*X(:,i)) / m;
    end
    theta = theta - coef * alpha;
    
    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
