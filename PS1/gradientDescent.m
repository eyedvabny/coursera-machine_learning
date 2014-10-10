function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    
    % [m 1] = [m 2]*[2 1]
    h_x = X * theta;
    
    % [2 1] = [2 1] - [1 1] * (([m 1] - [m 1]).* [m 2]) / [1 1]
    
    coef = zeros(2,1);
    coef(1) = sum(h_x - y) / m;
    coef(2) = sum((h_x - y).*X(:,2)) / m;
    
    theta = theta - coef * alpha;

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
