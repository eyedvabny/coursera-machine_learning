function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%
pos_x = X(y==1,:);
neg_x = X(y==0,:);

plot(pos_x(:,1),pos_x(:,2),'k+','MarkerFaceColor','k');
plot(neg_x(:,1),neg_x(:,2),'ko','MarkerFaceColor','y');

hold off;

end
