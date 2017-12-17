function [train_set,train_labels,test_set,test_labels] = partition(X,l,ratio)

% random indices used to select trainig and validation 
% datasets from input dataset, according to ratio 
% passed in.

% generate random, non-repeatable indices 
N = size(X,2); % 0.8 * 520 = 416 in our case

training_indices = randperm(N,round(ratio*N));
test_indices = setdiff(1:N,training_indices);
 
% random indices are used to select training set 
% remaining indices ( All indices - random indices ) are used to
% select the validation set.
train_set = X(:,training_indices);
train_labels = l(:,training_indices);

test_set = X(:,test_indices);
test_labels = l(:,test_indices);