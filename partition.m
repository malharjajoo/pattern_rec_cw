function [train_set,validation_set] = partition(X,ratio)
%random indices used to select trainig and validation 
% datasets from input dataset, according to ratio ( training:validation)
% passed in.


%generate random, non-repeatable indices 
total_rows = size(X,1);
training_indices = randperm(total_rows,round(ratio*total_rows));
validation_indices = setdiff(1:total_rows,training_indices);

% random indices are used to select training set 
% remaining indices ( All indices - random indices ) are used to
% select the validation set.
train_set = X(training_indices,:);
validation_set = X(validation_indices,:);
