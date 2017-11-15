function [train_set,validation_set] = partition(X,ratio)
%random indices used to select trainig and validation 
% datasets from input dataset, according to ratio ( training:validation)
% passed in.


%generate random, non-repeatable indices 
total_samples = size(X,2); % 520
training_indices = randperm(total_samples,round(ratio*total_samples));
validation_indices = setdiff(1:total_samples,training_indices);

% random indices are used to select training set 
% remaining indices ( All indices - random indices ) are used to
% select the validation set.
train_set = X(:,training_indices);
validation_set = X(:,validation_indices);
