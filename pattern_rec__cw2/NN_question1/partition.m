function [train_set,train_label,test_set,test_label] = partition(filename)
% Read input csv data file
data = dlmread(filename);


% random indices are used to select training set 
% remaining indices ( All indices - random indices ) are used to
% select the validation set.
train_idx = find(data(:,1)==1);
test_idx = find(data(:,1)==2);

X = data(train_idx,:);
Y = data(test_idx,:);

% remove 1st coloumn since it is not required.
% 2nd coloumn contains labels
train_set = X(:,3:end);
test_set = Y(:,3:end);

train_label = X(:,2);
test_label = Y(:,2);

end