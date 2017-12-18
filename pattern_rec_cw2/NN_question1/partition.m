function [train_set,train_labels,test_set,test_labels] = partition(filename)
% Read input csv data file
data = dlmread(filename);


train_idx = find(data(:,1)==1);
test_idx = find(data(:,1)==2);

% extract training/test and sample/label into X and Y.
X = data(train_idx,:);
Y = data(test_idx,:);

% remove 1st coloumn since it is not required.
% 2nd coloumn contains labels
train_set = X(:,3:end);
test_set = Y(:,3:end);

train_labels = X(:,2);
test_labels = Y(:,2);

end
