% This script checks performance of a neural network object on a test set.
% It requires the wine.data.csv file to be in a folder "data" in parent
% directory of this file.

load('five_best_networks.mat');
net = five_best_Params{1}{5}; 

% Kindly place the wine.data.csv in a "data" folder outside this folder 
% or change the file path below.
[train_set,train_labels,test_set,test_labels] = partition('../data/wine.data.csv');

warning off; % this is required to prevent trainbr from issuing warning about changing reg param.

% Check performance on Test set
y1 = net(test_set);

% Please Note- At this point if a debugger is used to 
% check values of y against test_labels, they may "appear" 
% different. More details in displayPerformance() below.
displayPerformance(net,test_labels,y1,'true');

% Check performance on Trainign set
y2 = net(train_set);
displayPerformance(net,train_labels,y2,'true');

% view(net)  ; % View the Network


% =============== Helper function ===============


function [] = displayPerformance(net,t,y,displayThis)
    
    performance = perform(net,t,y);
    
    % On running "nprtool", and generating a "Simple Script" from that
    % The matlab Neural Network tool box provides the method 
    % below for finding accuracy. 
    
    % On a closer look at each coloumn of y,
    % the index of the maximum value in each row is the target.is the same as the test labels.
    tind = vec2ind(t);
    yind = vec2ind(y);
    accuracy = (sum(tind == yind)/numel(tind)) * 100 ;
    
    if(strcmp(displayThis,'true'))
        fprintf('(metric=val,accuracy)= (%s=%f,%f %%)\n',net.performFcn, performance,accuracy);
    end
    
end



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
train_set = X(:,3:end)';
test_set = Y(:,3:end)';

% this is done for the form that neural net gives output.
% more convenient to keep consistency.
train_labels = ind2vec(X(:,2)');
test_labels = ind2vec(Y(:,2)');

end

