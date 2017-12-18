function [bestC,bestSigma] = kFoldCrossValidation(train_set,train_labels,numFolds)
% Summary 
% Use K-fold cross validation to find best "C" and "sigma" for a given training set

% Procedure:
% Assume numFolds = 5, hence 5-fold cross validation.

% 1) Partitiion trainig_set = training_set + validation set. This is done using
% matlabs crossvalind() as shown below.

% 2) Choose a range of C and sigma as shown below. 

% 4) For each validation set, calculate the binary misclassification error.
% Note - Either of one-vs-all or one-vs-one can be used to find error.
% Sum it up "k" times and then average it by dividing by "k". 
% This is the Cross Validation(CV) error.

% 5) Find the parameters that give the least CV error and this is the final result.

idx2Class = unique(train_labels);
total_classes = size(idx2Class,1);

% separate data into training and validation.
% assigns an index between 1-numFolds to each training sample
indices = crossvalind('Kfold',train_labels, numFolds);

% Range of values to search over.
C_range = 2.^(-1:1:8);
sigma_range = 2.^(1:1:7);

% Default values
bestCV_error = Inf;
bestC = -Inf;
bestSigma = -Inf;

% Find combination of (C,sigma) that gives minimum Cross validation error. 
for C = C_range
   for sigma = sigma_range

        fprintf('\n (C,sigma) = (%d,%d) %% \n', C,sigma); 
        fprintf(' (bestC,bestSigma) = (%d,%d) %% \n\n', bestC,bestSigma); 
       % Calculate cross validation error for each pair 
       % of given (c,sigma)
       sum = 0;
       for i = 1:numFolds

           validation_samples_cv = train_set(indices==i,:);
           validation_labels_cv = train_labels(indices==i,:);

           train_samples_cv = train_set(indices~=i,:);
           train_labels_cv = train_labels(indices~=i,:);

           % find Cross validation (CV) error 
           % Use one-vs-all classification to find CV error.
           % Note - Either of one-vs-all or one-vs-one can be used to find error.
           
           % Also, passing C and sigma doesnt imply that the kernel used 
           % is always RBF. These parameters can be (easily) ignored inside the code of
           % one_vs_all.m if a linear kernel is being used. Please see the code in one_vs_all.m
           acc = one_vs_all(train_samples_cv, train_labels_cv, ...
                            validation_samples_cv, validation_labels_cv,C,sigma);

           sum = sum + (1 - acc);  % since acc is a value between [0,1)                   

       end

       currentCV_error = sum/numFolds ;
       
       if( currentCV_error < bestCV_error ) 
           bestCV_error = currentCV_error;
           bestC = C;
           bestSigma = sigma;
       end
           
   end
end
