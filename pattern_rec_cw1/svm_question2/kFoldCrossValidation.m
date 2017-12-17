function [bestC,bestSigma] = kFoldCrossValidation(train_samples,train_labels,numFolds)

% Use K-fold cross validation to find best "C" and "sigma" for a given training set

idx2Class = unique(train_labels);
total_classes = size(idx2Class,1);

% separate data into training and validation.
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

           validation_samples_cv = train_samples(indices==i,:);
           validation_labels_cv = train_labels(indices==i,:);

           train_samples_cv = train_samples(indices~=i,:);
           train_labels_cv = train_labels(indices~=i,:);

           % find Cross validation (CV) error 
           % Use one-vs-all classification to find CV error.
           acc = one_vs_one(train_samples_cv, train_labels_cv, ...
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

