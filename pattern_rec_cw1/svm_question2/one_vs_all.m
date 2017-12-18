function [accuracy] = one_vs_all(train_set,train_labels,test_set,test_labels,bestC,bestSigma)
% Summary 
% This function extends SVM to multi-class SVM by learning 52 classifiers,
% one for each class (Based on partitioning, training data always has at 
% least 1 sample belonging to each class ).

% The parameters "C" and "sigma" used during learning are obtained from
% 5-fold cross validation. See kFoldCrossValidation.m for details on procedure.

% After training, for each test sample, the classifier with the maximum
% "score" is found, and the corresponding class label is chosen as the 
% predictin label. Here "score" refers to an continuous value, not a 
% binary label of {-1,+1}.


    % Imp: maintain mapping between indices and actual classes.
    % eg: if there are 30 unique classes in training labels, but their
    % value is from , eg: 70 to 100. This is not the case in the input
    % data for this cw but for other cases, will be needed.
    idx2Class = unique(train_labels);
    total_classes = size(idx2Class,1); 

    tic;    % start timer for training phase. 
    
    % ===== One-vs-all ( Training Phase ) =====
    
    % SVMModels stores all 52 learnt classifiers.
    SVMModels = cell(total_classes,1); 

    % training "total_classes" classifiers
    for index = 1:total_classes

        % imp, cannot assume input set has all classes in a given range.
        % Hence safer to use an index to find out class.
        currentClass = idx2Class(index); 

        % for each class, convert all y labels of current class
        % to +1, and remaining labels to -1
       
        Y = train_labels; 
        Y(train_labels == currentClass)=  1;
        Y(train_labels ~= currentClass)= -1;

        % Note - Imp to standardize the training data set.
        % SVMModels{index} = fitcsvm(train_set,Y,'Standardize',true,...
          %  'KernelFunction','linear'); 
        
        SVMModels{index} = fitcsvm(train_set,Y,'Standardize',true,...
         'KernelFunction','rbf','KernelScale',bestSigma,'BoxConstraint',bestC); 

    end
    
    toc;
    
    
    tic;    % start timer for testing phase.
    % ===== One-vs-all ( Testing Phase ) =====

    % Matrix storing scores (for positive class) by all classifiers for 
    % each test sample.
    Scores = zeros(size(test_set,1),total_classes);

    % For each trained SVM, get the score for the "positive" class.
    for index = 1:total_classes 
        [~,score] = predict(SVMModels{index},test_set);
        % Second column contains positive-class scores
        Scores(:,index) = score(:,2); 
    end

    % Most imp, use the max score( max of all classifiers ) 
    % to obtain indices of the predicted classes.
    [~,maxScoreIndices] = max(Scores,[],2);

    predictorLabels = idx2Class(maxScoreIndices);
    toc;
    
    
    % ===== Performance metric summary ============
    
    [accuracy,precision,recall,specificity] = EvaluateClassifier(predictorLabels,test_labels);
    fprintf('(accuracy,precision,recall,specificity) of OVA = (%f %%,%f,%f,%f)\n',accuracy,precision,recall,specificity);


end





% Compares prediction labels with given labels for a test dataset
% and calculates accuracy as a percentage value.
function [acc] = findAccuracy(predictorLabels, testLabels)
    N = size(testLabels,1);
    acc = (sum(predictorLabels ==  testLabels )/N)*100;
end



% Inputs are prediciton and ground truth labels 
% Output are various statistics ( as an average over all classes)
% obtained from the confusion matrix.
function [accuracy,precision,recall,specificity] = EvaluateClassifier(prediction_labels, test_labels)

    total_classes = size(unique(test_labels),1); % 48 for current partition.

    % find accuracy
    accuracy = findAccuracy(prediction_labels, test_labels);
    confMatrix = confusionmat(test_labels',prediction_labels');

    sumConfMatrix = sum(confMatrix(:));
    
    % find precision , take average over all classes    
    precisionSum = 0;
    recallSum = 0 ;
    specificitySum = 0 ;
    
    for i=1:size(confMatrix,1)
    
         TP = confMatrix(i,i);
         colSum = sum(confMatrix(:,i));
         rowSum = sum(confMatrix(i,:));
         
         % Note - These guards are required since prediction
         % may not contain a particular class (but ground truth may)
         if(colSum ~= 0)
             % precision = TP/(TP+FP)
             precisionSum = precisionSum + (TP/colSum);
         end
         
         % Note - These guards are required since ground truth (test_labels)
         % may not contain a particular class.(but prediction may)
         if(rowSum ~= 0)
             % recall/sensitivity = TP/(TP+FN)
             recallSum = recallSum + (TP/rowSum);
         end
         
         % specificity =
         TN = sumConfMatrix - rowSum - colSum;
         val = (TN + (colSum - TP));
         
         if(val~=0)
             specificitySum = specificitySum + (TN/val);
         end
         
         
    end
    
    precision = precisionSum/total_classes;
    recall = recallSum/total_classes;
    specificity = specificitySum/total_classes;
    
end 



