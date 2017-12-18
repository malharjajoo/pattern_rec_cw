function [accuracy] = one_vs_one(train_samples,train_labels,test_samples,test_labels,bestC,bestSigma)
   
 
    % imp to maintain mapping between indices and actual classes.
    % eg: if there are 30 unique classes in training labels, but their
    % value is from , eg: 70 to 100.
    idx2Class = unique(train_labels);
    total_classes = size(idx2Class,1); % <=52(since depends on partition) in our case.

    tic; % start timer for training phase. 
    
    
    % ======= One-vs-one ( Training Phase ) ===========
    % 1) For each pair of classes, first extract training labels 
    % and corresponding samples.

    % 2) Then convert all y labels of first class
    % to +1, and labels of other class to -1
    
    SVMModel_cell = cell(1326,1);
    
    % Iterate over all possible combinations
    % ( in an efficient manner, looks similar to bubble sort )
    % and train 52C2 = 1326 SVM classifiers.
    idx1 = 1;
    for i = 1:total_classes 
        for j = i+1:total_classes 

            % unlike OvA method, we need to filter the training labels
            % and samples based on class pair in current iteration.
            indices = train_labels == i | train_labels == j ;
            training_samples12 = train_samples(indices,:) ;
            temp_labels12  = train_labels(indices) ;

            % convert class labels to +1 or -1 depending on positive class.
            % Here the "i" class is the positive class.( and "j" is negative)
            training_labels12 = temp_labels12; 
            training_labels12(temp_labels12 == i)= 1;
            training_labels12(temp_labels12 == j)= -1;

            % uncomment 1 of the 2 lines below to use either linear or gaussian kernel.
             SVMModel_cell{idx1} = fitcsvm(training_samples12,training_labels12,'Standardize',true,...
            'KernelFunction','linear'); 
             %SVMModel_cell{idx1} = fitcsvm(training_samples12,training_labels12,'Standardize',true,'KernelFunction','rbf',...
             %   'KernelScale',bestSigma,'BoxConstraint',bestC);
            
            idx1 = idx1 + 1;
            
        end
    end
    
    toc;
    
    
    
    tic; % start timer for testing phase. 
    
    % ========= One-vs-one ( Testing Phase )===========
    % A "voting/histogram" method for predictions -
    
    % This is done by using a hashtable with a row for each
    % class and a coloumn for each test sample, for storing "votes".
    
    % Each vote is generated based on prediction by a single classifier ( see below
    % inside the two nested for-loops ). The corresponding entry of the hashtable
    % is then incremented.
    
    % Finally, predictions are extracted from the hash table
    HashTable = zeros(total_classes,size(test_samples,1));
    idx2 = 1;
    
    % loop over all classifiers.
    for i = 1:total_classes 
        for j = i+1:total_classes 

            % unlike one-vs-all, we don't need the prediction score here. 
            [labels,~] = predict(SVMModel_cell{idx2},test_samples);
    
            % For each prediction, increment entries of hashtable by 1 
            for k = 1:size(labels,1)
                if(labels(k)==1)
                    HashTable(i,k) = HashTable(i,k) + 1; 
                else 
                    HashTable(j,k) = HashTable(j,k) + 1; 
                end
            end
    
            idx2 = idx2 +1;
        end
    end
            
    % Convert Hashtable to prediction labels 
    predLabels = zeros(size(test_samples,1),1);
    
    % Gather labels from each coloumn of the hashtable
    for i = 1:size(HashTable,2)
        [~,idx] = max(HashTable(:,i));
        predLabels(i,1) = idx2Class(idx);
    end
    
    
    
    toc;
    
    
    
     % ===== Performance metric summary ============
         
    [accuracy,precision,recall,specificity] = EvaluateClassifier(predLabels,test_labels);
    fprintf('(accuracy,precision,recall,specificity) of OVO = (%f %% ,%f,%f,%f)\n',accuracy,precision,recall,specificity);

end


% ============ Helper function ========================

% Compares prediction labels with given labels for a test dataset
% and calculates accuracy as a percentage value.
function [acc] = findAccuracy(predictorLabels, testLabels)
    N = size(testLabels,1);
    acc = (sum(predictorLabels ==  testLabels )/N)*100;

end



% Inputs are prediction and ground truth labels 
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




