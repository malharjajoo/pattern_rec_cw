function [acc] = one_vs_all(train_samples,train_labels,test_samples,test_labels,bestC,bestSigma)
    
    % Imp: maintain mapping between indices and actual classes.
    % eg: if there are 30 unique classes in training labels, but their
    % value is from , eg: 70 to 100.
    idx2Class = unique(train_labels);
    % <=52(since depends on partition) in our case.
    total_classes = size(idx2Class,1); 

        
    % One-vs-all ( Training Phase )
    
    % some value <= 52 depending on training set partition.
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

        % Use cross validation here to choose bestC and bestSigma
       
        SVMModels{index} = fitcsvm(train_samples,Y,'Standardize',true,...
            'KernelFunction','rbf','KernelScale',bestSigma,'BoxConstraint',bestC);

    end


    % ===== One-vs-all ( Testing Phase ) =====

    % Matrix storing scores (for positive class) by all classifiers for 
    % each test sample.
    Scores = zeros(size(test_samples,1),total_classes);

    % For each trained SVM, get the score for the "positive" class.
    for index = 1:total_classes 
        [~,score] = predict(SVMModels{index},test_samples);
        % Second column contains positive-class scores
        Scores(:,index) = score(:,2); 
    end

    % Most imp, use the max score( max of all classifiers ) 
    % to obtain indices of the predicted classes.
    [~,maxScoreIndices] = max(Scores,[],2);

    predictorLabels = idx2Class(maxScoreIndices);

    acc = findAccuracy(predictorLabels, test_labels);
    fprintf('Accuracy of one-vs-all method = %f %% \n', acc); 

end





% Compares prediction labels with given labels for a test dataset
% and calculates accuracy as a percentage value.
function [acc] = findAccuracy(predictorLabels, testLabels)
    N = size(testLabels,1);
    acc = (sum(predictorLabels ==  testLabels )/N)*100;
end
