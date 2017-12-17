function [acc] = one_vs_one(train_samples,train_labels,test_samples,test_labels,bestC,bestSigma)
   
%     % Use default values ( found using 5-fold cross validation for entire
%     % dataset ) 
%     switch nargin
%         case 4
%             bestC = 32;
%             bestSigma = 128;
%         case 5
%             bestSigma = 128;
%     end

    % imp to maintain mapping between indices and actual classes.
    % eg: if there are 30 unique classes in training labels, but their
    % value is from , eg: 70 to 100.
    idx2Class = unique(train_labels);
    total_classes = size(idx2Class,1); % <=52(since depends on partition) in our case.


    % ======= One-vs-one ( Training and Testing Phase )===========
    % 1) For each pair of classes, first extract training labels 
    % and corresponding samples.

    % 2) Then convert all y labels of first class
    % to +1, and labels of other class to -1
    
    % 3) Cross validation is used to find best parameters(smallest
    % validation erorr) for "C" and "sigma".

    % 4) A "voting/histogram" method for predictions -
    
    % It is used to find the prediction for a test sample. 
    % This is done by using a hashtable with a row
    % for each class and a coloumn for each label, for storing "votes".
    
    % Each vote is generated based on prediction by a single classifier ( see inside 
    % the two nested loops below )
    
    HashTable = zeros(total_classes,size(test_samples,1));
    
    % Iterate over all possible combinations
    % ( in an efficient manner, similar to bubble sort )
    % and train 52C2 = 1326 SVM classifiers.
    for i = 1:total_classes 
        for j = i+1:total_classes 

            % unlike OvA method, we need to filter the training labels
            % and samples based on class pair in current iteration.
            indices = train_labels == i | train_labels == j ;
            training_samples12 = train_samples(indices,:) ;
            temp_labels12  = train_labels(indices) ;

             training_labels12 = temp_labels12; 
             training_labels12(temp_labels12 == i)= 1;
             training_labels12(temp_labels12 == j)= -1;

            % Use cross validation on reduced training set
            
            SVMModels2 = fitcsvm(training_samples12,training_labels12,'Standardize',true,'KernelFunction','rbf','KernelScale',bestSigma,'BoxConstraint',bestC);
 
            % unlike one-vs-all, we don't need the prediction score here. 
            [labels,score] = predict(SVMModels2,test_samples);
            
            % For each prediction, increment entries of hashtable by 1 
            for k = 1:size(labels,1)
                if(labels(k)==1)
                    HashTable(i,k) = HashTable(i,k) + 1; 
                    
                else 
                    HashTable(j,k) = HashTable(j,k) + 1; 
                end
               
            end
            
        end
    end
    
    
    % Convert Hashtable to prediction labels 
    predLabels = zeros(size(test_samples,1),1);
    
    % Gather labels from each coloumn of the hashtable
    for i = 1:size(HashTable,2)
        [d,idx] = max(HashTable(:,i));
        predLabels(i,1) = idx2Class(idx);
    end

    acc = findAccuracy(predLabels, test_labels);
    fprintf('Accuracy of one-vs-one method = %f %% \n', acc); 
    

end





% Compares prediction labels with given labels for a test dataset
% and calculates accuracy as a percentage value.
function [acc] = findAccuracy(predictorLabels, testLabels)
    N = size(testLabels,1);
    acc = (sum(predictorLabels ==  testLabels )/N)*100;

end



% Use cross validation to find best paramters for a given training set
function [bestC,bestSigma] = findBestParameters(train_samples,train_labels,numFolds)

    % separate data into training and validation.
    indices = crossvalind('Kfold',train_labels, numFolds);
    
    c_range = 70:110;
    sigma_range = 20:40;
    % For each combination of C and sigma values 
    % Do 5 fold cross validation.
    for c = 1:set
       for sigma = 1:set
           
           % Calculate cross validation error for each 
           % for a given (c,sigma)
           
           for i = 1:numFolds
              
               validation_samples = train_samples(indices==i,:);
               train_samples_cv = train_samples(indices~=i,:);
               train_labels_cv = train_labels(indices~=i,:);
           end
           
       end
    end
end
