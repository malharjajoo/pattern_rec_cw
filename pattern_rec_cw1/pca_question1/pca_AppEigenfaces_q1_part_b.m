function [] = pca_AppEigenfaces_q1_part_b(train_set,train_labels,test_set,test_labels)

    %1) Find the total number of unique class labels 
    classes = unique(train_labels');      
    total_classes = size(classes,1);
    
   
    % This is required later (see below)
    % Generate struct names dynamically at runtime.
    className = cell(total_classes,1);
    for j = 1:total_classes
        classnameString = "l_"+string(j);
        className{j} = classnameString;
    end

    
    % 2) Group training samples based on labels
    class_to_sample_index =  zeros(total_classes,size(train_set,2));

    total_samples = size(train_set,2); 
    for j = 1:total_samples
        index = train_labels(j);
        class_to_sample_index(index,j) = 1;
    end 

    
    
    % Find W_i for each of 52 classes
    W_matrix = findPCAForAllClasses(total_classes,class_to_sample_index,train_set,className);


    total_test_samples = size(test_set,2);     % 104 for current paritition           
    prediction_labels = zeros(total_test_samples,1);
    
    % Mean-centre test set
    mu = mean(test_set,2); %takes mean along coloumns ( gives a mean value for each row )
    test_centred = test_set-mu;
    
    % For each test image in test_centred
    for i = 1:total_test_samples
        
    current_test_sample = test_centred(:,i);
    
    % Once eigenbases have been obtained,
    % calculate which "W"( principal eigenvectors or subspace ) 
    % will lead to lowest re-projection error on projecting test image.
    lowest_recon_error = +Inf;
    
        for j = 1:total_classes

            % get the "W"(pca eigenvectors) for the current class.
            w_name = char(className{j});
            W = W_matrix.(w_name); % eg: W = W_matrix.l_34 ;
            
            PCA_Score = W'* current_test_sample; % PCA_score should have dimension 8x1, 6x1, etc.
            current_recon_error = findReconstructionError_SingleSample(PCA_Score,W,mu,current_test_sample);
            %fprintf('The reconstruction2 error for class = %d is = %f \n',j,current_recon_error);

            % update
            if current_recon_error < lowest_recon_error 
                lowest_recon_error = current_recon_error;
                prediction_labels(i) = j;
            end

        end
    end

    
    % Find prediction accuracy of PCA, this is compared with SVM later...
  
    [accuracy,precision,recall,specificity] = EvaluateClassifier(prediction_labels,test_labels');
    fprintf('The accuracy of PCA = %f \n',accuracy);
    fprintf('The precision of PCA = %f \n',precision);
    fprintf('The recall of PCA = %f \n',recall);
    fprintf('The specificity of PCA = %f \n',specificity);

end





% =========================== find PCA ===========================

% stores W for each class in a struct.
function [W_matrix] = findPCAForAllClasses(total_classes,class_to_sample_index,train_set,className)

    % For each class - calculate the PCA eigenvectors ( before testing begins later ) 
    % each row is a W_vector for the class at that index
    % how to store W_matrix of variable coloumns? - use matlab struct
    for i = 1:total_classes

        samples = getSamplesForClassLabel(i,class_to_sample_index,train_set);

        % now calculate PCA
        % we use matlab struct since each W found can have 
        % different dimensions ( because of different number of trainng
        % samples in each, since dimension of each W_i is [N_i x k_i]) 
        % and "N_i" and "k_i" are different for each class since "N_i" 
        % is number of training samples of a particular class.
        W_matrix.(char(className{i})) = findPCA(samples);

    end
    
end




function [W] = findPCA(train_set)
    % calculate mean image : mu 
    mu = mean(train_set,2); %takes mean along coloumns ( gives a mean value for each row )
    X_centred = train_set-mu;
    
    % calculate covariance matrix - S 
    % then calculate SVD of S

    t_cols = size(X_centred,2);
    S = 1/t_cols * (X_centred)'*(X_centred); % note - this is efficeint
    [U,diag_matrix,all_eigenvectors] = svd(S) ;
    
    % transform eigenvectors , normalize each coloumn of the product;
    normalized_eigenvectors = train_set * all_eigenvectors;
    for i = 1:size(normalized_eigenvectors,2)

        Temp = normalized_eigenvectors(:,i);
        vector_norm = sqrt(sum(Temp.*Temp)); % vector norm
        normalized_eigenvectors(:,i) = normalized_eigenvectors(:,i)/vector_norm;
    end

    % eigenvalues, aka singular values
    all_eigenvalues = diag(diag_matrix); 

    %truncating eigenvalues and eigenvectors since indices > 416 
    % were very close to zero ( as seen in matlab variable view window ).
    % ( Why do we truncate eigenvectors like shown below ? More explanation 
    % given in report )
    nz_eigenvalues = all_eigenvalues(1:size(X_centred,2));
    nz_eigenvectors = normalized_eigenvectors(:,1:size(nz_eigenvalues,1)); 

    % Method of choosing "k" -
    [best_k,best_reconstruction_error,best_variance] = choose_bestK(size(nz_eigenvalues,1),nz_eigenvalues);
%     fprintf('best k found = %d \n', best_k); 
%     fprintf('Reconstruction error found using eigenvalues= %f \n', best_reconstruction_error); 
%     fprintf('variance of chosen number (k) principal components = %f \n', best_variance);

    % since eigenvectors are in descending order ( eigenvectrs for
    % decreasing values of corresponding eigenvalues ) and we want 
    % the first "k" eigenvectors.
    W = nz_eigenvectors(:,1:best_k);
        
end

%====================== finding "k" ======================

% Method 1(Theoretical) - Calculate reconstruction error using eigenvalues
function [best_k,best_reconstruction_error,best_variance] = choose_bestK(max_dim,eigvalues)
        
    % imp - k values need to be in ascending order 
    % since we want to find the smallest "k" that 
    % meets the "golden ratio" defined below. 
    k_values = 1:max_dim; % [1,2,3,4...415]
 
    % 99 % variance retained
    threshold = 0.05;

    % For each k value, find the ratio
    % As we keep increasing "k", the ratio numerator
    % decreases and the denominator increases and hence 
    % overall ratio decreases and converges to "threshold" before 
    % the loop iterations reach max_dim .
    best_k = k_values(1);
    best_reconstruction_error = 0;
    best_variance = 0;
   
    for k = 1:max_dim
       reconstruction_error = sum(eigvalues(k+1:end));
       variance             = sum(eigvalues);
       
        if reconstruction_error/variance <= threshold
            best_k = k;
            %just storing error and variance for value of chosen "k"
            best_reconstruction_error = reconstruction_error;
            best_variance = variance;

            break;
        end 
    end 
end
 
 
 
 
 
 
% Method 2(Practical)- calculate reconstruction error by re-constructing input and
% finding squared euclidean distance
function [recon_error] =  findReconstructionError_SingleSample(PCA_Score,W,mu,test_sample)
    
% since Y = X_centred * W 
%       YW' = X_centred * I ( W*W' = I since W is orthogonal )
%       X   = X_centred + mean_image = YW' + mean_image
reconstructed_sample = ( W * PCA_Score )+ mu ; 
 
% calculates squared euclidean distance/vector norm - |X-X_recons|^2
% for each coloumn vector in both matrices.
recon_error =  sum((test_sample(:,1) - reconstructed_sample(:,1)).^2);
recon_error = recon_error/size(test_sample,1);
end 



  
% ======================== Helper functions =========================

% Given a class label, finds all training samples having that label
function [samples] = getSamplesForClassLabel(class,class_to_sample_index,train_set)
    indices = find(class_to_sample_index(class,:)==1);
    samples = train_set(:,indices);
end


% Compares prediction labels(coloumn vector) with given labels(coloumn vector)
% for a test dataset and calculates accuracy as a percentage value.
function [acc] = findAccuracy(predictionLabels, testLabels)
    N = size(testLabels,1);
    acc = (sum(predictionLabels ==  testLabels )/N)*100;
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

