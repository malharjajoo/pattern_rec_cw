function [] = pca_AppEigenfaces_q1_part_b(myTrain,labels)

    %Add a Summary 

    %1) find the total number of unique class labels 
    classes = unique(labels');
    total_classes = size(classes,1);
    
    for i = 1:total_classes
       classnameString = "l_"+string(i);
       className(i) = classnameString;
    end
    
    %2) filter the samples that correspond to each class label
    % this is done by storing indices of corresponding sample, in a array
    % at class label index
    class_to_sample_index =  zeros(total_classes,size(myTrain,2));

    total_samples = size(myTrain,2); %0.8 * coloumns of X
    for i = 1:total_samples
        index = labels(i);
        class_to_sample_index(index,i) = 1;

    end 


  W_matrix = findPCAForAllClasses(total_classes,class_to_sample_index,myTrain);
  

    % Once eigenbases have been obtained, for each test image 
    % calculate which "W"( principal eigenvectors or subspace ) 
    % will lead to lowest projection error on projecting test image.
    lowest_recon_error = +Inf;
    best_class_index = 0; % note - 0 doesnt correspond to any index and hence 
                            % can be used as default value ( since matlab array
                            % indexing begins from 1 )
                            
    for i = 1:total_classes
        
        % get the "W"(pca eigenvectors) for the current class.
        W = W_matrix.className;
   
        reconstruction2_error  = findReconstructionError_method2(W,myTrain);
        fprintf('The reconstruction2 error for class i = %d is = %f \n',i,reconstruction2_error);

        % update
        if current_recon_error < lowest_recon_error 
            lowest_recon_error = current_recon_error;
            best_class_index = i;
        end

    end


    % now that the best candidate has been found, just construct 
    % the test image belongs to the class.
    fprintf('Test image belongs to class = %s',string(best_class_index));

end

%stores W for each class in a struct.
function [W_matrix] = findPCAForAllClasses(total_classes,class_to_sample_index,myTrain)
    % For each class - calculate the PCA eigenvectors ( before testing begins later ) 
    % each row is a W_vector for the class at that index
    % how to store W_matrix of variable coloumns?
    for i = 1:total_classes

        samples = getSamplesForClassLabel(i,class_to_sample_index,myTrain);

        % now calculate PCA
        % we use matlab struct since each W found can have 
        % different dimensions ( because of different number of trainng
        % samples,etc) 
        W_matrix.a = findPCA(samples);

    end
    
end

function [W] = findPCA(myTrain)
    % calculate mean image : mu 
    mu = mean(myTrain,2); %takes mean along coloumns ( gives a mean value for each row )
    X_centred = myTrain-mu;
    
    % calculate covariance matrix - S 
    % then calculate SVD of S

    t_cols = size(X_centred,2);
    S = 1/t_cols * (X_centred)'*(X_centred); % note - this is efficeint
    [U,diag_matrix,all_eigenvectors] = svd(S) ;
    
    % transform eigenvectors , normalize each coloumn of the product;
    normalized_eigenvectors = myTrain * all_eigenvectors;
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
    fprintf('best k found = %d \n', best_k); 
    fprintf('Reconstruction error found using eigenvalues= %f \n', best_reconstruction_error); 
    fprintf('variance of chosen number (k) principal components = %f \n', best_variance);

    % since eigenvectors are in descending order ( eigenvectrs for
    % decreasing values of corresponding eigenvalues ) and we want 
    % the first "k" eigenvectors.
    W = nz_eigenvectors(:,1:best_k);
        
end


%Method 1- Theoretical Method - calculate reconstruction error using eigenvalues
 function [best_k,best_reconstruction_error,best_variance] = choose_bestK(max_dim,eigvalues)
        
    % imp - this needs to be in ascending order 
    % since we want to find the smallest "k" that 
    % meets the "golden ratio" defined below. 
    % However, if desired then this can be done in reverse order too if you
    % invert the ratio (calculated below inside the loop ) , then can use the
    % k values in descending order too.
    
   k_values = 1:max_dim; % [1,2,3,4...520]
 
    % 95 % variance retained
   golden_ratio = 0.1;

    % For each k value, find the ratio
    % As we keep increasing "k", the ratio numerator
    % decreases and the denominator increases and hence 
    % overall ratio decreases and the loop iterations 
    % will stop since ratio will converge to "golden_ratio" eventually.
    best_k = k_values(1);
    best_reconstruction_error = 0;
    best_variance = 0;
   
    for k = 1:max_dim
       reconstruction_error = sum(eigvalues(k+1:end));
       variance             = sum(eigvalues);
       
        
        if reconstruction_error/variance <= golden_ratio
            best_k = k;
            %just storing error and variance for value of chosen "k"
            best_reconstruction_error = reconstruction_error;
            best_variance = variance;

            break;
        end
        
    end
       
 end
 
 
 
 

%Method 2- calculate reconstruction error by re-constructing input and
%finding squared euclidean distance
function [recon2_error] =  findReconstructionError_method2(W,original_data)
    mu = mean(myTrain,2);
    X_centred = myTrain-mu;

    PCA_Score = X_centred'*W;
    % since Y = XW 
    %       YW' = X ( since W is orthogonal ?)
    reconstructed_X = (PCA_Score * W')'+ mu ; 

    total_samples = size(original_data,2);

    %calculates vector norm - |X-X_recons|^2
    total_err = 0.0;
    for i = 1:total_samples
         total_err = total_err +  sum((original_data(:,i) - reconstructed_X(:,i)).^2);
    end


    recon2_error = total_err/total_samples;

end 


  
% Helper function 
function [samples] = getSamplesForClassLabel(l,label_to_sample_index,myTrain)
    sample_indices = find(label_to_sample_index(l,:)==1);
    
    samples = myTrain(:,sample_indices);
end

