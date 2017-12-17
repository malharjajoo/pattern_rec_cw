function [all_eigenvectors,all_eigenvalues,mu,W,diag_matrix,reconstructed_train_set] = pca_eigenfaces_q1_part_a(train_set)

    % calculate mean image : mu 
    mu = mean(train_set,2); %takes mean along coloumns ( gives a mean value for each row )
    X_centred = train_set-mu;

    % calculate covariance matrix: S 
    % then calculate SVD of S
    N = size(X_centred,2);
    S = 1/N* (X_centred) * (X_centred)'; % Note - this is computationally expensive.
    [U,diag_matrix,all_eigenvectors] = svd(S) ;


    % Simply extract diagonal as a coloumn vector
    all_eigenvalues = diag(diag_matrix); 

    % truncating eigenvalues and eigenvectors since indices > N-1
    % were very close to zero ( as seen in matlab variable view window ).
    % Eigenvalues are already returned in descending order by Matlab svd().
    N = size(X_centred,2);
    nz_eigenvalues = all_eigenvalues(1:N-1);
    nz_eigenvectors = all_eigenvectors(:,1:N-1); 

    % maximum possible vlaue of "k" - WHY ? Need to find answer.
    max_dim = size(nz_eigenvalues,1);

    % Uncomment below to plot mean image and eigenvectors
    % plotFigures(nz_eigenvectors,mu);
    % plotReconErrorAgainst_k(max_dim,nz_eigenvalues);

    % Heuristic for choosing "k" -
    [best_k,reconstruction_error1,best_variance] = choose_bestK(max_dim,nz_eigenvalues);
    fprintf('best k found = %d \n', best_k); 
    fprintf('Reconstruction error found using theoretical method = %f \n', reconstruction_error1); 
    fprintf('variance of chosen number (k) principal components = %f \n', best_variance);

    % since eigenvectors are in descending order ( eigenvectrs for
    % decreasing values of corresponding eigenvalues ) and we want 
    % the first "k" eigenvectors.
    W = nz_eigenvectors(:,1:best_k);

    % finally use PCA to project data samples to reduced dimension subspace.
    % each ROW corresponds to an observation, but now has reduced
    % dimensions ( = best_k obtained above ) 

    % dimensions of each term.
    % [ m x best_k ] = [ m x n ] * [ n x best_k ]  
    PCA_Score        = X_centred'* W;
    [reconstruction_error2, reconstructed_train_set]  = findReconstructionError_method2(PCA_Score,W,mu,train_set);
    fprintf('Reconstruction error found using practical method = %f\n',reconstruction_error2);
    
    
end


% ============== 2 Methods for calculating Reconstruction Error ===========

% Method 1(Theoretical) - Calculate reconstruction error using eigenvalues
 function [best_k,best_reconstruction_error,best_variance] = choose_bestK(max_dim,eigvalues)
        
    % 99 % variance retained
    threshold = 0.05;

    % For each k value, find the ratio as shown below.
    % As we keep increasing "k", the ratio numerator
    % decreases and the denominator increases and hence 
    % overall ratio decreases and converges to "threshold" before 
    % the loop iterations reach max_dim .
    best_k = -1;
    best_reconstruction_error = -1;
    best_variance = -1;
   
    variance = sum(eigvalues);
    
    % imp - k values need to be in ascending order 
    % since we want to find the smallest "k" that 
    % meets the "golden ratio" defined below. 
    
    for k = 1:max_dim % [1,2,3,4...415]
       reconstruction_error = sum(eigvalues(k+1:end));
       
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
function [recon_error, reconstructed_X] =  findReconstructionError_method2(PCA_Score,W,mu,original_data)
    
% since Y = X_centred * W 
%       YW' = X_centred ( since W is orthogonal )
%       X   = YW' + mean_image
reconstructed_X = (PCA_Score * W')'+ mu ; 
 
total_samples = size(original_data,2);

% calculates squared euclidean distance/vector norm - |X-X_recons|^2
% for each coloumn vector in both matrices.
total_err = 0.0;
    for i = 1:total_samples
         total_err = total_err + sum((original_data(:,i) - reconstructed_X(:,i)).^2);
    end
    
recon_error = (total_err)/total_samples;

end 



% ======================= Helper functions: plot data =================

% Plots the mean image and eigenvectors
function plotFigures(eigenvectors, mean_image)
    
% images we want to compare visually 
% resize selected face feature vector to a 46x56 matrix and display as
% grayscale image

% display first 10 eigenvectors.

    for i = 1:10
        img = mat2gray(vec2mat(eigenvectors(:,i),56));
        subplot(2,5,i);
        imshow(img);
    end
    
    figure;
    
    img = mat2gray(vec2mat(mean_image(:,1),56));
    imshow(img);
end

    
 % plotReconErrorAgainst_k 
 function [] = plotReconErrorAgainst_k(max_dim,eigvalues)
        
    % imp - k values need to be in ascending order 
    % since we want to find the smallest "k" that 
    % meets the "golden ratio" defined below. 
    k_values = 1:max_dim; % [1,2,3,4...415]
 
    % different thresholds in order to obtain different "k"
    threshold_values = 0:0.01:0.20;
    best_k_values = zeros(size(threshold_values));
    best_recon_errors = zeros(size(threshold_values));
   
     % for each threshold find a "k"
     for i = 1:size(threshold_values,2)
         
        for k = 1:max_dim
           reconstruction_error = sum(eigvalues(k+1:end));
           variance             = sum(eigvalues);
  
            if reconstruction_error/variance <= threshold_values(i)
                best_k_values(i) = k;
                %just storing error and variance for value of chosen "k"
                best_recon_errors(i) = reconstruction_error;
                break;
            end
            
        end
        
     end
       
    
     % plot the data
     c = linspace(30,40,length(best_k_values));
     scatter(best_k_values,best_recon_errors,25,c,'filled');
     title('Reconstruction error with increasing k');
     xlabel('k (reduced dimension)');
     ylabel('Reconstruction error');
 end
 
 
