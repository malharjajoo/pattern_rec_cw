function [all_eigenvectors,all_eigenvalues,mu,W,diag_matrix,reconstructed_X] = pca_eigenfaces_q1_part_a(myTrain)

% calculate mean image : mu 
mu = mean(myTrain,2); %takes mean along coloumns ( gives a mean value for each row )
X_centred = myTrain-mu;

% calculate covariance matrix - S 
% then calculate SVD of S

N = size(X_centred,2);
%t_rows = size(X_centred,1);
% temp_sum = zeros(t_rows,t_rows);
% for i = 1:N
%     temp_sum = temp_sum + ( X_centred(:,i) * X_centred(:,i)' ) ;
% end
% S = 1/N*temp_sum;
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

% Heuristic for choosing "k" -
[best_k,best_reconstruction_error,best_variance] = choose_bestK(size(nz_eigenvalues,1),nz_eigenvalues);
fprintf('best k found = %d \n', best_k); 
fprintf('Reconstruction error found using eigenvalues= %f \n', best_reconstruction_error); 
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
reconstructed_X = (PCA_Score * W')'+ mu ;  % imp step
checkOrthogonal(W);
reconstruction2_error  = findReconstructionError_method2(PCA_Score,W,mu,myTrain);
fprintf('The reconstruction2 error = %f \n',reconstruction2_error);
end




% Method 1- calculate reconstruction error using eigenvalues
 function [best_k,best_reconstruction_error,best_variance] = choose_bestK(max_dim,eigvalues)
        
    % imp - k values need to be in ascending order 
    % since we want to find the smallest "k" that 
    % meets the "golden ratio" defined below. 
    
    k_values = 1:max_dim; % [1,2,3,4...415]
 
    % 99 % variance retained
    threshold = 0.01;

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
 
 
 
 

% Method 2- calculate reconstruction error by re-constructing input and
% finding squared euclidean distance
function [recon2_error] =  findReconstructionError_method2(PCA_Score,W,mu,original_data)
    
% since Y = X_centred * W 
%       YW' = X_centred ( since W is orthogonal )
%       X   = YW' + mean_image
reconstructed_X = (PCA_Score * W')'+ mu ; 
 
total_samples = size(original_data,2);

% calculates squared euclidean distance/vector norm - |X-X_recons|^2
% for each coloumn vector in both matrices.
total_err = 0.0;
    for i = 1:total_samples
         total_err = total_err +  sum((original_data(:,i) - reconstructed_X(:,i)).^2);
    end
    
 
recon2_error = total_err/total_samples;

end 


 
 
%  %Just a helper function
 function checkOrthogonal(A)
     if size(nonzeros(A*A'-(eye(size(A,1),size(A,1)))),1) > 1 
        disp('Matrix is not orthogonal !');
     else
             disp('Matrix is orthogonal.');
     end
 
end


    
 
