function [all_eigenvectors,all_eigenvalues,mu,W,diag_matrix,reconstructed_X] = pca_q1(myTrain)
%Add a Summary 

% calculate mean image : mu 
mu = mean(myTrain,2); %takes mean along coloumns ( gives a mean value for each row )
X_centred = myTrain-mu;

% calculate covariance matrix - S 
% then calculate SVD of S

t_cols = size(X_centred,2);
%t_rows = size(X_centred,1);
% temp_sum = zeros(t_rows,t_rows);
% for i = 1:t_cols
%     temp_sum = temp_sum + ( X_centred(:,i) * X_centred(:,i)' ) ;
% end
% S = 1/t_cols*temp_sum;
S = 1/t_cols* (X_centred) * (X_centred)'; % note - this is computationally expensive.
[U,diag_matrix,all_eigenvectors] = svd(S) ;


% eigenvalues, aka singular values
all_eigenvalues = diag(diag_matrix); 


%truncating eigenvalues and eigenvectors since indices > 416 
% were very close to zero ( as seen in matlab variable view window ).
% ( Why do we truncate eigenvectors like shown below ? More explanation 
% given in report )
nz_eigenvalues = all_eigenvalues(1:size(X_centred,2));
nz_eigenvectors = all_eigenvectors(:,1:size(nz_eigenvalues,1)); 

% Method of choosing "k" -
 [best_k,best_reconstruction_error,best_variance] = choose_bestK(size(nz_eigenvalues,1),nz_eigenvalues);
 fprintf('best k found = %d \n', best_k); 
 fprintf('Reconstruction error found using eigenvalues= %f \n', best_reconstruction_error); 
 fprintf('variance of chosen number (k) principal components = %f \n', best_variance);

% since eigenvectors are in descending order ( eigenvectrs for
% decreasing values of corresponding eigenvalues ) and we want 
% the first "k" eigenvectors.
W = nz_eigenvectors(:,1:best_k);

%finally use PCA to project data samples to lower subspace.
% each row corresponds to an observation, but now has reduced
% dimensions ( = best_k ) 
PCA_Score = X_centred'*W;
reconstructed_X = (PCA_Score * W')'+ mu ; 

reconstruction2_error  = findReconstructionError_method2(PCA_Score,W,mu,myTrain);
fprintf('The reconstruction2 error = %f \n',reconstruction2_error);
end




%Method 1- calculate reconstruction error using eigenvalues
 function [best_k,best_reconstruction_error,best_variance] = choose_bestK(max_dim,eigvalues)
        
    % imp - this needs to be in ascending order 
    % since we want to find the smallest "k" that 
    % meets the "golden ratio" defined below. 
    % However, if desired then this can be done in reverse order too if you
    % invert the ratio (calculated below inside the loop ) , then can use the
    % k values in descending order too.
    
   k_values = 1:max_dim; % [1,2,3,4...520]
 
    % 95 % variance retained
   golden_ratio = 0.01;

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
function [recon2_error] =  findReconstructionError_method2(PCA_Score,W,mu,original_data)
    
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


 
 
%  %Just a helper function
%  function checkOrthogonal(A)
%      if size(nonzeros(A*A'-(eye(size(A,1),size(A,1)))),1) > 1 
%         disp('Matrix is not orthogonal !');
%      else
%              disp('Matrix is orthogonal.');
%      end
%  
% end


    
 
