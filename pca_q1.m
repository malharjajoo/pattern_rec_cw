function [all_eigenvectors,all_eigenvalues,mu,W] = pca_q1(X)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

% calculate mean image - mu 
mu = mean(X')'; 

% calculate covariance matrix - S 

S = (1/size(X,1)) * (X-mu) * (X-mu)'; % note - this is computationally expensive.
%calculate SVD of S 
[U,diag_matrix,all_eigenvectors] = svd(S) ;

% disp('size of S,U,D,V=');
% size(S) 
% size(U)
% size(diag_matrix)

% eigenvalues, aka singular values
all_eigenvalues = diag(diag_matrix);

%truncating eigenvalues and eigenvectors since noticed values 
% were very close to zero. How can we truncate eigenvectors like this ?
%There is a V and V_T on SVD of (symmetric) covariance matrix
nz_eigenvalues = all_eigenvalues(1:size(X,2));
nz_eigenvectors = all_eigenvectors(1:size(X,2),1:size(X,2)); 

% how to choose "k" ?
size(nz_eigenvalues)

[best_k,best_reconstruction_error,best_variance] = choose_bestK(size(nz_eigenvalues,1),nz_eigenvalues);
fprintf('best k found = %d \n', best_k); 
fprintf('best reconstruction error = %f \n', best_reconstruction_error); 
fprintf('best variance = %f \n', best_variance); 
W = nz_eigenvectors(:,1:best_k);

%finally use PCA to project data samples to lower subspace.
% each row corresponds to an observation, but now has reduced
% dimensions ( = best_k ) 
projected_data = X*W; % Y = W' * X ;
 
%disp('check the sizes');size(projected_data)


recon2_error  = findReconstructionError_method2(projected_data,W,X);
fprintf('The reconstruction2 error = %d \n',recon2_error);
end


function [recon2_error] =  findReconstructionError_method2(projected_data,W,X)
    
% since Y = XW , and W is (almost?) orthogonal 
reconstructed_X = projected_data * W' ; 

total_rows = size(X,1);

total_err = 0.0;
    for i = 1:total_rows
         total_err = total_err +  sum((X(i,:) - reconstructed_X(i,:)).^2);
    end
    
recon2_error = total_err/total_rows;

end 

function checkOrthogonal(A)
     if size(nonzeros(A*A'-(eye(size(A,1),size(A,1)))),1) > 1 
        disp('Matrix is not orthogonal !');
     else
             disp('Matrix is orthogonal.');
     end
 
end

 function [best_k,best_reconstruction_error,best_variance] = choose_bestK(max_dim,eigvalues)
        
    % imp - this needs to be in ascending order 
    % since we want to find the smallest "k" that 
    % meets the "golden ratio" defined below.
    k_values = 1:max_dim;
 
    % 95 % variance retained
   golden_ratio = 0.05;

    % For each k value, find the ratio
    % As we keep increasing "k",in the ratio, the numerator
    % decreases and the denominator increases.
    best_k = k_values(1);
    best_reconstruction_error = +Inf;
    best_variance = 0;
    k_values
    for k = k_values
       reconstruction_error = sum(eigvalues(k+1:end));
       variance             = sum(eigvalues);
       
       %just update best results so far
        best_reconstruction_error = min(best_reconstruction_error,reconstruction_error);
        best_variance = max(best_variance,variance);
       reconstruction_error/variance
        if reconstruction_error/variance <= golden_ratio
            best_k = k;
            break;
        end
        
    end
       
 end

    
 
