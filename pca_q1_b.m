function [all_eigenvectors,all_eigenvalues,mu,W] = pca_q1_b(X)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

% calculate mean image - mu 
mu = mean(X')'; 

% calculate covariance matrix - S 

S = (1/size(X,1)) * (X-mu)'* (X-mu); % note - this is computationally expensive.
%calculate SVD of S 
[U,diag_matrix,all_eigenvectors] = svd(S) ;

disp('size of S,U,D,V=');
size(S) 
size(U)
size(diag_matrix)
size(diag_matrix)

% eigenvalues, aka singular values
all_eigenvalues = diag(diag_matrix);

% how to choose "k" ?
max_dim = size(all_eigenvalues,1);
[best_k,best_reconstruction_error,best_variance] = choose_bestK(max_dim,all_eigenvalues);
fprintf('best k found = %d ', best_k); 
fprintf('best reconstruction error = %f ', best_reconstruction_error); 
fprintf('best variance = %f ', best_variance); 
W = all_eigenvectors(:,1:best_k);

%finally use PCA to project data samples to lower subspace.
% each row corresponds to an observation, but now has reduced
% dimensions ( = best_k ) 
projected_data = X*W; % Y = W' * X ;
 
disp('check the sizes');
%size(projected_data)

disp('check the reproj error');
reproj  = findReprojectionError(projected_data,W,X)

end


function [reproj] =  findReprojectionError(projected_data,W,X)
    
% since Y = XW , and W is (hopefully) orthogonal 
reconstructed_X = projected_data * W' ; 

total_rows = size(X,1);

total_err = 0.0;
    for i = 1:total_rows
         total_err = total_err +  sum((X(i,:) - reconstructed_X(i,:)).^2);
    end
    
reproj = total_err/total_rows;

end 


 function [best_k,best_reconstruction_error,best_variance] = choose_bestK(max_dim,all_eigenvalues)
        
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
    
    for k = k_values
       reconstruction_error = sum(all_eigenvalues(k+1:end));
       variance             = sum(all_eigenvalues);
       
       %just update best results so far
        best_reconstruction_error = min(best_reconstruction_error,reconstruction_error);
        best_variance = max(best_variance,variance);
       
        if reconstruction_error/variance <= golden_ratio
            best_k = k;
            break;
        end
        
        

    end
       
 end

    
 
