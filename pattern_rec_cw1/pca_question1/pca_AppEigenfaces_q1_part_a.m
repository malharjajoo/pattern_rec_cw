function [] = pca_AppEigenfaces_q1_part_a(train_set, W, test_set) 
% Use reconstructed training set to visually see results of PCA
% Resize selected face feature vector to a 46x56 matrix and display as
% grayscale image

% =================== Reconstruction of Training data ===================

% Calculate reconstruction as in previous question ..
mu1 = mean(train_set,2);
test_set_centred = train_set-mu1;
PCA_Score        = test_set_centred'* W;
reconstructed_train_set = (PCA_Score * W')'+ mu1 ;


% Chosen randomly by picking manually from 3 different classes
% Can be done using code but meh ...
training_indices = [10,47,244]; 

% 3 images from training set
image1 = mat2gray(vec2mat(train_set(:,training_indices(1)),56));
subplot(2,3,1);
imshow(image1);

image2 = mat2gray(vec2mat(train_set(:,training_indices(2)),56));
subplot(2,3,2);
imshow(image2);

image3 = mat2gray(vec2mat(train_set(:,training_indices(3)),56));
subplot(2,3,3);
imshow(image3);

% corresponding reconstructed images
image4 = mat2gray(vec2mat(reconstructed_train_set(:,training_indices(1)),56));
subplot(2,3,4);
imshow(image4);

image5 = mat2gray(vec2mat(reconstructed_train_set(:,training_indices(2)),56));
subplot(2,3,5);
imshow(image5);

image6 = mat2gray(vec2mat(reconstructed_train_set(:,training_indices(3)),56));
subplot(2,3,6);
imshow(image6);

figure;


% =================== Reconstruction of Test data ====================

% Calculate reconstruction as in previous question ..
mu2 = mean(test_set,2);
test_set_centred = test_set-mu2;

% Note - The "W" used here is from the PCA output for training data.
% ie, the "W" is PCA subspace for training data.
PCA_Score        = test_set_centred'* W;
reconstructed_test_set = (PCA_Score * W')'+ mu2 ;


test_indices = [32,67,100]; %chosen randomy from different classes

% 3 images from test/validation set
image7 = mat2gray(vec2mat(test_set(:,test_indices(1)),56));
subplot(2,3,1);
imshow(image7);

image8 = mat2gray(vec2mat(test_set(:,test_indices(2)),56));
subplot(2,3,2);
imshow(image8);

image9 = mat2gray(vec2mat(test_set(:,test_indices(3)),56));
subplot(2,3,3);
imshow(image9);

% corresponding reconstructed images
image10 = mat2gray(vec2mat(reconstructed_test_set(:,test_indices(1)),56));
subplot(2,3,4);
imshow(image10);

image11 = mat2gray(vec2mat(reconstructed_test_set(:,test_indices(2)),56));
subplot(2,3,5);
imshow(image11);

image12 = mat2gray(vec2mat(reconstructed_test_set(:,test_indices(3)),56));
subplot(2,3,6);
imshow(image12);


end

