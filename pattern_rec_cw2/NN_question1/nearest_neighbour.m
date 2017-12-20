function [best_metric, lowest_class_error, best_accuracy] = nearest_neighbour(train_set,train_label,test_set,test_label)

    % M_train = size(train_set,1); %rows in train data
    % N_train = size(train_set,2); %cols in train data
    % 
    % M_test = size(test_set,1);  %rows in test data
    % N_test = size(test_set,2);  %cols in test data

    % 9 metrics
    metrics_nameList = ["cityblock","chebychev","correlation","cosine",...
                         "euclidean","hamming","jaccard","mahalanobis",...
                         "minkowski"];

    % row contains prediction labels for each metric.
    % coloumn corresponds to metric.
    % dimension is 60x<number of metrics> for now....
    metric_prediction_labels = zeros(size(test_set,1), size(metrics_nameList,2));            

    % k-value for each neighbour
    k_values = [1,2,3,4,5,6,7,8,9,10,11,12];
    Error_list = ones(size(metrics_nameList,2),size(k_values,2));
    best_accuracy = ones(1,size(k_values,2));
    % probably a bad way of initializing this .. TODO:change later
    best_metric = string(k_values);
    
    

    % For each k
        % For each distance metric, 
            % Perform nearest neighbour classification. Store prediction labels.
            % Find the classification error on TEST set.
            % Choose distance metric with least classification error.
    for k = k_values
        for i = 1:size(metrics_nameList,2)
   
            metric_name = char(metrics_nameList(i));
        
            Mdl = fitcknn(train_set,train_label,'NumNeighbors',k,'Standardize',1,...
                    'Distance',metric_name,'NSMethod','exhaustive') ; 

            [label,score,cost] = predict(Mdl,test_set);
            metric_prediction_labels(:,i) = label;
            
           
        end
        
        % Find binary classification error for a metric for a given k.
        binary_class_error = findClassificationError(metric_prediction_labels,test_label);
        Error_list(:,k) = binary_class_error;
        
    end
    
    % For each k, find metric with lowest classification error
    % and save results.
    for i = 1:size(k_values,2)
    
        binary_class_error = Error_list(:,i);
        [lowest_class_error,idx] = min(binary_class_error);
        best_metric(i) = metrics_nameList(idx);
        best_accuracy(i) = (1-lowest_class_error)*100;
    end
    
    
    % ====== Plot all results ==========
    
    % plot (Binary) classification error ?
    plotGraph(k_values,metrics_nameList,Error_list);
end



% ================= Helper functions ===========================

function [binary_class_error] = findClassificationError(metric_prediction_labels,test_label)

    total_metrics = size(metric_prediction_labels,2);
    total_test_labels = size(test_label,1); % This value is 60 for now ...
    
    % output of function : each entry stores classification error for a
    % metric with that index in metric_nameList ( in above function ) 
     binary_class_error = zeros(total_metrics,1) ; 
     
    % loop over each metric
    for j = 1:total_metrics
        % Find classification Error for each metric
        % by comparing with test_label
        sum = 0 ; 
        for i = 1:size(test_label,1)
            if test_label(i) ~= metric_prediction_labels(i,j) 
                sum = sum + 1;
            end
        end
        
        err = sum/total_test_labels;
        binary_class_error(j) = err;
    end

end



% Creates a scatterplot with each metric as series plot
% For each value of k.
function plotGraph(k_values,metrics_nameList,Error_list)
    
    total_metrics = size(metrics_nameList,2);
    Legend = cell(total_metrics,1);
    
    for j = 1:size(k_values,2)
        k_val = k_values(j);
        
        for i = 1:total_metrics
            scatter(k_val,Error_list(i,j));
            Legend{i} = char(metrics_nameList(i)) ;
        
            % used to create series plot.
            hold on 
        end
    end
    
    xlabel('k values');
    ylabel('binary classification error');
    title('Classification Error for distance metrics');
    legend(Legend);
    
end
