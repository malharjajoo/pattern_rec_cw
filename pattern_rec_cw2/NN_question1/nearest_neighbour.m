function [best_metric, lowest_class_error, best_accuracy] = nearest_neighbour(train_set,train_label,test_set,test_label)

    % M_train = size(train_set,1); %rows in train data
    % N_train = size(train_set,2); %cols in train data
    % 
    % M_test = size(test_set,1);  %rows in test data
    % N_test = size(test_set,2);  %cols in test data

    knn_method =  'exhaustive'; %'kdtree';
    
    metrics_nameList = [];
    % 9 metrics
    if(strcmp(knn_method,'exhaustive'))
        metrics_nameList = ["cityblock","chebychev","correlation",...
                         "cosine","euclidean","hamming"...
                         "jaccard","mahalanobis","minkowski","spearman"];
                     
                         
    else 
        metrics_nameList = ["cityblock","chebychev","euclidean","minkowski"];
    end
    

    
                     
    % row contains prediction labels for each metric.
    % coloumn corresponds to metric.
    % dimension is 60x<number of metrics> for now....
    metric_prediction_labels = zeros(size(test_set,1), size(metrics_nameList,2));            

    % k-value for each neighbour
    k_values = 1:3:30;
    Error_list = ones(size(metrics_nameList,2),size(k_values,2));

		% find best accuracy among all metrics for a given k.
    best_accuracy = ones(1,size(k_values,2));
    best_metric = string(k_values);  % probably a bad way of initializing this .. TODO:change later
    
    
    
    % For each k
        % For each distance metric, 
            % Perform nearest neighbour classification. Store prediction labels.
            % Find the classification error on TEST set.
            % Choose distance metric with least classification error.
    for j = 1:size(k_values,2)
        for i = 1:size(metrics_nameList,2)
   
            metric_name = char(metrics_nameList(i));
        
            Mdl = fitcknn(train_set,train_label,'NumNeighbors',k_values(j),'Standardize',1,...
                    'Distance',metric_name,'NSMethod',knn_method) ; 

            [label,score,cost] = predict(Mdl,test_set);
            metric_prediction_labels(:,i) = label;
            
           
        end
        
        % Find binary classification error for all metrics for a given k.
        binary_class_error = findClassificationError(metric_prediction_labels,test_label);
        Error_list(:,j) = binary_class_error;
        
    end
    
    % For each k, find metric with lowest classification error
    % and save results. The find the metric 
   
    for i = 1:size(k_values,2)
    
        binary_class_error = Error_list(:,i);
        [lowest_class_error,idx] = min(binary_class_error);
        best_metric(i) = metrics_nameList(idx);
        best_accuracy(i) = (1-lowest_class_error)*100;
    end
    [final_accuracy,idx] = max(best_accuracy);
    final_metric = best_metric(idx);
    fprintf(' (Best Metric, Best accuracy) = (%s,%f) found at k = %d',final_metric,final_accuracy,k_values(idx));
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
function plotGraph_Scatter(k_values,metrics_nameList,Error_list)
    
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
    
    
    
    %yticks([[0:0.2:1],[2:10:70]]);
    xticks(k_values);
    xlabel('k values');
    ylabel('binary classification error');
    title('Classification Error for distance metrics');
    legend(Legend);
    
end



% Creates a normal plot with each metric as series plot
% For each value of k.
function plotGraph(k_values,metrics_nameList,Error_list)
    
    total_metrics = size(metrics_nameList,2);
    Legend = cell(total_metrics,1);
    
   
        for i = 1:total_metrics
            plot(k_values,Error_list(i,:));
            
            Legend{i} = char(metrics_nameList(i)) ;
        
            % used to create series plot.
            hold on 
        end
        
    
    %yticks([[0:0.2:1],[2:10:70]]);
    xticks(k_values);
    xlabel('k values');
    ylabel('binary classification error');
    title('Classification Error for distance metrics');
    legend(Legend);
    
end


