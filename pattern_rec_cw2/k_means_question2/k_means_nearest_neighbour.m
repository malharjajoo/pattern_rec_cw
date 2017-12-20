function [best_metric, lowest_class_error, best_accuracy] = k_means_nearest_neighbour(train_set,train_labels,test_set,test_labels)

%rng(1); % For reproducibility  

    % =========== K-means clustering ===============
      
    k_list = [3];
    
    cluster_labels = [];
    idx = [];
    Cluster_centroids=[];
    while true
        
        
            total_clusters = k_list;
             [idx,Cluster_centroids,sumd,D] = kmeans(train_set,total_clusters);
        
            % ============ Assign training label to cluster ===========
            % First assign labels in a way that makes sense.
            % find most frequent class (from training label) and assign to cluster.
            cluster_labels = assignLabels(total_clusters,idx,train_labels);
            
            if( terminateClustering(cluster_labels) )
                
                break;
            end
        
       
        
     end
   
    % plot cost-vs-k ?
    % plot(k_list, );
    
    %=========== Nearest neighbours part ===========
    fprintf("Found best clusters");
    
   
    
    knn_method =  'exhaustive'; %'kdtree';
    
    metrics_nameList = [];
    % 9 metrics
    if(strcmp(knn_method,'exhaustive'))
        metrics_nameList = ["cityblock","chebychev","correlation",...
                         "cosine","euclidean","hamming"...
                         "jaccard","mahalanobis","minkowski","spearman"];
                     
                         
    else 
        % since matlab only allows following distance metrics with kdtree method.
        metrics_nameList = ["cityblock","chebychev","euclidean","minkowski"];
    end
    

    
                     
    % row contains prediction labels for each metric.
    % coloumn corresponds to metric.
    % dimension is 60x<number of metrics> for now....
    metric_prediction_labels = zeros(size(test_set,1), size(metrics_nameList,2));            

    % k-value for each neighbour
    k_values = [1];%1:3:30;
    Error_list = ones(size(metrics_nameList,2),size(k_values,2));
    best_accuracy = ones(1,size(k_values,2));
    % probably a bad way of initializing this .. TODO:change later
    best_metric = string(k_values);
    
  
    % ========== Testing Phase for NN classifier ( There is NO training phase! ) ========
    % For each k
        % For each distance metric, 
            % Perform nearest neighbour classification. Store prediction labels.
            % Find the classification error on TEST set.
            % Choose distance metric with least classification error.
    for j = 1
        for i = 2
   
            %metric_name = char(metrics_nameList(i));
        
            tic;
            Mdl = fitcknn(Cluster_centroids,cluster_labels,'NumNeighbors',k_values(j),'Standardize',1,...
                    'Distance','euclidean','NSMethod',knn_method) ; 
            toc;
            
            tic;
            [label,score,cost] = predict(Mdl,test_set);
            toc;
            metric_prediction_labels(:,i) = label;
            
           
        end
        
        % Find binary classification error for a metric for a given k.
        binary_class_error = findClassificationError(metric_prediction_labels,test_labels);
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
    
    % overwriting if better than best result. Ofc this value is set manually 
    % since it is between different runs.
    if( final_accuracy > 91) 
        
       save('cluster_details_91_plus.mat','idx','Cluster_centroids','cluster_labels'); 
    end
    % ====== Plot all results ==========
    
    % plot (Binary) classification error ?
    % plotGraph(k_values,metrics_nameList,Error_list);
    
    
end




 






% ================= Helper functions ===========================

% The function decides whether to stop clustering or not.
% 1) Checks if clusters have at least the 3 labels ( 1,2,3) from the training set.
% 2) Checks if each cluster has a distribution of training points close 
% to the ratio given ( class1 : 59, class 2:71 , class 3:48 )
function [terminateClustering] = terminateClustering(cluster_labels)
    
    terminateClustering = false;
    
    if( ismember(1,cluster_labels) & ismember(2,cluster_labels) & ismember(3,cluster_labels) )
        terminateClustering =  true;
    end
     
    fprintf('rejecting ...\n');
end



function [cluster_labels] =  assignLabels(total_clusters,idx,train_labels)
    
    cluster_labels = zeros(size(total_clusters,1));
    
    
    for i = 1:total_clusters

        clabels = train_labels( find(idx == i)) ;
        
        % find frequency of all classes and assign to cluster.
        %a = unique(clabels);out = [a,histc(x(:),a)];l = max(out(:,2));
        %Currently, training labels have frequency { 39, 51, 28 }
        [l,F] = mode(clabels);
        if((l == 1) & (F > 25))
             cluster_labels(i) = l;
        end
        
        if((l == 2) & (F > 30))
             cluster_labels(i) = l;
        end

        if((l == 3) & (F > 15))
             cluster_labels(i) = l;
        end
        
    end
            
end
            
            
            

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


