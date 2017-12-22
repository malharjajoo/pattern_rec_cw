function [best_metric_k, highest_accuracy_k] = nearest_neighbour(train_set,train_label,test_set,test_label)

    knn_method =  'exhaustive'; %'kdtree';
    
    metrics_nameList = [];
    % 9 metrics
  
    if(strcmp(knn_method,'exhaustive'))
        metrics_nameList = ["cityblock","chebychev","correlation",...
                         "cosine","euclidean","chi_square",...
                         "jaccard","mahalanobis","minkowski"];                
    else 
        % since matlab only allows following distance metrics with kdtree method.
        % for now, kdtree cannot be used with this dataset since attributes > 10.
        % however, for some other data ....
        metrics_nameList = ["cityblock","euclidean","minkowski"];
    end
    

    
                     
    % row contains prediction labels for each metric.
    % coloumn corresponds to metric.
    % dimension is 60x<number of metrics> for this test set....
    metric_prediction_labels = zeros(size(test_set,1), size(metrics_nameList,2));            

    % k-value for each neighbour
    k_values =  1:3:118; 
    Error_list = ones(size(metrics_nameList,2),size(k_values,2));

    % ========== Testing Phase ( There is NO training phase! ) ========
    % For each k
        % For each distance metric, 
            % Perform nearest neighbour classification. Store prediction labels.
            % Find the classification error on TEST set.
            % Choose distance metric with least classification error.
    w = [1;1; 1; 1; 1; 1; 1; 0.01; 1; 1; 0.01 ;1 ; 10];
    chiSqrDist = @(x,Z,wt)sqrt((bsxfun(@minus,x,Z).^2)*wt);
    
    Md1 = [];
    for j = 1:size(k_values,2)
        for i = 1:size(metrics_nameList,2)
   
            if( strcmp(metrics_nameList(i),"chi_square"))
                 metric_name = @(x,Z)chiSqrDist(x,Z,w) ;
            else
                 metric_name = char(metrics_nameList(i));
            end
            
            
            if( strcmp(metrics_nameList(i),"minkowski"))
                Mdl = fitcknn(train_set,train_label,'NumNeighbors',k_values(j),'Standardize',1,...
                    'Distance',metric_name,'DistanceWeight','equal','NSMethod',knn_method,'Exponent',3) ; 
                
            else
                Mdl = fitcknn(train_set,train_label,'NumNeighbors',k_values(j),'Standardize',1,...
                    'Distance',metric_name,'DistanceWeight','equal','NSMethod',knn_method) ; 
            end
            
            % tic;
            [label,~,~] = predict(Mdl,test_set);
            % toc;
            metric_prediction_labels(:,i) = label;
            
           
        end
        
        % Find binary classification error for all metric for a given k.
        Error_list(:,j)  = findClassificationError(metric_prediction_labels,test_label);

    end
    
    % For a given "k", store best accuracy found and which metric it corresponds to.
    highest_accuracy_k = ones(1,size(k_values,2));
    % probably a bad way of initializing this .. TODO:change later
    best_metric_k = string(k_values);
    
    
     % For a given "k", store best accuracy found and which metric it corresponds to.
    lowest_accuracy_k = ones(1,size(k_values,2));
    % probably a bad way of initializing this .. TODO:change later
    worst_metric_k = string(k_values);
      
    % For each k, find metric with lowest(best) and highest(worst) classification error
    % and save results.
    for i = 1:size(k_values,2)
    
        [lowest_error_k,idx] = min(Error_list(:,i));
        best_metric_k(i) = metrics_nameList(idx);
        highest_accuracy_k(i) = (1-lowest_error_k)*100;
        
        [highest_error_k,idx] = max(Error_list(:,i));
        worst_metric_k(i) = metrics_nameList(idx);
        lowest_accuracy_k(i) = (1-highest_error_k)*100;
    end
   
    
     [best_accuracy,idx1] = max(highest_accuracy_k);
     best_metric = best_metric_k(idx1);
    
     [worst_accuracy,idx2] = min(lowest_accuracy_k);
     worst_metric = worst_metric_k(idx2);
    
    fprintf(' (Best Metric, Best accuracy) = (%s,%f) found at k = %d\n',best_metric,best_accuracy,k_values(idx1));
    fprintf(' (Worst Metric, Worst accuracy) = (%s,%f) found at k = %d\n',worst_metric,worst_accuracy,k_values(idx2));
    % ====== Plot all results ==========

    % plot (Binary) classification error ?
    plotGraph(k_values,metrics_nameList,Error_list);
    
end



% ================= Helper functions ===========================


function [] = displayBestFiveResults(accuracy_k,metric_k,k_values)
    
     % Get 5 best metrics ... across all k 
     [s_accuracy,originalpos] = sort( accuracy_k, 'descend' );
     n = s_accuracy(1:5);
     p = originalpos(1:5) ;
     
     for i = p
          fprintf(' (Best Metric, Best accuracy) = (%s,%f) found at k = %d\n',metric_k(i),n(i),k_values(i));
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
            
           if( strcmp(metrics_nameList(i),"cityblock"))
                Legend{i} = 'manhattan' ;
                
            else
                 Legend{i} = char(metrics_nameList(i)) ;
           end
        
            % used to create series plot.
            hold on 
        end
        
    
    %yticks([[0:0.2:1],[2:10:70]]);
    
    xlabel('no. of nearest neighbours(k)');
    ylabel('Test (classification) error/100');
    title('Test Error for distance metrics');
    legend(Legend);
    
end


