function [five_best_Params] = basic_generated_script(train_set,train_labels,test_set,test_labels)

warning off; % this is required to prevent trainbr from issuing warning about changing reg param.
%setdemorandstream(391418381);

% Choose a Training Function
% For a list of all training functions type: doc nntrain
% 'trainlm' is usually fastest.
% 'trainbr' takes longer but may be better for challenging problems.
% 'trainscg' uses less memory. Suitable in low memory situations.

% Create a Pattern Recognition Network
hidden_layersizes = [4,6,10];
training_Fcns = ["trainscg","trainlm","trainbr","traingdm"]; 
repetitions = 10;
performance_Fcns = ["mse","crossentropy"];
regParam = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9];

% stores 5 Params for best result among several runs
% {training method,mse,accuracy,regParam,Matlab network object};
bestParams = {'default',-1,-1,-1,'store_net'};
five_best_Params = {bestParams;bestParams;bestParams;bestParams;bestParams};

% This is hardcoded but was when experimenting.
% These are the values that are found to be best.
hiddenLayerSize = hidden_layersizes(2); 
performFcn = char(performance_Fcns(1));


% ======== Main code ==========
for i = 1:size(training_Fcns,2)
  
    % Repetitions since backprop(train() below) begins with random weights
    % everytime.
	for k = 1:repetitions
		for j = 1:size(regParam,2)

			currentRegParam = regParam(j);
			trainFcn = char(training_Fcns(i));  

			% 1) Create the network 
			net = createNetwork(hiddenLayerSize,trainFcn,performFcn);

			% 2) Train the Network
			net.trainParam.showWindow = false; % avoid gui while training.
			net.performParam.regularization = currentRegParam;
			[net,tr] = train(net,train_set,train_labels);

			% 3) Test the Network
			y = net(test_set);

			[perf,accuracy] = displayPerformance(net,test_labels,y,'true');

            five_best_Params = updateFiveBestParams(five_best_Params,net,perf,accuracy,currentRegParam ) ; 
			%bestParams = updateParam(bestParams,net,perf,accuracy,currentRegParam);

		end
            fprintf('\n');
	end
 
        fprintf('end of %d repetitions for %s',repetitions,net.trainFcn);
end


% ========== Display result ============

fprintf('\nFinal result\n');


% view(net)  ; % View the Network

% plotData(tr,test_labels,y);

end








% =============== Helper functions ===============


% separated into a function because idea is to 
% train on a particular performance funciton but
% display reduction across different performance measures ?
function [performance,accuracy] = displayPerformance(net,t,y,displayThis)
    %performance = crossentropy(net,t,y)
    %performance = mse(net,t,y)
    performance = perform(net,t,y);
    
    tind = vec2ind(t);
    yind = vec2ind(y);
    accuracy = (sum(tind == yind)/numel(tind)) * 100 ;
    
    if(strcmp(displayThis,'true'))
        fprintf('(metric=val,accuracy)= (%s=%f,%f %%)\n',net.performFcn, performance,accuracy);
    end
    
end


function [] = plotData(tr,test_labels,y)
    
    e = gsubtract(test_labels,y);
   % Plots
    % Uncomment these lines to enable various plots.
    figure, plotperform(tr)
    %figure, plottrainstate(tr)
    %figure, ploterrhist(e)
    %figure, plotconfusion(t,y)
    %figure, plotroc(t,y) 
end


function [net] = createNetwork(hiddenLayerSize,trainFcn,performFcn)
    net = patternnet(hiddenLayerSize, trainFcn,performFcn);
 
    % Setup Division of Data for Training, Validation, Testing
    net.divideParam.trainRatio = 1;
    net.divideParam.valRatio = 0;
    net.divideParam.testRatio = 0;
end


% bestParams = {training method,performance,accuracy,regParam};
function [bestParams] = updateParam(bestParams,net,performance,accuracy,regParam)
   
    if(accuracy > bestParams{3})
        bestParams{1} = net.trainFcn;
        bestParams{2} = performance;
        bestParams{3} = accuracy;
        bestParams{4} = regParam;
        bestParams{5} = net;   
    end
end


% Function updates and stores 5 best values ( sorted according to test accuracy, but criteria can be changed)
% bestParams = {training method,performance,accuracy,regParam};
function [five_best_Params] = updateFiveBestParams(five_best_Params,net,performance,accuracy,regParam ) ; 
   
    len = size(five_best_Params,1);
    % Notice 
    % This stores the best 5 values.
    for i = 1:len
        % The structure of five_best_Params is {trainingFcn,performance,accuracy,regParam,networkObj};
         if(accuracy > five_best_Params{i}{3})
             % copy over values
            for j = i:len-1
                five_best_Params{j+1}{1} = five_best_Params{j}{1};
                five_best_Params{j+1}{2} = five_best_Params{j}{2};
                five_best_Params{j+1}{3} = five_best_Params{j}{3};
                five_best_Params{j+1}{4} = five_best_Params{j}{4};
                five_best_Params{j+1}{5} = five_best_Params{j}{5};
            end
            
            five_best_Params{i}{1} = net.trainFcn;
            five_best_Params{i}{2} = performance;
            five_best_Params{i}{3} = accuracy;
            five_best_Params{i}{4} = regParam;
            five_best_Params{i}{5} = net;   
            break;
           
            
         end
         
    end
   
end

