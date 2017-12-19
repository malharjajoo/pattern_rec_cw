function [] = basic_generated_script(train_set,train_labels,test_set,test_labels)

setdemorandstream(391418381);
x = train_set;
t = train_labels;

% Choose a Training Function
% For a list of all training functions type: doc nntrain
% 'trainlm' is usually fastest.
% 'trainbr' takes longer but may be better for challenging problems.
% 'trainscg' uses less memory. Suitable in low memory situations.


% Create a Pattern Recognition Network
hidden_layersizes = [4,6,10];
training_Fcns = ["trainscg","trainlm","trainbr","traingdm"]; 
performance_Fcns = ["mse","crossentropy"];

 
hiddenLayerSize = hidden_layersizes(2);
trainFcn = char(training_Fcns(2));  
performFcn = char(performance_Fcns(1));

net = patternnet(hiddenLayerSize, trainFcn,performFcn);
net.trainParam.mc = 0.9;

% Setup Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 30/100;
net.divideParam.testRatio = 0;

% Train the Network
[net,tr] = train(net,x,t);

% Test the Network
y = net(test_set);
t = test_labels;
e = gsubtract(t,y);

displayPerformance(net,t,y);


% view(net)  ; % View the Network

plotData(tr,e,t,y);

end



% =============== Helper functions ===============

% separated into a function because idea is to 
% train on a particular performance funciton but
% display reduction across different performance measures ?
function [] = displayPerformance(net,t,y)
    %performance = crossentropy(net,t,y)
    %performance = mse(net,t,y)
    performance = perform(net,t,y);
    
    tind = vec2ind(t);
    yind = vec2ind(y);
    accuracy = (sum(tind == yind)/numel(tind)) * 100 ;

    fprintf('performance metric %s = %f\n',net.performFcn, performance);
    fprintf('accuracy = %f\n %%',accuracy);
end


function [] = plotData(tr,e,t,y)
    
   % Plots
    % Uncomment these lines to enable various plots.
    figure, plotperform(tr)
    %figure, plottrainstate(tr)
    %figure, ploterrhist(e)
    %figure, plotconfusion(t,y)
    %figure, plotroc(t,y) 
end
