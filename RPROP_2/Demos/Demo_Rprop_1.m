% DEMO_RPROP_1 IRprop- on MNIST-small dataset
%   This Demo train a Neural Network on the MNIST-small dataset
%

%   Copyright (c) 2011 Roberto Calandra
%   $Revision: 0.60 $



function Demo_Rprop_1(batch)

if nargin<1, batch=0; end

%% Load Data

% Load the MNIST-small dataset
load rob_mnist_small

% Shuffle data
%[train_in ti]   = shuffledata(train_in);
%train_out       = shuffledata(train_out,ti);
%[test_in to]    = shuffledata(test_in);
%test_out        = shuffledata(test_out,to);

[train_in ti] = shuffledata(dlmread('/afs/inf.ed.ac.uk/user/s15/s1581976/Downloads/ChaLearn_DS_modified/results/training_data.txt'));
train_out = shuffledata(dlmread('/afs/inf.ed.ac.uk/user/s15/s1581976/Downloads/ChaLearn_DS_modified/results/training_classes.txt'),ti);

[test_in to] = shuffledata(dlmread('/afs/inf.ed.ac.uk/user/s15/s1581976/Downloads/ChaLearn_DS_modified/results/testing_data.txt'));
test_out = shuffledata(dlmread('/afs/inf.ed.ac.uk/user/s15/s1581976/Downloads/ChaLearn_DS_modified/results/testing_classes.txt'),to);

% Convert labels to the internal class system used in the package
[nn train_out_c]  = lab2class(train_out);
[nn test_out_c]   = lab2class(test_out,nn);

clear ti to


%% Init Network

% Declare the structure of the network
ndim = size(train_in,2);
neurons = [ndim,300,300,nn.nlabels,1];  %%can add more, like [ndim,300,200,nn.nlabels,1];

% Declare the Transfer function 
% (Optional: will override parameter in init_in)
nn.o.init_nn.tf = 'tan-sigmoid'; % 'tan-sigmoid'  'log-sigmoid' 

% Create and initialize the Network
nn = init_nn(neurons,nn);

disp(nn);

% (Optional: will override parameter in opt_rprop)
if batch
    nn.o.opt_rprop.display      = 0;
    nn.o.opt_rprop.verbose      = 0;
else
    nn.o.opt_rprop.display      = 1;
    nn.o.opt_rprop.verbose      = 2;
end

% Maximum number of Iterations for the Training
% (Optional: will override parameter in opt_rprop)
nn.o.opt_rprop.MaxIter      = 500;

% Desired MSE for the Training
% (Optional: will override parameter in opt_rprop)
nn.o.opt_rprop.dmse         = 1.0e-3;


%% Train Network

% Train the network
% (As default IRprop- is used in opt_rprop)
% (Optional: test_in and test_out are not required, but allow to compute
% more statistics of the training process)
[nn1 error1] = opt_rprop(nn,train_in,train_out_c,test_in,test_out_c);

% Classify the test set
output_c = computenetwork(nn1,test_in);


% Convert from the internal class system back to labels
output = class2lab(nn,output_c);


disp(output);
disp('');
disp(test_out); %the actual output from out nodes

% Compute statistics
stats = class_stat(output,test_out,10);


%% Evaluate Results

if batch
else

    fprintf('Test set accuracy: %f\n',stats.accuracy)
    
    % Plot the confusion matrix
    fprintf('Confusion Matrix\n')
    %disp(stats.confusion)
    stats.confusion = stats.confusion - min(stats.confusion(:));
    stats.confusion = stats.confusion ./ max(stats.confusion(:));
    disp(stats.confusion)
    
end

%disp(nn.W{1});

end

