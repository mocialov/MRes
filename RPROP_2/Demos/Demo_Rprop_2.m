% DEMO_RPROP_2 All 4 Rprop on MNIST-small dataset
%   This Demo train a Neural Network on the MNIST-small dataset, using 
%   the 4 different Rprop methods.
%

%   Copyright (c) 2011 Roberto Calandra
%   $Revision: 0.60 $



function Demo_Rprop_2(batch)

if nargin<1, batch=0; end

%% Load Data

% Load the MNIST-small dataset
load rob_mnist_small

% Shuffle data
[train_in ti]   = shuffledata(train_in);
train_out       = shuffledata(train_out,ti);
[test_in to]    = shuffledata(test_in);
test_out        = shuffledata(test_out,to);

% Convert labels to the internal class system used in the package
[nn train_out]  = RNN.lab2class(train_out);
[nn test_out]   = RNN.lab2class(test_out,nn);

clear ti to


%% Init Network

% Declare the structure of the network
ndim = size(train_in,2);
neurons = [ndim,300,200,nn.nlabels,1];

% Declare the Transfer function 
% (Optional: will override parameter in init_in)
nn.o.init_nn.tf = 'log-sigmoid'; % 'tan-sigmoid'

% Create and initialize the Network
nn = RNN.init_nn(neurons,nn);

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
nn.o.opt_rprop.MaxIter      = 30;

% Desired MSE for the Training
% (Optional: will override parameter in opt_rprop)
nn.o.opt_rprop.dmse         = 1.0e-3;


%% Train Network

% (Optional: will override parameter in opt_rprop)
nn.o.opt_rprop.method = 'Rprop-';
% Train the network
% (Optional: test_in and test_out are not required, but allow to compute
% more statistics of the training process)
[nn1 error1] = RNN.opt_rprop(nn,train_in,train_out,test_in,test_out);

% (Optional: will override parameter in opt_rprop)
nn.o.opt_rprop.method = 'Rprop+';
% Train the network
% (Optional: test_in and test_out are not required, but allow to compute
% more statistics of the training process)
[nn2 error2] = RNN.opt_rprop(nn,train_in,train_out,test_in,test_out);

% (Optional: will override parameter in opt_rprop)
nn.o.opt_rprop.method = 'IRprop-';
% Train the network
% (Optional: test_in and test_out are not required, but allow to compute
% more statistics of the training process)
[nn3 error3] = RNN.opt_rprop(nn,train_in,train_out,test_in,test_out);

% (Optional: will override parameter in opt_rprop)
nn.o.opt_rprop.method = 'IRprop+';
% Train the network
% (Optional: test_in and test_out are not required, but allow to compute
% more statistics of the training process)
[nn4 error4] = RNN.opt_rprop(nn,train_in,train_out,test_in,test_out);


%% Evaluate Results

if batch
else
    % Plot MSE during training
    figure()
    i1 = plot(error1.mse);
    hold all
    i2 = plot(error2.mse);
    i3 = plot(error3.mse);
    i4 = plot(error4.mse);
    legend([i1 i2 i3 i4],'Rprop-','Rprop+','IRprop-','IRprop+',...
        'Location','NorthEast')
    title('MSE during training (on the train set)')
    xlabel('Number of Iterations')
    hold off
    
    % Plot accuracy during training (on the train set)
    figure()
    i1 = plot(error1.accuracy);
    hold all
    i2 = plot(error2.accuracy);
    i3 = plot(error3.accuracy);
    i4 = plot(error4.accuracy);
    legend([i1 i2 i3 i4],'Rprop-','Rprop+','IRprop-','IRprop+',...
        'Location','SouthEast')
    title('Classification accuracy during training (on the train set)')
    xlabel('Number of Iterations')
    hold off
    
    % Plot accuracy during training (on the test set)
    figure()
    i1 = plot(error1.test.accuracy);
    hold all
    i2 = plot(error2.test.accuracy);
    i3 = plot(error3.test.accuracy);
    i4 = plot(error4.test.accuracy);
    legend([i1 i2 i3 i4],'Rprop-','Rprop+','IRprop-','IRprop+',...
        'Location','SouthEast')
    title('Classification accuracy during training (on the test set)')
    xlabel('Number of Iterations')
    hold off
end


end

