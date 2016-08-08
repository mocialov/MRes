% OPT_RPROP Optimize a NN using Rprop
%   [NN,STATS] = OPT_RPROP(NN,TRAIN_IN,TRAIN_OUT) return a Neural Network
%   trained using Rprop on the Train set.
%
%   [NN,STATS] = OPT_RPROP(NN,TRAIN_IN,TRAIN_OUT,TEST_IN,TEST_OUT) also
%   perform analysis on the Test set for every iteration of the training
%   process.
%

%   Copyright (c) 2011 Roberto Calandra
%   $Revision: 0.53 $


function [nn,stats] = opt_rprop(nn,train_in,train_out,test_in,test_out)
%% Input validation

if nargin==5
    testmode = 1;
else
    testmode = 0;
end

assert(size(train_in,1)==size(train_out,1),...
    'Number of train_in and train_out samples do not match')
assert(size(train_in,2)==nn.neurons(1),...
    'Number of neurons in the first layer and input dimensions do not match')
assert(size(train_out,2)==nn.neurons(end),...
    'Number of neurons in the last layer and output dimensions do not match')

if testmode
    assert(size(test_in,1)==size(test_out,1),...
        'Number of test_in and test_out samples do not match')
    assert(size(test_in,2)==nn.neurons(1),...
        'Number of neurons in the first layer and input dimensions do not match')
    assert(size(test_out,2)==nn.neurons(end),...
        'Number of neurons in the last layer and output dimensions do not match')
end


%% Parameters

% Default Parameters
p.verbose       = 0;            % [0-3] Verbose mode
p.display       = 0;            % [0-1] Plot training process
p.method        = 'IRprop-';    % Rprop method used
p.MaxIter       = 50;           % Maximum number of iterations
p.decay         = 0.01;         % [0-1] Weight decay
p.mu_neg        = 0.5;          % Decrease factor
p.mu_pos        = 1.2;          % Increase factor
p.delta0        = 0.0123;       % Initial update-value
p.delta_min     = 0;            % Lower bound for step size
p.delta_max     = 50;           % Upper bound for step size
p.dmse          = 0.01;         % Desired MSE

% Use passed Parameters
if isfield(nn,'o')
    func_name = mfilename;
    if isfield(nn.o,func_name)
        t_p = fieldnames(nn.o.(func_name));
        for i = 1:size(t_p,1)
            p.(t_p{i}) = nn.o.(func_name).(t_p{i});
        end
    end
    clear func_name t_p i
end


%% Initialization

plus = sum(strcmp(p.method,{'Rprop+','IRprop+'}));

n_sample                = size(train_in,1);
n_output                = size(train_out,2);

stats.mse               = zeros([p.MaxIter,1]);
stats.n_correct         = zeros([p.MaxIter,1]);
stats.accuracy          = zeros([p.MaxIter,1]);
stats.class_acc         = zeros([p.MaxIter,nn.nlabels]);

delta                   = cell(nn.nlayers-1,1);
grad                    = cell(nn.nlayers-1,1);
old_grad                = cell(nn.nlayers-1,1);
deltaW                  = cell(nn.nlayers-1,1);

if plus
    old_deltaW          = cell(nn.nlayers-1,1);
end

for i = 1:nn.nl_regr-1 %nn.nl_opt
    
    tb = size(nn.W{i})+[1 0];   % Add the Bias
    
    delta{i}            = repmat(p.delta0,tb);
    grad{i}             = zeros(tb);
    old_grad{i}         = zeros(tb);
    deltaW{i}           = zeros(tb);
    
    if plus
        old_deltaW{i}   = zeros(tb);
        old_E           = inf;
    end
end

% Convert output classes to binary matrix
D = class2matrix(train_out,nn.nlabels);

if testmode
    n_sample_test           = size(test_in,1);
    stats.test.mse          = zeros([p.MaxIter,1]);
    stats.test.n_correct    = zeros([p.MaxIter,1]);
    stats.test.accuracy     = zeros([p.MaxIter,1]);
    stats.test.class_acc    = zeros([p.MaxIter,nn.nlabels]);
    D_test = class2matrix(test_out,nn.nlabels);
end


%% Training

if p.verbose>0
    fprintf(['Training Network with ' p.method '\n']);
end

if p.display
    h = figure();
end

for Iter = 1:p.MaxIter
    
    if p.verbose>1
        indent(2)
        fprintf('Iter %d (of %d)\n',Iter,p.MaxIter);
    end
    
    % Compute network output
    a = computenetwork(nn,train_in,1);
    
    err = (D-a{nn.nl_regr});
    E   = MSE(D,a{nn.nl_regr})/(n_sample*n_output);
    
    if testmode
        a_test = computenetwork(nn,test_in,1);
        E_test = MSE(D_test,a_test{nn.nlayers-1})/(n_sample_test*n_output);
    end
    
    % Compute gradients
    for i = nn.nl_regr:-1:2
        switch nn.typelayer{i-1}
            case {'log-sigmoid'}
                v = a{i}.*(1-a{i}).*err;
            case {'tan-sigmoid'}
                v = (1+a{i}).*(1-a{i}).*err;
            otherwise
                error('Unknown transfer function')
        end
        t_grad = (-v'*a{i-1})';
        if i<nn.nl_regr
            grad{i-1} = t_grad(:,1:end-1);
        else
            grad{i-1} = t_grad;
        end
        if i>1
            if i<nn.nl_regr
                err = v(:,1:end-1)*[nn.W{i-1};nn.B{i-1}]';
            else
                err = v*[nn.W{i-1};nn.B{i-1}]';
            end
        end
    end
    
    % Update weights
    for i = nn.nl_opt
        
        % Add the weight decay
        if p.decay
            grad{i} = grad{i} + p.decay.*[nn.W{i}; nn.B{i}];
        end
        
        gg = grad{i}.*old_grad{i};
        delta{i} =  min(delta{i}*p.mu_pos,p.delta_max).*(gg>0) +...
            max(delta{i}*p.mu_neg,p.delta_min).*(gg<0) +...
            delta{i}.*(gg==0);
        
        switch p.method
            case 'Rprop-'
                deltaW{i}       = -sign(grad{i}).*delta{i};
                
            case 'Rprop+'
                deltaW{i}       = -sign(grad{i}).*delta{i}.*(gg>=0) -...
                    old_deltaW{i}.*(gg<0);
                grad{i}         = grad{i}.*(gg>=0);
                old_deltaW{i}   = deltaW{i};
                
            case 'IRprop-'
                grad{i}         = grad{i}.*(gg>=0);
                deltaW{i}       = -sign(grad{i}).*delta{i};
                
            case 'IRprop+'
                deltaW{i}       = -sign(grad{i}).*delta{i}.*(gg>=0) -...
                    old_deltaW{i}.*(gg<0)*(E>old_E);
                grad{i}         = grad{i}.*(gg>=0);
                old_deltaW{i}   = deltaW{i};
                old_E           = E;
                
            otherwise
                error('Unknown method')
                
        end
        
        old_grad{i} = grad{i};
        nn.W{i}     = nn.W{i} + deltaW{i}(1:end-1,:);
        nn.B{i}     = nn.B{i} + deltaW{i}(end,:);
        
        if p.verbose>2
            indent(3)
            fprintf('Update weights layer %d: %d-%d\n',...
                i,nn.neurons(i),nn.neurons(i+1))
        end
        
    end
    
    % Compute statistics
    train = class_stat(a{nn.nlayers},train_out,nn.nlabels);
    stats.mse(Iter)                 = E;
    stats.n_correct(Iter)           = train.n_correct;
    stats.accuracy(Iter)            = train.accuracy;
    stats.class_acc(Iter,:)         = train.class_acc;
    
    if testmode
        test = class_stat(a_test{nn.nlayers},test_out,nn.nlabels);
        stats.test.mse(Iter)            = E_test;
        stats.test.n_correct(Iter)      = test.n_correct;
        stats.test.accuracy(Iter)       = test.accuracy;
        stats.test.class_acc(Iter,:)    = test.class_acc;
    end
    
    % Plot training process
    if p.display>0
        set(0,'CurrentFigure',h);
        if testmode
            plot(1:Iter,[stats.accuracy(1:Iter) stats.test.accuracy(1:Iter)]);
            legend('Training set','Test set','Location','SouthEast')
        else
            plot(stats.accuracy(1:Iter));
            %legend('Training set','Location','SouthEast')
        end
        title('Classification accuracy during training')
        xlabel('Number of Iterations')
        drawnow
    end
    
    if E < p.dmse
        if p.verbose>1
            indent(2)
            fprintf(2,'Stopping criterion reached (MSE < desired MSE)\n')
        end
        break
    end
    
end


%%

stats.mse               = stats.mse(1:Iter);
stats.n_correct         = stats.n_correct(1:Iter);
stats.accuracy          = stats.accuracy(1:Iter);
stats.class_acc         = stats.class_acc(1:Iter,:);

if testmode
    stats.test.mse          = stats.test.mse(1:Iter);
    stats.test.n_correct    = stats.test.n_correct(1:Iter);
    stats.test.accuracy     = stats.test.accuracy(1:Iter);
    stats.test.class_acc    = stats.test.class_acc(1:Iter,:);
end


end

