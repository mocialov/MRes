% INIT_NN Initialize a NN
%   Initialize a NN where DBN and Rprop optimization are applied to all the
%   layers of the network
%

%   Copyright (c) 2011 Roberto Calandra
%   $Revision: 0.12 $


function [nn] = init_nn(size_layer,nn)
%%

assert(isfield(nn,'nlabels'))
assert(isfield(nn,'labels'))


%% Parameters

p.tf = 'log-sigmoid';
%p.tf = 'tan-sigmoid';

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


%% Create network structure

nn.neurons  = size_layer;
nn.nlayers  = size(nn.neurons,2);
nn.nl_regr  = nn.nlayers-1;
nn.nl_opt   = 1:nn.nlayers-2;
nn.nl_dbn   = 1:nn.nlayers-3;
nn.nl_class = 1;

% Initialize Weights
for i=1:nn.nlayers-2
    % p.b_up    = 1;
    % p.b_down  = -1;
    % nn.W{i}=RNN.rrand([nn.neurons(i) nn.neurons(i+1)],p.b_up,p.b_down);
    w_scale     = 1/sqrt(nn.neurons(i)+nn.neurons(i+1));
    nn.W{i}     = 2*w_scale*(rand([nn.neurons(i) nn.neurons(i+1)])-0.5);
    % nn.B{i}=RNN.rrand([nn.neurons(i+1) 1],p.b_up,p.b_down)';
    % nn.Bh{i}=RNN.rrand([nn.neurons(i+1) 1],p.b_up,p.b_down)';
    nn.B{i}     = zeros([1 nn.neurons(i+1)]);
    nn.Bh{i}    = zeros([1 nn.neurons(i)]);
    nn.typelayer{i} = p.tf;
end

%nn.typelayer{nn.nlayers-2}='softmax';

i=nn.nlayers-1;
nn.W{i}=ones(nn.neurons(i),nn.neurons(i+1));
nn.typelayer{nn.nlayers-1}='class';


end

