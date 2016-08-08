% COMPUTENETWORK Compute the output of a NN
%   This function compute the output of a complete Neural Network.
%   'fullout' is a flag (default 0), that when activated, override the
%   default output, and return all the values computed for each layer,
%   inside the network
%

%   Copyright (c) 2011 Roberto Calandra
%   $Revision: 0.11 $


function [output] = computenetwork(nn,inputs,fullout)
%% Input validation

if nargin<3, fullout=0; end

assert(nn.neurons(1)==size(inputs,2),...
    'Number of neurons in the first layer and input dimensions do not match')


%% Init

ninputs=size(inputs,1);
out=cell(nn.nlayers,1);
out{1}=[inputs ones(ninputs,1)];


%% Compute Output

% TODO include the bias in computelayer, to make it easier?
for i=2:nn.nlayers
    % 'Class' layers or not?
    if ~strcmp(nn.typelayer{i-1},'class')
        % If not, Bias has to be added
        if ~strcmp(nn.typelayer{i},'class')
            % If next layer is not 'Class', we already add the next bias
            out{i}=[computelayer(out{i-1},[nn.W{i-1};nn.B{i-1}],nn.typelayer{i-1}) ones(ninputs,1)];
        else
            out{i}=computelayer(out{i-1},[nn.W{i-1};nn.B{i-1}],nn.typelayer{i-1});
        end
    else
        out{i}=computelayer(out{i-1},nn.W{i-1},nn.typelayer{i-1});
    end
    
end


if fullout
    output=out;
else
    output=out{nn.nlayers};
end


end
