% COMPUTELAYER Compute the output of a NN layer
%   This function compute the output of a single Neural Network layer
%

%   Copyright (c) 2011 Roberto Calandra
%   $Revision: 0.11 $

%   TODO: Implement linear transfer function


function [output,allclass,probclass] = computelayer(inputs,Weights,typelayer)
 %% Input validation
 
assert(size(Weights,1)==size(inputs,2),...
    'Weights and input dimensions do not match')


%% Init

allclass    = [];
probclass   = [];
ninputs     = size(inputs,1);
noutputs    = size(Weights,2);


%% Compute output

switch typelayer
    case {'linear'}
        error('Not yet implemented')
    case {'log-sigmoid'}
        output=1./(1+exp(-inputs*Weights));
    case {'tan-sigmoid'}
        v=inputs*Weights; 
        t1 = exp(-v);
        t2 = exp(v);
        output=(t2-t1)./(t2+t1);
        %output=tanh(inputs*Weights);
        %out{i}=tansig(out{i-1}*nn.W{i-1});
    case {'softmax'}
        v=inputs*Weights; 
        t1=exp(v);
        output=t1./repmat(sum(t1,2),1,noutputs);
    case {'class'}
        [probclass, allclass] = sort(inputs.*repmat(Weights,1,ninputs)',...
            2,'descend');
        output=allclass(:,1);
end


%% Output validation

assert(~any(any(isnan(output))),'Numerical instability led to a NaN value')


end
