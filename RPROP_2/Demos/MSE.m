% MSE Compute the Mean Square Error (MSE)
%   Compute the Mean Square Error (MSE)
%

%   Copyright (c) 2011 Roberto Calandra
%   $Revision: 0.12 $


function [mse] = MSE(inputs,targets)

assert(size(inputs,1)==size(targets,1),...
    'Number of inputs and targets samples do not match')
assert(size(inputs,2)==size(targets,2),...
    'Number of inputs and targets dimensions do not match')

mse = sum(sum((inputs - targets).^2,2)/2);

end