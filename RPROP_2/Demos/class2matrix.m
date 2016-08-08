% CLASS2MATRIX Convert classes to a binary matrix
%   Convert classes to a binary matrix
%

%   Copyright (c) 2011 Roberto Calandra
%   $Revision: 0.10 $ 


function [out] = class2matrix(in,n_classes)

n_in = size(in,1);

if nargin<2
     % No number of classes passed
     error('missing argument')
%     n_classes=max(inputs);
 end

out = zeros(n_in,n_classes);
for i = 1:n_in
    out(i,in(i)) = 1;
end

end

