% CLASS2LAB Convert RNN classes back to labels
%   Before training a NN all the labels from the dataset are converted to
%   internal classes that are used during the optimization. This function
%   convert back classes to the equivalent labels
%

%   Copyright (c) 2011 Roberto Calandra
%   $Revision: 0.11 $


function [labels] = class2lab(nn,class)

n_class=size(class,1);
labels=zeros([n_class 1]);

for i=1:n_class
    for x=1:nn.nlabels
        if class(i)==x
            labels(i)=nn.labels{x};
        end
    end
end


end