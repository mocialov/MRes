% LAB2CLASS Convert labels to RNN classes
%   This function convert an input vector of labels [N x 1]  to the
%   internal structure used in RNN. In particular it map every labels to a
%   sequence of positive scalars starting from 1. The function optionally
%   accept a  pre-existent structure NN that is eventually modified to
%   incorporate new labels. This function return the structure NN and a
%   vector of the re-mapped labels.
%

%   Copyright (c) 2011 Roberto Calandra
%   $Revision: 0.11 $

% TODO: implement the use of cell (also for labels/string)


function [nn,class] = lab2class(labels,nn)

n_labels=size(labels,1);
class=zeros([n_labels 1]);

if isvector(labels)
    if nargin==1
        % No previous nn provided
        nn.labels{1}=labels(1);
        nn.nlabels=1;
    end
    for i=1:n_labels
        new=1;
        for x=1:nn.nlabels
            if nn.labels{x}==labels(i)
                new=0;
                class(i)=x;
                break;
            end
        end
        if new
            nn.nlabels=nn.nlabels+1;
            nn.labels{nn.nlabels}=labels(i);
            class(i)=nn.nlabels;
        end
    end
end

if iscell(labels)
    
end

end