% CLASS_STAT Compute various classification statistics
%   Compute various classification statistics
%

%   Copyright (c) 2011 Roberto Calandra
%   $Revision: 0.12 $


function [stats] = class_stat(inputs,targets,n_class)
%% Input validation

assert(size(inputs,1)==size(targets,1),...
    'Number of inputs and targets samples do not match')
assert(size(inputs,2)==1,...
    'Wrong number of inputs dimensions')
assert(size(targets,2)==1,...
    'Wrong number of targets dimensions')


%% Init

stats.n_samples = size(inputs,1);

if nargin < 3
    % Compute the number of classes (either in inputs or targets)
    classes=[];
    for i=1:stats.n_samples
        if ~sum(classes==targets(i))
            classes=[classes targets(i)];
        end
        if ~sum(classes==inputs(i))
            classes=[classes inputs(i)];
        end
    end
    n_class = size(classes,2);
    %n_class = max(max(inputs),max(targets));
end


%% Compute Confusion Matrix

stats.confusion = zeros([n_class n_class]);
for x=1:n_class
    for y=1:n_class
        stats.confusion(y,x) = sum((inputs == x).*(targets == y));
    end
end


%% Compute more statistics

stats.n_targets     = sum(stats.confusion)';
stats.class_acc     = diag(stats.confusion)*100./stats.n_targets;
stats.n_correct     = trace(stats.confusion);
stats.n_wrong       = stats.n_samples-stats.n_correct;
stats.accuracy      = stats.n_correct*100/stats.n_samples;
stats.errorrate     = stats.n_wrong*100/stats.n_samples;
stats.binary        = (inputs~=targets);


end

