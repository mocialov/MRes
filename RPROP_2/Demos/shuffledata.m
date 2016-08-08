% SHUFFLEDATA Shuffle the rows of a matrix
%   [MATRIX,INDEX] = SHUFFLEDATA(MATRIX) Shuffle the input MATRIX of
%   dimensions [ndata x ndim], returning the same MATRIX (with the same
%   dimensions) and rows randomly permuted. Optionally it also return a
%   vector INDEX = [ndata x 1] with the indexes of the random permutations
%   applied.
%
% 	[MATRIX,INDEX] = SHUFFLEDATA(MATRIX,INDEX) As above, but the indexes of
%   the permutations are forced from the vector INDEX = [ndata x 1].
%

%   Copyright (c) 2011 Roberto Calandra
%   $Revision: 0.11 $


function [matrix,index] = shuffledata(matrix,index)

[ndata ndim] = size(matrix); % [Number of samples, Number of input dimensions]

if nargin<2
    % No index passed
    
    index = randperm(ndata)';
    matrix = matrix(index,:);
else
    % Index passed
    
    assert(ndata==size(index,1),...
        'Number of samples and size of index do not match')
    
    matrix = matrix(index,:);
end

end