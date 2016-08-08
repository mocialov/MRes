% INDENT Indent text
%   This function is used to indent a text that will be printed to the 
%   standard output, depending on the verbose level.
%

%   Copyright (c) 2011 Roberto Calandra
%   $Revision: 0.10 $


function indent(verbose)

for i=1:verbose-1
    fprintf('    ')
end

end