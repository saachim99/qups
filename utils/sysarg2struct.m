function stru = sysarg2struct(stri)
    args = split((" " + string(stri)), ' --'); % create cells for each double -- arg
    args(~contains(args, '=')) = []; % delete cells without an equals
    args = cellfun(@(c)split(c,'='), args(:)', 'UniformOutput', false); % get each side of argument
    args = [args{:}]; % make 2 x N cells
    args(1,:) = cellfun(@(s)regexprep(s,'-','___'), args(1,:), 'UniformOutput', false);% hack: turn '-' to '___'
    stru = struct(args{:}); % make struct
end

