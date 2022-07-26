function stri = struct2sysarg(stru)
    [nms, vals] = deal(shiftdim(fieldnames(stru),-1), shiftdim(struct2cell(stru),-1));
    nms = string(cellfun(@(c)regexprep(c, '___', '-'), nms, 'UniformOutput', false)); % convert '___' back to '-'
    vals = cellfun(@string, vals, 'UniformOutput', false); % ensure text
    stri = join(compose("--%s=%s", [nms; vals]'));    
end