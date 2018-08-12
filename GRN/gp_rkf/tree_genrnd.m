function tree = tree_genrnd(maxtreedepth,symbols);
%Generates a random tree
% tree = tree_genrnd(maxtreedepth,symbols);
%   tree <- output tree
%   maxtreedepth -> maximum tree depth
%   symbols -> cell arrays of operator and terminator node strings
%

%Random filling
nn = [length(symbols{1}), length(symbols{2})];  
n = 2^floor(maxtreedepth)-1;
vt = zeros(n,1); 
vn = zeros(n,1);
for i=1:(n-1)/2, 
  [vt(i) vn(i)] = tree_genrndsymb(1/2,nn);
end
for i=(n+1)/2:n,
  [vt(i) vn(i)] = tree_genrndsymb(1,nn);
end
%Result
tree.maxsize = n;  
tree.nodetyp = vt;
tree.node = vn;    
tree.param = zeros(floor((tree.maxsize+1)/2),1);
tree.paramn = 0;

%------------------------------------------------------------------
function [nodetyp,node] = tree_genrndsymb(p0,nn)
%Generate a random symbol (tarminate or operate)
%  [nodetyp,node] = tree_genrndsymb(p0,nn)
%   nodetyp,node <- results
%   p0 -> probability of terminate node
%   nn -> vector [number of operators, variables]
%

if rand<p0,   
  nodetyp = 2;
else
  nodetyp = 1;
end
node = floor(nn(nodetyp)*rand)+1;  
