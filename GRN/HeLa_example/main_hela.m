
% n: the number of sample
% X: the time-series data 
% Y: the differential of X, i.e. dX/dt
% symbols: the symbols of operational character and terminal symbols
% popsize: the number of population
% maxtreedepth: the depth of the tree-like individuals 

clear
load hela
iteration=20;
n=46;
X(:,1:5)=x(1:n,1:5);
Y=x(1:n,6);  % 6,7,8,9


% GP equation symbols 
symbols{1} = {'+','*','/','+'};
symbols{2} = {'x1','x2','x3', 'x4','x5'}; 

% Initial population
popusize = 80;
maxtreedepth = 5;
% generate the initial population  
popu = gppf_init(popusize,maxtreedepth,symbols);

% first evaluation
opt = [0.8 0.5 0.3 2 1 0.2 30 0.05 0 0];
popu = gppf_evaluate(popu,[1:popusize],X,Y,[],opt(6:9));

% info
disp(gppf_result([],0));       
disp(gppf_result(popu,1));

% GP+pf loops
 for c = 2:iteration
  %iterate 
  popu = gppf_mainloop(popu,X,Y,[],opt);
  disp(gppf_result(popu,1));
end

% Result: return the optimum solution and polynome in the final generation
[s,tree,func] = gppf_result(popu,2);
disp(s);

