% the inference of the first equation of E_cell model under Laplace noise by GP and PF
%
% the equation : Y=-10.32*X1*X3;
% n: the number of sample
% X: the time-series data 
% Y: the differential of X, i.e. dX/dt
% symbols: the symbols of operational character and terminal symbols
% popsize: the number of population
% maxtreedepth: the depth of the tree-like individuals 

clear
load e_cell
n=53;
X(:,1:3)=x(1:n,1:3);
Y=x(1:n,4); 
% the Laplace noise with variance of 1
mu=0;
sigma=1; 
b=sigma/sqrt(2);
a=rand(size(Y))-0.5;
xx=mu-b*sign(a).*log(1-2*abs(a));
y=randn(size(Y));
z=0.5*xx+0.5*y;

Y = Y+z;

% GP equation symbols 
symbols{1} = {'+','*','/'};
symbols{2} = {'x1','x2','x3'}; 

% Initial population
popusize = 40;
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
 for c = 2:20,
  %iterate 
  popu = gppf_mainloop(popu,X,Y,[],opt);
  disp(gppf_result(popu,1));
end

% Result: return the optimum solution and polynome in the final generation
[s,tree,func] = gppf_result(popu,2);
disp(s);

