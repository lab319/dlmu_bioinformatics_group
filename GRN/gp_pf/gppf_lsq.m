function [mse,corcfsq,theta] = gppf_lsq(fss,X,Y);
%Calculates linear parameters and error values
%  [mse,corcfsq,theta,Yv] = gppf_lsq(fss,X,Y);
%    mse <- mean squared error 
%    corcfsq <- square correlation coeffc. 
%    theta <- identified linear parameters (last element is the bias)
%    fss -> cell array of function strings
%    X,Y -> regression matrix and output vector
%

mse = Inf; 
corcfsq = 0;  
theta = zeros(length(fss)+1,1);

warning off;

%Calculate model terms (eval functions)
XX = [];
for i = 1:length(fss),
  xxi = eval(fss{i});
  XX = [XX, xxi]; 
end
d=Y;
n=53;

% Initializations
L=length(fss);     
N = 10000;            
P0 =60;  
%noise
Q=0.001;
R=1;
% Step of time
T=n;    

% Perform particle filtering
wp = sqrt(P0)*randn(L,N);    
for t=1:T
    wp=wp+sqrt(Q)*randn(L,N);
    y = (XX(t,:)*wp)';
    e = d(t)-y;  
    q0 = 1/(2*sqrt(R/2))*exp(-abs(e)/sqrt(R/2)); 
    q = q0/sum(q0);

    P = cumsum(q); 
    ut(1)=rand(1)/N;
    k = 1; 
    i = zeros(1,N);
    for j = 1:N
        ut(j)=ut(1)+(j-1)/N;
        while(P(k)<ut(j));
            k = k + 1;
        end;
        i(j) = k;
        q(j)=1/N;
    end;                  
    wp = wp(:,i);              
    for i = 1:L
        w(i,t) = mean(wp(i,:));
    end          
end;

theta(n,1)=0;
for i=1:L
    theta(i,1)=w(i,n);   
end;
YV = XX * theta(1:L,1);

%MSE:
mse = mean((YV-Y).^2);
if isnan(mse), mse = Inf; end

%Correlation coefficient:
c = corrcoef(YV,Y);
c = c(1,2);
if isnan(c), c=0; end
corcfsq = max(0,min(c*c,1));
