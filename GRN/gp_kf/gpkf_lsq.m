function [mse,corcfsq,theta] = gpkf_lsq(fss,X,Y)
%Calculates linear parameters and error values
%  [mse,corcfsq,theta,Yv] = gpkf_lsq(fss,X,Y);
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

% Initializations
L=length(fss);          
K = eye ( L, L )*5;         
wK = zeros ( L, 1 ) ;      
%noise
Q=0.001*eye(L,L);    
R=1;
% Number of iterations  
n=53;

% Perform Kalman filtering
for i=1:n
	wK(:,i)=wK(:,i)+sqrt(Q)*randn(L,1);
    y(i)=XX(i,:)*wK(:,i);   
	e(i) = d(i) - y(i) ;    
    G=K*XX(i,:)'/(XX(i,:)*K*XX(i,:)'+R);   
	wK(:,i+1) = wK(:,i) + G*e(i) ;        
 	K=K-G*XX(i,:)*K+Q;                    
end;
theta(n,1)=0;
for i=1:L
    theta(i,1)=wK(i,n);   
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
