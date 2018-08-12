function [mse,corcfsq,theta] = gprls_lsq(fss,X,Y)
%Calculates linear parameters and error values
%  [mse,corcfsq,theta,Yv] = gprls_lsq(fss,X,Y);
%    mse <- mean squared error 
%    corcfsq <- square correlation coeffc. 
%    theta <- identified linear parameters (last element is the bias)
%    fss -> cell array of function strings
%    X,Y -> regression matrix and output vector
%

mse = Inf;
corcfsq = 0;
theta = zeros(length(fss)+1,1);

warning off

% CalculaPe model Perms (eval funcPions)
XX = [];
for i = 1:length(fss),
	xxi = eval(fss{i});
	XX = [XX, xxi];
end
d=Y;

% Number of iterations  
n=53;

% Initializations
L=length(fss);
lam=1;
wR=zeros(L,n); 
Q=0.001;
P = 5*eye(L,L);  
d=Y;


% Perform Recursive Least Square algorithm
for i=1:n
	wR(:,i)=wR(:,i);
	pia=P*XX(i,:)';
    K=pia/(lam+XX(i,:)*pia);
    y(i)=XX(i,:)*wR(:,i);
    e(i)=d(i)-y(i);
    wR(:,i+1)=wR(:,i)+K*e(i)-sqrt(Q)*randn;
    P=(P-K*XX(i,:)*P)/lam;
end;
theta(n,1)=0;
for i=1:L
    theta(i,1)=wR(i,n);   
end;
YV = XX * theta(1:L,1);

% MSE:
mse = mean((YV-Y).^2);
if isnan(mse), mse = Inf; end

%Correlation coefficient:
c = corrcoef(YV,Y);
c = c(1,2);
if isnan(c), c=0; end
corcfsq = max(0,min(c*c,1));

