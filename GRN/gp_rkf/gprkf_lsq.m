function [mse,corcfsq,theta] = gprkf_lsq(fss,X,Y)
%Calculates linear parameters and error values
%  [mse,corcfsq,theta,Yv] = gprkf_lsq(fss,X,Y);
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
K =eye ( L, L ) *5;
wK = zeros ( L, 1 ) ;
w = zeros ( L, 1 ) ;
%noise
Q=0.001*eye(L,L);
R=1;
% Number of iterations  
n=53;

% Perform Robust Kalman filtering
for i=1:n
	wK(:,i)=wK(:,i);
    P=K+Q;
    G=(P*XX(i,:)')/(XX(i,:)*P*XX(i,:)'+R);
    y(i)=XX(i,:)*wK(:,i);
    y_new(i)=XX(i,:)*w(:,i);
    e(i) = d(i) - y(i) ; 
    e_new(i) = d(i) - y_new(i) ;
    wK(:,i+1) = wK(:,i) + G * e(i) ;
	K=P-G*XX(i,:)*P; 
    yy=XX(i,:);
    juge=0;
    for j=1:length(yy)
        if yy(j)==inf||isnan(yy(j))||G(j)==inf||isnan(G(j))
            juge=1;
        end
    end
    if juge==0
        gg=gprkf_ga(G,yy,K,Q,R);
    else
        gg=G;
    end
    w(:,i+1) = w(:,i) + gg* e(i) ;
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

