function g=gprkf_ga(G,z,K,Q,R)
%Uses genetic algorithm to find the optimal robust kalman filter gain
%    g=gprkf_ga(G,z,K,R);
%    g <- the optimal robust kalman filter gain
%    G -> the kalman filter gain
%    z -> model terms
%    K -> the estimatin error's covariance 
%    R -> the measurement noise's covariance
%
    options=gaoptimset('Generations',20,'StallGenLimit',10,'display','off');
    p=0.01;    
    [x,fval]=ga(@(x)gprkf_fun(x,z,G,K,Q,R),1,[],[],[],[],(1-p),(1+p),[],options); %xx为搜寻的增益G的范围即自变量
    g=G*x;   
end
   
