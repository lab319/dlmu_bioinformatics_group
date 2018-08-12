function J=gprkf_fun(x,z,G,K,Q,R)
%Calculates the performance index
%    J=gprkf_fun(x,z,G,K,R)
%    J <- the performance index
%    x -> the value of x when J is at the extreme value
%    z -> model terms
%    G -> the kalman filter gain
%    K -> the estimatin error's covariance 
%    R -> the measurement noise's covariance
%
    L=length(K);
    I=eye(L);
    k=G*x;
    y=I-k*z;
    zero_temp = eps*eye(L,L);
    %Lyapunov equation
    [Pr,L_r,G_r,report_r] = dare ( y', zeros(L,L), (I*Q*I'), zero_temp );
    [Pq,L_q,G_q,report_q] = dare ( y', zeros(L,L), (k*R*k'), zero_temp); 
    J=-(trace(Pq)+trace(Pr));
end