for num = 1:8
    for loop=1:50
        load hela
        load hela_rand
        iteration=20;
        n=46;
        X(:,1:4)=x(1:n,1:4);
        X(:,5) = y(:,num);
        Y=x(1:n,6);  % 6,7,8,9
        symbols{1} = {'+','*','/','+'};
        symbols{2} = {'x1','x2','x3', 'x4','x5'}; 

        popusize = 80;
        maxtreedepth = 5;
        popu = gppf_init(popusize,maxtreedepth,symbols);

        opt = [0.8 0.5 0.3 2 1 0.2 30 0.05 0 0];
        popu = gppf_evaluate(popu,[1:popusize],X,Y,[],opt(6:9));

        for c = 2:iteration
            popu = gppf_mainloop(popu,X,Y,[],opt);
        end
        [s,tree,func] = gppf_result(popu,2);
        disp(s);
    end
end