function x = tiaozheng2(x0,x1,x2)
    for i = 1:82
        if ismember(i,[1:26])
            m1 = sum(x0(i,1:5));
            m2 = sum(x1(i,1:5));
            m3 = sum(x2(i,1:5));
            if m1+m2+m3 <= 0.00001
                j = randperm(5, 1);
                x2(i,j) = randperm(4, 1);
            end
        
        elseif ismember(i,[27:50])
            m1 = sum(x0(i,17:19));
            m2 = sum(x1(i,17:19));
            m3 = sum(x2(i,17:19));
            if m1+m2+m3 <= 0.00001
                j = randperm(3, 1)+16;
                x2(i,j) = randperm(4, 1);
            end
        
            
        elseif ismember(i,[51:54])
            m1 = sum(x0(i,17:19)) + sum(x0(i+28,17:19));
            m2 = sum(x1(i,17:19)) + sum(x1(i+28,17:19));
            m3 = sum(x2(i,17:19)) + sum(x2(i+28,17:19));
            if m1+m2+m3 <= 0.00001
                if rand(1)<=0.5
                    j = randperm(3, 1)+16;
                    x2(i,j) = randperm(4, 1);
                else
                    j = randperm(3, 1)+16;
                    x2(i+28,j) = randperm(4, 1);
                end
            end

        else
            x2 = x2;
        end
    end
    x = x2;
end