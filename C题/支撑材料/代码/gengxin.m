function x = gengxin(change_id,x1)
    for i = change_id
        if ismember(i,[1:26])
            % 在作物1~15中随机选取3种作物进行种植，因为要尽可能的集中种植
            id = randperm(15, 3);
            %确定三种作物种植面积的比例，并且允许比例为0
            for j = 1:3
                if rand(1) < 0.2
                    id1 = 0;
                elseif all([rand(1)>=0.2,rand(1)<0.4])
                    id1 = 1;
                elseif all([rand(1)>=0.4,rand(1)<0.6])
                    id1 = 2;
                elseif all([rand(1)>=0.6,rand(1)<0.8])
                    id1 = 3;
                else 
                    id1 = 4;
                end
                x1(i,id(j)) = id1;
            end
        elseif ismember(i,[27:34])
            % 在作物16~34中随机选取3种作物进行种植，因为要尽可能的集中种植
            id = randperm(34-15, 3)+15;
            %确定三种作物种植面积的比例，并且允许比例为0
            for j = 1:3
                if rand(1) < 0.2
                    id1 = 0;
                elseif all([rand(1)>=0.2,rand(1)<0.4])
                    id1 = 1;
                elseif all([rand(1)>=0.4,rand(1)<0.6])
                    id1 = 2;
                elseif all([rand(1)>=0.6,rand(1)<0.8])
                    id1 = 3;
                else 
                    id1 = 4;
                end
                x1(i,id(j)) = id1;
            end
            
        elseif ismember(i,[35:54])
            % 在作物17~34中随机选取3种作物进行种植，因为要尽可能的集中种植
            id = randperm(37-16, 3)+16;
            %确定三种作物种植面积的比例，并且允许比例为0
            for j = 1:3
                if rand(1) < 0.2
                    id1 = 0;
                elseif all([rand(1)>=0.2,rand(1)<0.4])
                    id1 = 1;
                elseif all([rand(1)>=0.4,rand(1)<0.6])
                    id1 = 2;
                elseif all([rand(1)>=0.6,rand(1)<0.8])
                    id1 = 3;
                else
                    id1 = 4;
                end
                x1(i,id(j)) = id1;
            end

        elseif ismember(i,[55:62])
            % 在作物35~37中随机种植
            %确定三种作物种植面积的比例，并且允许比例为0
            id1 = [];
            for j = 1:3
                if rand(1) < 0.2
                    id1(j) = 0;
                elseif all([rand(1)>=0.2,rand(1)<0.4])
                    id1(j) = 1;
                elseif all([rand(1)>=0.4,rand(1)<0.6])
                    id1(j) = 2;
                elseif all([rand(1)>=0.6,rand(1)<0.8])
                    id1(j) = 3;
                else
                    id1(j) = 4;
                end
            end
            x1(i,35:37) = id1;
         elseif ismember(i,[63:78])
            % 在作物38~41中随机种植
            %确定四种作物种植面积的比例，并且允许比例为0
            id1 = [];
            for j = 1:4
                if rand(1) < 0.3
                    id1(j) = 0;
                elseif all([rand(1)>=0.3,rand(1)<0.4])
                    id1(j) = 1;
                elseif all([rand(1)>=0.4,rand(1)<0.6])
                    id1(j) = 2;
                elseif all([rand(1)>=0.6,rand(1)<0.8])
                    id1(j) = 3;
                else
                    id1(j) = 4;
                end
            end
            x1(i,38:41) = id1;
        elseif ismember(i,[79:82])
            % 在作物17~34中随机选取3种作物进行种植，因为要尽可能的集中种植
            id = randperm(34-16, 3)+16;
            %确定三种作物种植面积的比例，并且允许比例为0
            id1 = [];
            for j = 1:3
                if rand(1) < 0.2
                    id1(j) = 0;
                elseif all([rand(1)>=0.2,rand(1)<0.4])
                    id1(j) = 1;
                elseif all([rand(1)>=0.4,rand(1)<0.6])
                    id1(j) = 2;
                elseif all([rand(1)>=0.6,rand(1)<0.8])
                    id1(j) = 3;
                else
                    id1(j) = 4;
                end
            end
            x1(i,id) = id1;
        end       
    end
    x = x1;
end