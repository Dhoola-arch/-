% 考虑互补的影响，即在前一年种植过豆类作物的地块在第二年会有增产现象
function q1 = hubu_tiaozheng(x0,q)
q1 = q;
r = find(sum(x0(:,1:5),2)~=0);
for i = r
    q1(i,:) = q(i,:)*1.2;
end

r = find(sum(x0(:,17:19),2)~=0);
for i = r
    q1(i,:) = q(i,:)*1.2;
end

end