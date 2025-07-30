function R = calculateR4(x0, x1, n, S, P, D, q, C, Y_r_D1, Y_r_D2, Y_r_q, Y_r_C1, Y_r_C2, Y_r_P1, Y_r_P2)
% 预期售价P
% 预期销售量D
% 预期亩产量q
% 预期每亩成本C

e1 = -0.41;
e2 = -0.51;

% 改变预期销售量矩阵
% 玉米（7列）和小麦（6列）单独变化
D(:,[6,7]) = D(:,[6,7])* Y_r_D1(n);
% 其余农作物
D(:,[1:5 8:end]) = D(:,[1:5 8:end])* Y_r_D2(n);

% % 还原预期销售量矩阵
% D1 = [repmat(D(1,:), 6,1)
%       repmat(D(2,:), 14,1)
%       repmat(D(3,:),6,1)
%       repmat(D(4,:),8,1)
%       repmat(D(5,:),16,1)
%       repmat(D(6,:),4,1)
%       repmat(D(7,:),8,1)
%       repmat(D(8,:),16,1)
%       repmat(D(9,:),4,1)];
% D1;

% 根据蔬菜17~34的价格变化对其预期销售做出变化
% % 还原售价矩阵
% P1 = [repmat(P(1,:), 6,1)
%       repmat(P(2,:), 14,1)
%       repmat(P(3,:),6,1)
%       repmat(P(4,:),8,1)
%       repmat(P(5,:),16,1)
%       repmat(P(6,:),4,1)
%       repmat(P(7,:),8,1)
%       repmat(P(8,:),16,1)
%       repmat(P(9,:),4,1)];
% P = P1;
D1 = D;
D(:,17:34) = (Y_r_C2(n)-1)*e1*D1(:,17:34)+D1(:,17:34);

% 根据食用菌38~40的价格变化对其预期销售做出变化
D(:,38:40) = (Y_r_P1(n)-1)*e2*D1(:,38:40)+D1(:,38:40);
D(:,41) = (Y_r_P2(n)-1)*e2*D1(:,41)+D1(:,41);

nonZeroIndex1 = D ~= 0;
n1 = sum(nonZeroIndex1);
D = sum(D)./n1;

% 改变亩产量矩阵q
q = q.*Y_r_q(n);

%还原q(82*42)
q1(1:6,:) = repmat(q(1,:), 6,1);
q1(7:20,:) = repmat(q(2,:), 14,1);
q1(21:26,:) = repmat(q(3,:),6,1);
q1(27:34,:) = repmat(q(4,:),8,1);
q1(35:50,:) = repmat(q(5,:),16,1);
q1(51:54,:) = repmat(q(6,:),4,1);
q1(55:62,:) = repmat(q(7,:),8,1);
q1(63:78,:) = repmat(q(8,:),16,1);
q1(79:82,:) = repmat(q(9,:),4,1);

q = q1;
q = hubu_tiaozheng(x0,q);           %因为豆类植物


% 改变成本矩阵C
C = C.*Y_r_C1(n);

% 还原C
% C1 = [repmat(C(1,:), 6,1)
%       repmat(C(2,:), 14,1)
%       repmat(C(3,:),6,1)
%       repmat(C(4,:),8,1)
%       repmat(C(5,:),16,1)
%       repmat(C(6,:),4,1)
%       repmat(C(7,:),8,1)
%       repmat(C(8,:),16,1)
%       repmat(C(9,:),4,1)];

nonZeroIndex = C ~= 0;
n0 = sum(nonZeroIndex);
C = sum(C)./n0;


% 改变售价矩阵，蔬菜17~37，使用菌38~40，羊肚菌41
P(:,17:37) = P(:,17:37)*Y_r_C2(n);
P(:,38:40) = P(:,38:40)*Y_r_P1(n);
P(:,41) = P(:,41)*Y_r_P2(n);

P = sum(P)./n0;


x1 = x1./sum(x1,2);
Q1 = x1.*S;
Q1(isnan(Q1)) = 0;

Q11 = Q1.*q;
Q11 = sum(Q11);
Q1 = sum(Q1);

R = sum(P.*min(Q11,D) + 0.5*P.*max(Q11-D,0) - C.*Q1, "all");
end











