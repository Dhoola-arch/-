function R = calculateR2(xxx ,S, P, D, q, C)
xxx = xxx./sum(xxx,2);
Q1 = xxx.*S;
Q1(isnan(Q1)) = 0;
Q_1(1,:) = sum(Q1(1:6,:));
Q_1(2,:) = sum(Q1(7:20,:));
Q_1(3,:) = sum(Q1(21:26,:));
Q_1(4,:) = sum(Q1(27:34,:));
Q_1(5,:) = sum(Q1(35:50,:));
Q_1(6,:) = sum(Q1(51:54,:));
Q_1(7,:) = sum(Q1(55:62,:));
Q_1(8,:) = sum(Q1(63:78,:));
Q_1(9,:) = sum(Q1(79:82,:));

Q_11 = Q_1.*q;

R = sum(P.*min(Q_11,D) + 0.5*P.*max(Q_11-D,0) - C.*Q_1, "all");
end