function x = tiaozheng1(x0,x1)
[r,c] = find(x0);
for i = r
    for j = c
        x1(i,j) = 0;
    end
end
x = x1;
end