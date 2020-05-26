[x,y] = meshgrid(0.1:0.2:5,0.1:0.2:5);
a = 1;
b = 2;
u = ones(length(x), length(x));
% v = -x .* b^2 ./ (y .* a^2);
v = -b^2 .* x ./ (a^2 .* y);
quiver(x, y, u, v)