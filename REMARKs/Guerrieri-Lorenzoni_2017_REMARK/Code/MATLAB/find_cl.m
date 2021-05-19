function F = find_cl(c, j, b1, b2, r)
% Find consumption for current assets b1 and next assets b2

global theta z fac gameta

n = max(0, 1 - fac(j)*c.^gameta);
F = b1 - b2/(1+r) - c + theta(j)*n + z(j);

end