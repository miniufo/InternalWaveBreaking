dset ^wndStrs.dat
options big_endian
undef -9999
title wind stress
xdef 2000 linear 0 0.1
ydef    1 linear 0 0.1
zdef    1 linear 0 1
tdef    1 linear 01Jan2000 1dy
vars 1
taux 0 99 taux
endvars
