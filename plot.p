set datafile separator ","
set border 3
set xtics nomirror
set ytics nomirror
set xtics 0,50,500
set ytics 0,50,500

set grid

p "train.csv" lc 6 pt 7 ps 1.8 , "pred.csv" lc -1 lw 5   with lines 
pause 1
reread
