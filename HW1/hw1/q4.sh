# python q3.py {SEED} {LMBD}
for lmbd in 1000000
do
	for (( dim=1; dim<4; dim=dim+1))
	do
		# echo $lmbd
		python q4.py $lmbd $dim
	done
done
