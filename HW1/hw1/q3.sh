# python q3.py {SEED} {DIM}
for(( seed=7; seed<20; seed=seed+1))
do
	for(( dim=1; dim<4; dim=dim+1))
	do
		python q3.py $seed $dim
	done
done
