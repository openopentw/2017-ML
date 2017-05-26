for ((x=0; x<50; x++))
do
	echo "${x}"
	./Q1.sh YJC_test/${x}_A.in YJC_test/${x}_B.in
	mv -f ans_one.txt YJC_test/${x}.out
	# mv -f ans_one.txt YJC_test/${x}_ans.out
	# diff YJC_test/${x}.out YJC_test/${x}_ans.out
done
