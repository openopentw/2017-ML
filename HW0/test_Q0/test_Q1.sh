for x in {0..9}
do
	echo "${x}"
	./Q1.sh YJC_test/${x}_A.in YJC_test/${x}_B.in
	mv -f ans_one.txt YJC_test/${x}_your_ans.out
	diff YJC_test/${x}.out YJC_test/${x}_your_ans.out
done
