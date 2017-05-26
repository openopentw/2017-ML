#include <cstdio>

int main()
{
	int m, n;
	scanf("%d%d", &m, &n);
	while(m || n) {
		int left[2] = {0, 1};
		int middle[2] = {1, 1};
		int right[2] = {1, 0};

		while(m != middle[0] || n != middle[1]) {
			if(m * middle[1] < n * middle[0]) {		// m/n < middle[0]/middle[1]
				right[0] = middle[0];
				right[1] = middle[1];
				middle[0] = left[0] + middle[0];
				middle[1] = left[1] + middle[1];
				putchar('L');
			} else {
				left[0] = middle[0];
				left[1] = middle[1];
				middle[0] = right[0] + middle[0];
				middle[1] = right[1] + middle[1];
				putchar('R');
			}
		}
		putchar('\n');

		scanf("%d%d", &m, &n);
	}
	return 0;
}
