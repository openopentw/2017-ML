#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cstring>

int main()
{
	// n_gen should <= 10
	int n_gen = 99;

	srand( (unsigned)time(NULL) );
	for(int i = 3; i < n_gen; ++i) {
		int h = rand() % 10 + 1;
		int w = rand() % 10 + 1;
		for(int j = 0; j < 2; ++j) {
			h = w;
			w = rand() % 10 + 1;

			char fn[64];
			if(i < 10) {
				strcpy(fn, "YJC_test/i_j.in");
				fn[9] = i + '0';
				fn[11] = j + 'A';
			} else if (i < 100) {
				strcpy(fn, "YJC_test/ii_j.in");
				fn[9] = i / 10 + '0';
				fn[10] = i % 10 + '0';
				fn[12] = j + 'A';
			}
			puts(fn);

			FILE *fp = fopen(fn, "w");

			for(int k = 0; k < h; ++k) {
				if(k != 0)
					fprintf(fp, "\n");
				for(int l = 0; l < w; ++l) {
					if(l != 0)
						fprintf(fp, ",");
					fprintf(fp, "%d", int(rand() % 1000));
				}
			}

			fclose(fp);
		}
	}

	return 0;
}
