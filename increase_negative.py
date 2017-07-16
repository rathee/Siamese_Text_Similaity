import sys
import pandas as pd
import random

df = pd.read_csv(sys.argv[1], delimiter='\t', header=None)

stats = df[2].value_counts()
pos = stats[1]
neg = stats[0]
size = len(df)

percent = pos * 100.0 / neg
df.to_csv(sys.argv[2], sep = '\t', index=False)

wr = open(sys.argv[2], 'a')
while percent > 17.0 :

		first_index = random.randint(0, size - 1)
		second_index = random.randint(0,size - 1)

		q1 = df.ix[first_index][0]
		if q1 is None or q1 == '' or q1 == 'nan':
				continue
		q2 = df.ix[second_index][1]
		if q2 is None or q2 == '' or q2 == 'nan':
				continue
		is_duplicate = 0
		try :
			wr.write(q1 + '\t' + q2 + '\t' + '0' + '\n')
			neg += 1
			percent = pos * 100.0 / neg
		except Exception as e :
				print q1,q2

wr.close()
