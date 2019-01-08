print("Candidate elimination algorithm")


s=['0']*6
g=['?']*6
d = (['Sunny','Warm','Normal','Strong','Warm','Same','yes'],['Sunny','Warm','High','Strong','Warm','Same','yes'],
['Rainy','Cool','High','Strong','Warm','Change','no'],
['Sunny','Warm','High','Strong','Cool','Change','yes'])

for i in range(0,len(d)):
	if d[i][-1]=="no":
		for k in range(0,6):
			if d[i][k]=='Rainy':
				g[k]=['Sunny','?','?','?','?','?']
			elif d[i][k]=="Cool":
				g[k]=['?','Warm','?','?','?','?']
			elif d[i][k]=="High":
				g[k]=['?','?','Low','?','?','?']
			elif d[i][k]=="Strong":
				g[k]=['?','?','?','Strong','?','?']
			elif d[i][k]=="Warm":
				g[k]=['?','?','?','?','Cool','?']
			elif d[i][k]=="Change":
				g[k]=['?','?','?','?','?','Change']
	elif d[i][-1]=="yes":
		for j in range(0,6):
			if s[j]=='0':
				s[j]=d[i][j]
			elif s[j]!=d[i][j]:
				s[j]='?'
print("most specific",s)
#print(g)
f=[]			
for i in range (0,5):
	for j in range (0,6):
		if i == j :
			if(s[i] == g[i][j] and g[i] not in f):
				f.append(g[i])
			else:
				continue
		else:
			continue
print("Most general :",f[:len(f)-1])


