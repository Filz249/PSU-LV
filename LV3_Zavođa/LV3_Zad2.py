import pandas as pd
import matplotlib.pyplot as plt
 
mtcars = pd.read_csv('mtcars.csv')
 
cyl_groups = mtcars.groupby('cyl')['mpg'].mean()
 
plt.figure()
cyl_groups.plot(kind='bar')
plt.title('Srednja potrošnja po broju cil.')
plt.xlabel('Broj cil.')
plt.ylabel('Potrosnja (mpg)')
 
weights = [
    mtcars[mtcars['cyl'] == cyl]['wt']
    for cyl in [4, 6, 8]
]
 
plt.figure()
plt.boxplot(weights, labels=['4 cilindra', '6 cilindara', '8 cilindara'])
plt.title('Distribucija težine po broju cilindara')
plt.xlabel('Broj cilindara')
plt.ylabel('Tezina (1000 lbs)')
 
auto = mtcars.loc[mtcars['am'] == 0, 'mpg']
manual = mtcars.loc[mtcars['am'] == 1, 'mpg']
 
plt.figure()
plt.boxplot([auto, manual], labels=['Automatski', 'Ručni'])
plt.title('Potrosnja: automatski vs ručni mjenjač')
plt.xlabel('Vrsta mjenjača')
plt.ylabel('Potrosnja (mpg)')
 
auto_data = mtcars[mtcars['am'] == 0]
manual_data = mtcars[mtcars['am'] == 1]
 
plt.figure()
plt.scatter(auto_data['hp'], auto_data['qsec'], label='Automatski')
plt.scatter(manual_data['hp'], manual_data['qsec'], label='Ručni')
plt.title('Ubrzanje vs Snaga po vrsti mjenjača')
plt.xlabel('Snaga (hp)')
plt.ylabel('Ubrzanje (qsec)')
plt.legend()
 
plt.show()