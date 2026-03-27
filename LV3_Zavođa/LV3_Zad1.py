import pandas as pd
import numpy as np

 
cars=pd.read_csv('mtcars.csv')
 
print(cars.sort_values("mpg").tail(5).car)
print("\n")
print(cars[cars.cyl==8].sort_values("mpg").head(3).car)
print("\n")
print(cars[cars.cyl == 6].mpg.mean())
print("\n")
print(cars[(cars.cyl == 4) & ((cars.wt>2.0) & (cars.wt <2.2))].mpg.mean())
print("\n")
print("automatski mijenjač = ", cars[cars.am ==1].am.count())
print("ručni mijenjač = ", cars[cars.am ==0].am.count())
print("\n")
print(cars[(cars.am==1) & (cars.hp > 100)].hp.count())
print("\n")
print(cars.wt.aggregate([sum])*0.453592*1000)
print(cars.head(5).max().car)


