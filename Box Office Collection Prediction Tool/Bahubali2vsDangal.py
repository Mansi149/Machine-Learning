import pandas as pd  

dataset = pd.read_csv("Bahubali2_vs_Dangal.csv")

# days
feature = dataset.iloc[:, 0:1].values
# bahubali 2
label1 = dataset.iloc[:, 1:2].values  
# dangal
label2 = dataset.iloc[:, 2:3].values

from sklearn.linear_model import LinearRegression

regressor1, regressor2 = LinearRegression(), LinearRegression()
regressor1.fit( feature , label1 ), regressor2.fit( feature , label2 )

day = int(input("Enter the day for which you want to find the collection : "))

import numpy as np

x = [day]
x = np.array(x)
x = x.reshape(1, 1)


print("Bahubali 2 Collection for", day,"th day :", regressor1.predict(x))

print("Dangal Collection for", day,"th day :", regressor2.predict(x))



# plotting graphically
import matplotlib.pyplot as mtp  


mtp.plot(feature, label1, color = 'pink', label = "Bahubali 2 Collection")
mtp.plot(feature, label2, color = 'brown', label = " Dangal Collection")
mtp.scatter(day, regressor1.predict(x), color = 'pink', s = 100)
mtp.scatter(day, regressor2.predict(x), color = 'brown', s = 100)
mtp.title('Bahubali 2 vs Dangal')
mtp.xlabel('Day')
mtp.ylabel('Collection')
mtp.legend()
mtp.show()


# comaparing the collections of the two movies 
if regressor1.predict(x) > regressor2.predict(x):
 print ("Bahubali 2 will earn more on the", day,"th day")
else:
 print ("Dangal will earn more on the", day,"th day")
 
 
 
 
 
 
 
 
 
 
 
 