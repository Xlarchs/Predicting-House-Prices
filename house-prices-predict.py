import pandas as pd
from sklearn.preprocessing import PolynomialFeatures,LabelEncoder,OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt

#import xlsx file
data = pd.read_excel('housePrices.xlsx')

##encoder:  Categoric -> Numeric
location= data.iloc[:,:1].values

le=LabelEncoder()
location[:,0] = le.fit_transform(location[:,0])

ohe = OneHotEncoder(categories='auto')
location=ohe.fit_transform(location).toarray()

#transformation  dataFrame
locationFrame=pd.DataFrame(data=location,index=range(1008),columns=['Arnavutköy','Ataşehir','Avcılar','Bahçelievler','Bakırköy','Bayrampaşa','Bağcılar','Başakşehir','Beykoz','Beylikdüzü','Beyoğlu','Beşiktaş','Büyükçekmece','Esenler','Esenyurt','Eyüpsultan','Fatih','Gaziosmanpaşa','Güngören','Kadıköy','Kartal','Kağıthane','Küçükcekmece','Maltepe','Pendik','Sancaktepe','Sarıyer','Silivri','Sultanbeyli','Sultangazi','Tuzla','Zeytinburnu','Çatalca','Çekmeköy','Ümraniye','Üsküdar','Şişli'])

##encoder:  Categoric -> Numeric
rooms=data.iloc[:,2:3].values
rooms[:,0] = le.fit_transform(rooms[:,0])

ohe = OneHotEncoder(categories='auto')
rooms=ohe.fit_transform(rooms).toarray()
result2=pd.DataFrame(data=rooms,index=range(1008),columns=['1+0','1+1','2+0','2+1','2+2','3+1','3+2','4+1','4+2','5+1','5+2','6+1','6+2'])

#transformation dataFrame
m2=data.iloc[:,1:2].values
m2=pd.DataFrame(data=m2,index=range(1008),columns=['m2'])

prices=data.iloc[:,3:4].values
prices=pd.DataFrame(data=prices,index=range(1008),columns=['Fiyat'])

#Combine Results
s=pd.concat([locationFrame,result2],axis=1)
s2=pd.concat([s,m2],axis=1)
result=pd.concat([s2,prices],axis=1)

#Polynomial Regression
poly_reg = PolynomialFeatures(degree=3)
X_poly = poly_reg.fit_transform(s2)


#GradiendBoosting Regressor
gbr = GradientBoostingRegressor(n_estimators=600,max_depth=5, learning_rate=0.01, min_samples_split=3)
gbr = GradientBoostingRegressor()
gbr.fit(X_poly,prices)
predictResults = gbr.predict(X_poly)

#visualize the original and predicted values in a plot.
x_prices = range(len(prices))
plt.scatter(x_prices, prices, s=5, color="blue", label="original value")
plt.plot(x_prices, predictResults, lw=0.8, color="red", label="predict value")
plt.legend()
plt.show()


