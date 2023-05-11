import numpy as np
import matplotlib.pyplot  as plt
import matplotlib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import sklearn.linear_model as lm
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score
from sklearn.metrics import max_error
from sklearn.linear_model import LogisticRegression

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix , ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.cluster import KMeans, AgglomerativeClustering

from tensorflow import keras
from tensorflow.keras import layers         #bug, radi
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
#NUMPY I MATPLOTLIB,lv2

#numpy ucitavanje polja
data = np.loadtxt('data.csv', delimiter=',', skiprows=1)#preskace prvi red jer je string
#ukupni broj istraživanja
print(f"Broj ljudi: {data.shape[0]}")
#scatter dvije veličine
visina = data[:, 1]
masa = data[:, 2]

plt.scatter(visina, masa, c='b', s=1, marker=".")
plt.xlabel('Visina(cm)')
plt.ylabel('Masa(kg)')
plt.title('Odnos visine i mase')
plt.show()

#scatter za svaku 50. osobu
visina50 = data[::50, 1]
masa50 = data[::50, 2]
plt.scatter(visina50, masa50, c='b', s=1, marker=".")

#max,min,srednja vrijednost
print(f'Najveca vrijednost visine: {visina.max()}cm')
print(f'Najmanja vrijednost visine: {visina.min()}cm')
print(f'Srednja vrijednost visine: {visina.mean()}cm')

#prvi stupac samo 1 ili 0 pa odvajamo za ž i m
visinaM = data[:, 1][data[:, 0] == 1]
visinaZ = data[:, 1][data[:, 0] == 0]

#PANDAS,lv3

data = pd.read_csv('data_C02_emission.csv')#ČITANJE
print(f'Broj mjerenja: {len(data)}')#broj mjerenja

#provjera kojeg su tipa,duplicirane vrijednosti
print(data.dtypes)
print(f'Broj dupliciranih vrijednosti:{data.duplicated().sum()}')
print('Broj izostalih vrijednosti po stupcima:')
print(data.isnull().sum())
data['Make']=pd.Categorical(data['Make'])
data['Vehicle Class']=pd.Categorical(data['Vehicle Class'])
data['Transmission']=pd.Categorical(data['Transmission'])
data['Fuel Type']=pd.Categorical(data['Fuel Type'])
print(data.dtypes)

#3 najveća i najmanja po uvjetu
topAndBottom3=data.sort_values(by='Fuel Consumption City (L/100km)')
print('Najmanja 3:')
print(topAndBottom3[['Make','Model','Fuel Consumption City (L/100km)']].head(3))
print('Najveća 3:')
print(topAndBottom3[['Make','Model','Fuel Consumption City (L/100km)']].tail(3))
#ostalo po srednjoj vrijednosti i ostalim uvjetima lv2 zad1
#korelacija između numeričkih veličina
print(data.corr(numeric_only=True))

#histogram po veličini
data['CO2 Emissions (g/km)'].plot(kind='hist', bins=20)
#scatter za 2 veličine,obojane točkice za 3. veličinu
data['Make']=pd.Categorical(data['Make'])
data['Vehicle Class']=pd.Categorical(data['Vehicle Class'])
data['Transmission']=pd.Categorical(data['Transmission'])
data['Fuel Type']=pd.Categorical(data['Fuel Type'])
data.plot.scatter(x='Fuel Consumption City (L/100km)', y='CO2 Emissions (g/km)', c='Fuel Type', cmap = 'viridis', s=20)
plt.show()

#s obzirom na uvjet(tip goriva) kutijasti dijagram za potrošnju
data.groupby('Fuel Type').boxplot(column='Fuel Consumption Hwy (L/100km)')

#LINEARNA REGRESIJA,lv4
#model procijenjuje emisiju plinova na sve ulazne 
data = pd.read_csv('data_C02_emission.csv')
y=data['CO2 Emissions (g/km)'].copy()
X=data[['Engine Size (L)','Cylinders','Fuel Consumption City (L/100km)','Fuel Consumption Hwy (L/100km)','Fuel Consumption Comb (L/100km)','Fuel Consumption Comb (mpg)']]
X_train , X_test , y_train , y_test = train_test_split (X, y, test_size = 0.2, random_state =1)#podijela u omjeru 80%-20%

#Pomocu matplotlib biblioteke i dijagrama raspršenja prikažite ovisnost emisije C02 plinova ´
#o jednoj numerickoj veli ˇ cini. Pri tome podatke koji pripadaju skupu za u ˇ cenje ozna ˇ cite ˇ
#plavom bojom, a podatke koji pripadaju skupu za testiranje oznacite crvenom bojom.
for col in X_train.columns:
    plt.scatter(X_train[col],y_train, c='b', label='Train', s=5)
    plt.scatter(X_test[col],y_test, c='r', label='Test', s=5)
    plt.xlabel(col)
    plt.ylabel('CO2 Emissions (g/km)')
    plt.legend()
    plt.show()

sc = MinMaxScaler () #transform vraca numpy array, zato se mora nazad u dataframe (da se sve radilo s numpy, ne bi bilo potrebe za time)
X_train_n = pd.DataFrame(sc.fit_transform(X_train), columns=X_train.columns, index=X_train.index)#standardizacija veličina
#histogram prije i nakon skaliranja
for col in X_train.columns:
    fig,axs = plt.subplots(2,figsize=(8, 8))
    axs[0].hist(X_train[col])
    axs[0].set_title('Before scaler')
    axs[1].hist(X_train_n[col])
    axs[1].set_xlabel(col)
    axs[1].set_title('After scaler')
    plt.show()

#Na temelju dobivenih parametara skaliranja ˇ transformirajte ulazne velicine skupa podataka za testiranje.
X_test_n = pd.DataFrame(sc.transform(X_test), columns=X_test.columns, index=X_test.index)
#linearni model i vrijednosti 
linearModel=lm.LinearRegression()
linearModel.fit(X_train_n, y_train)
print(f'Parametri modela: {linearModel.coef_}')
print(f'Intercept parametar: {linearModel.intercept_}')

y_prediction = linearModel.predict(X_test_n) #vraca numpy array
print(f'Mean squared error: {mean_squared_error(y_test, y_prediction)}')
print(f'Mean absolute error: {mean_absolute_error(y_test, y_prediction)}')
print(f'Mean absolute percentage error: {mean_absolute_percentage_error(y_test, y_prediction)}%')
print(f'R2 score: {r2_score(y_test, y_prediction)}')

#Prikažite ˇpomocu dijagrama raspršenja odnos izme ´ du stvarnih vrijednosti izlazne veli ¯ cine i procjene ˇdobivene modelom.
plt.scatter(X_test_n['Fuel Consumption City (L/100km)'],y_test, c='b', label='Real values', s=5)
plt.scatter(X_test_n['Fuel Consumption City (L/100km)'],y_prediction, c='r', label='Prediction', s=5)

#Na temelju rješenja prethodnog zadatka izradite model koji koristi i kategoricku ˇvarijable „Fuel Type“ kao ulaznu velicinu. Pri tome koristite 1-od-K kodiranje kategori ˇ ckih ˇvelicina. Radi jednostavnosti nemojte skalirati ulazne veli ˇ cine. Komentirajte dobivene rezultate. ˇKolika je maksimalna pogreška u procjeni emisije C02 plinova u g/km? O kojem se modeluvozila radi?
data = pd.read_csv('data_C02_emission.csv')
ohe=OneHotEncoder()
fuelTypeEncoded=ohe.fit_transform(data[['Fuel Type']]).toarray() #OneHotEncoder.fit_transform ocekuje 2d array(dataframe[[stupac(i)]]), ne moze 1d (series[stupac])
data[ohe.categories_[0]]=fuelTypeEncoded #kategorije su imena stupaca, pohranjene u listu lista kategorija atributa categories_, zato [0], tj. jedina lista kategorija iz prosle linije dobivena
output_variable='CO2 Emissions (g/km)'
y=data[output_variable].copy()
X = data.drop('CO2 Emissions (g/km)', axis=1)
X_train_all , X_test_all , y_train , y_test = train_test_split (X, y, test_size = 0.2, random_state =1)
#potrebne sve velicine iz dataframe za laksi pronalazak modela kasnije

#izdvajanje numerickih velicina
input_variables=['Engine Size (L)','Cylinders','Fuel Consumption City (L/100km)','Fuel Consumption Hwy (L/100km)','Fuel Consumption Comb (L/100km)','Fuel Consumption Comb (mpg)','D', 'E', 'X', 'Z']
X_train = X_train_all[input_variables]
X_test = X_test_all[input_variables]

linearModel = lm.LinearRegression()
linearModel.fit(X_train,y_train)
y_prediction = linearModel.predict(X_test)
plt.scatter(X_test['Fuel Consumption City (L/100km)'],y_test, c='b',label='Real values', s=5)
plt.scatter(X_test['Fuel Consumption City (L/100km)'],y_prediction, c='r',label='Prediction', s=5)
plt.xlabel('Fuel Consumption City (L/100km)')
plt.ylabel('CO2 Emissions (g/km)')
plt.legend()
plt.show()

maxError = max_error(y_test,y_prediction)
print(f"Model vozila s najvecom greskom u predvidanju: {X_test_all[abs(y_test-y_prediction)==maxError]['Model'].iloc[0]}")


#klasifikacija logistička regresija,lv5
#model logističke regresije
logisticRegression = LogisticRegression()
logisticRegression.fit(X_train, y_train)
#parametri modela logističke regresije
bias=logisticRegression.intercept_ #teta0, coef vraca samo parametre uz ulazne velicine (za linearnu isto! (lv4))
coefs=logisticRegression.coef_    #provjeriti pravac odluke
print(coefs.shape)

#granica odluke
colors=['blue', 'red']
a = -coefs[0,0]/coefs[0,1]
c = -bias/coefs[0,1]
x1x2min = X_train.min().min()-0.5
x1x2max = X_train.max().max()+0.5
xd = np.array([x1x2min, x1x2max]) #za pravac dovoljne dvije tocke
yd = a*xd + c
plt.plot(xd, yd, linestyle='--')
plt.fill_between(xd, yd, x1x2min, color='red', alpha=0.2) #1
plt.fill_between(xd, yd, x1x2max, color='blue', alpha=0.2) #0
plt.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap=matplotlib.colors.ListedColormap(colors), edgecolor="white")
plt.xlim(x1x2min, x1x2max)
plt.ylim(x1x2min, x1x2max)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Podaci za treniranje i granica odluke')
cbar=plt.colorbar(ticks=[0,1])
plt.show()

#Provedite klasifikaciju skupa podataka za testiranje pomocu izgra ´ denog modela e i prikažite matricu zabune na testnim podacima
y_prediction=logisticRegression.predict(X_test)
cm=confusion_matrix(y_test,y_prediction)#matrica zabune
disp=ConfusionMatrixDisplay(cm)
disp.plot()
plt.title('Matrica zabune')
plt.show()
#Izra ˇ cunate to ˇ cnost, ˇpreciznost i odziv na skupu podataka za testiranje.
print(f'Točnost: {accuracy_score(y_test,y_prediction)}')
print(f'Preciznost: {precision_score(y_test,y_prediction)}')
print(f'Odziv: {recall_score(y_test,y_prediction)}')

#dobro klasificirane vrijednosti i loše

colorsEvaluation=['black', 'green']
plt.scatter(X_test[:,0], X_test[:,1], c=y_test==y_prediction, cmap=matplotlib.colors.ListedColormap(colorsEvaluation)) #redom, false black, true green
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Točnost predikcije na podacima za testiranje')
cbar=plt.colorbar(ticks=[0,1])
cbar.ax.set_yticklabels(['Netočno','Točno'])
plt.show()

#lv6,k najbližih susjeda
#dataframe pretvori u numpy
X = data[["Age","EstimatedSalary"]].to_numpy()
y = data["Purchased"].to_numpy()

#model,skaliranje,knn
# podijeli podatke u omjeru 80-20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state = 10)

# skaliraj ulazne velicine, VRLO VAZNO KOD KNN-a i SVM-a!!!!!!!!!!!!!!!!!!!!!!!!
sc = StandardScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform(X_test)

# Model logisticke regresije
LogReg_model = LogisticRegression(penalty='none') 
LogReg_model.fit(X_train_n, y_train)

# Evaluacija modela logisticke regresije
y_train_p = LogReg_model.predict(X_train_n)
y_test_p = LogReg_model.predict(X_test_n)
#točnost podataka za učenje  i testnih podataka
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p))))

#granica odluke lv6 zad 1
#knn sa k=5
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_n,y_train)
y_train_prediction_knn = knn_model.predict(X_train_n)
y_test_prediction_knn = knn_model.predict(X_test_n)
print("KNN: (K=5) ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_prediction_knn))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_prediction_knn))))

#Pomocu unakrsne validacije odredite optimalnu vrijednost hiperparametra ´ Kalgoritma KNN za podatke iz Zadatka 1.
averageAccuracy = []
for i in range(1,51):
    scores_knn = cross_val_score(KNeighborsClassifier(n_neighbors=i), X=X_train_n, y=y_train, cv=5, scoring='accuracy')
    averageAccuracy.append(scores_knn.mean()) 
plt.plot(range(1,51),averageAccuracy)
plt.xlabel('Broj susjeda K')
plt.ylabel('Točnost')
plt.show()
print(f'Optimalan parametar ima prosječnu točnost {max(averageAccuracy)} i iznosi K={averageAccuracy.index(max(averageAccuracy))+1} (KNN)')

#drugi način za k i iscrtavanje
""" #ISCRTAVANJE I RAČUNANJE OPTIMALNOG HIPERPARAMETRA K ZA KNN (2. način)
param_grid_knn = {'n_neighbors': np.linspace(1,50, num=50, dtype=np.int64)}
knn_grid = GridSearchCV(KNeighborsClassifier(),param_grid_knn,cv=5,scoring='accuracy')
knn_grid.fit(X_train_n,y_train)
scores_knn=knn_grid.cv_results_
print(f'Rezultati GridSearcha za KNN za dane parametre:\n{pd.DataFrame(scores_knn)}')
print(f'Optimalan parametar je {knn_grid.best_params_}(broj susjeda) uz točnost od {knn_grid.best_score_}')
plt.plot(param_grid_knn['n_neighbors'], scores_knn['mean_test_score'])
plt.xlabel('Broj susjeda K')
plt.ylabel('Točnost')
plt.show()
"""
#svm model zad 6.5.3 u tom lv

#lv7,grupiranje podataka k srednjih vrijednosti algoritam
#k najbližih susjeda algoritam
kmeans = KMeans(n_clusters=3, init ='random')
kmeans.fit(X)
labels = kmeans.predict(X)
plt.figure()
plt.scatter(X[:,0],X[:,1], c=labels, cmap='viridis')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Grupirani podatkovni primjeri')
plt.show()
#neispravnim postavljanjem broja k dobija se previše ili premalo grupa
#kmeans kod nekih primjera ne grupira kako treba jer pretpostavlja da su grupe sferične, podjednake velicine i slicne gustoce,
#ne radi dobro s grupama nepravilnih oblika (jer radi na principu udaljenosti) (uz primjenu optimalnih vrijednosti k)
#kada flagc=1, radi dobro jer su grupe sfericne

#binarna slika,kvantizacija zad 2 lv7

#lv8 umjetne neuronske mreže BUDE TAKO NA ISPITU !!

#upoznavanje sa podacima,koliko za učenje i testiranje
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()#čitanje iz MINST
print(f'Broj primjera za ucenje: {len(X_train)}')
print(f'Broj primjera za testiranje: {len(X_test)}')
#moze se i skalirat sa X_train_s = X_train.astype("float32")/255 (istovjetno i X_test)
#ulazni podaci imaju oblik (broj primjera,28,28)(svaka slika je 28x28 piksela), svaki piksel predstavljen brojem 0-255
#izlazna velicina kodirana na nacin da su znamenke predstavljene brojevima 0-9
#svaka slika(primjer)-2d matrica, 28x28

#prikaz random slike i oznake
X_train_reshaped = np.reshape(X_train,(len(X_train),X_train.shape[1]*X_train.shape[2])) #umjesto len(X_train) moze i X_train.shape[0]
X_test_reshaped = np.reshape(X_test,(len(X_test),X_test.shape[1]*X_test.shape[2]))      #umjesto len(X_test) moze i X_test.shape[0]
plt.imshow(X_train[7,:,:])   #slike se prikazuju normalnim 2d poljem
plt.title(f'Slika broja {y_train[7]}')
plt.show()

#izrada mreze i ispis detalja,2 skrivena sloja i jedan izlazni-dense funkcija za sve,units su br neurona
model = keras.Sequential()
model.add(layers.Input(shape=(784,)))
model.add(layers.Dense(units=100, activation="relu"))
model.add(layers.Dense(units=50, activation="relu"))
model.add(layers.Dense(units=10, activation="softmax"))
model.summary()
#oneHotEncoding izlaza, da sve bude prema skici u predlosku, za ovo u kerasu postoji isto funkcija y_train = keras.utils.to_categorical(y_train, num_classes=10)
oh=OneHotEncoder()
y_train_encoded = oh.fit_transform(np.reshape(y_train,(-1,1))).toarray() #OneHotEncoder trazi 2d array, pa treba reshape (-1,1), tj (n,1),
y_test_encoded = oh.transform(np.reshape(y_test,(-1,1))).toarray() #-1 znaci sam skontaj koliko, mora toarray() obavezno kod onehotencodera

#podesavanje parametara treninga
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy",])
history = model.fit(X_train_reshaped , y_train_encoded, batch_size=32, epochs=20, validation_split=0.1)

#evaluacija i ispis 
score = model.evaluate(X_test_reshaped, y_test_encoded, verbose=0)
for i in range(len(model.metrics_names)):
    print(f'{model.metrics_names[i]} = {score[i]}')

#predict i matrica zabune
y_predictions = model.predict(X_test_reshaped)  #vraca za svaki primjer vektor vjerojatnosti pripadanja svakoj od 10 klasa (softmax) (10 000,10)
y_predictions = np.argmax(y_predictions, axis=1)  #vraća polje indeksa najvecih elemenata u svakom pojedinom retku (1d polju) (0-9) (10 000,) - 1d polje
cm = confusion_matrix(y_test, y_predictions)    #zbog prethodnog koraka, usporedba s y_test, a ne encoded
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

#spremanje modela
model.save('Model/')

#ucitavanje modela
model = load_model('Model/')
model.summary()
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
X_test_reshaped = np.reshape(X_test,(len(X_test),X_test.shape[1]*X_test.shape[2])) #za predikciju

#predikcija, za prikaz lose klasificiranih
y_predictions = model.predict(X_test_reshaped) 
y_predictions = np.argmax(y_predictions, axis=1)

#prikaz nekih krivih predikcija
wrong_predictions = y_predictions[y_predictions != y_test]   #krive predikcije modela
wrong_predictions_correct = y_test[y_predictions != y_test]  #ispravke krivih predikcija (koje je model promasio i stavio krive)
images_wrong_predicted = X_test[y_predictions != y_test]     #slike se prikazuju 2d poljem, ne 1d
fig, axs = plt.subplots(2,3, figsize=(12,9))
br=0 #brojac za prikaz slike
for i in range(2):
    for j in range(3):
        axs[i,j].imshow(images_wrong_predicted[br])
        axs[i,j].set_title(f'Model predvidio {wrong_predictions[br]}, zapravo je {wrong_predictions_correct[br]}')
        br=br+1
plt.show()

#ucitavanje slike, u MNIST-u su brojevi bijelom bojom na crnoj pozadini pa je potrebna zamjena vrijednosti crne i bijele boje jer je ova slika crna olovka bijela pozadina
img = plt.imread('test.png')[:,:,0]*255   #bude izmedu 0 i 1 pa treba mnozit s 255, bude rgb, pa treba odstranit nepotrebno jer je crno bijelo
img = img.astype('uint8')
img = np.where(img != 255, 255, 0)       #zamjena, da 255 simbolizira gdje nesto pise, a 0 gdje nema nista (kada ucita je obrnuto jer je crni tekst(0) na bijeloj pozadini(255))
img_reshaped = np.reshape(img, (1,img.shape[0]*img.shape[1]))    #mora biti shape (n, broj ulaznih velicina), tj. u ovom slucaju (1,784)

#predikcija
img_prediction = model.predict(img_reshaped)  #vraca za svaki primjer vektor vjerojatnosti pripadanja svakoj od 10 klasa (softmax) (1,10)
img_prediction = np.argmax(img_prediction, axis=1) #vrati (1,) (izdvaja max index u svakom retku pa se dobije 1d)

#prikaz slike
plt.imshow(img)
plt.title(f'Stvarni broj:2, predikcija:{img_prediction[0]}')
plt.show()

#lv9 konvolucijske mreže sve na gitu


#CIJELI ISPIT OD LVP1!!!!!!!!!!!!!!!!!!!
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, recall_score, precision_score
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import load_model

##################################################
# 1. zadatak
##################################################

# učitavanje dataseta
data = np.loadtxt('pima-indians-diabetes.csv', delimiter=',', skiprows=9)
# a)
print(f'Broj mjerenja: {len(data)}')

# b)
data_df = pd.DataFrame(data)
print(f'Broj dupliciranih: {data_df.duplicated().sum()}')
print(f'Broj izostalih: {data_df.isnull().sum()} ')
data_df = data_df.drop_duplicates()
data_df = data_df.dropna(axis=0) #trebalo je i izbacit sve 0 iz BMI
data = data[data[:,5]!=0.0] #ovo je falilo, izbacivanje sve s 0.0 BMI
data_df = pd.DataFrame(data) #kreiranje ponovno data_df ali ovaj put s očišćenim podacima bez redaka s BMI 0.0
print(f'Broj preostalih: {len(data_df)}') 

# c)
plt.scatter(x=data[:, 7], y=data[:, 5])
plt.title('Odnos dobi i BMI')
plt.xlabel('Age(years)')
plt.ylabel('BMI(weight in kg/(height in m)^2)')
plt.show()
# BMI je pretežito izmedu 20 i 40 (kroz cijeli životni vijek, vidljivo je da je vise mjerenja odrađeno na mlađim ženama), uz nekoliko outliera kod kojih je BMI 0 (pogrešno očitanje) i preko 50

# d)
print(f'Minimalni BMI: {data_df[5].min()}')
print(f'Maksimalni BMI: {data_df[5].max()}')
print(f'Srednji BMI: {data_df[5].mean()}')

# e)
print(f'Minimalni BMI (dijabetes): {data_df[data_df[8]==1][5].min()}')
print(f'Maksimalni BMI (dijabetes): {data_df[data_df[8]==1][5].max()}')
print(f'Srednji BMI: (dijabetes) {data_df[data_df[8]==1][5].mean()}')

print(f'Broj osoba s dijabetesom: {len(data_df[data_df[8]==1])}')

print(f'Minimalni BMI (nema dijabetes): {data_df[data_df[8]==0][5].min()}')
print(f'Maksimalni BMI (nema dijabetes): {data_df[data_df[8]==0][5].max()}')
print(f'Srednji BMI: (nema dijabetes) {data_df[data_df[8]==0][5].mean()}')

# Ljudi s dijabetesom u prosjeku imaju veći BMI, što je logično zbog posljedica same bolesti, maksimalni BMI osobe s dijabetesom je znatno veći nego one bez, a minimalni nije referentan jer je 0 u oba slučaja (nemoguće)

##################################################
# 2. zadatak
##################################################

# učitavanje dataseta
data_df = pd.DataFrame(data, columns=['num_pregnant', 'plasma', 'blood_pressure',
                       'triceps', 'insulin', 'BMI', 'diabetes_function', 'age', 'diabetes']) #koriste se ocisceni podaci za dataframe
X = data_df.drop(columns=['diabetes']).to_numpy()
y = data_df['diabetes'].copy().to_numpy()

# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=5)

# a)
logReg_model = LogisticRegression(max_iter=300)
logReg_model.fit(X_train, y_train)

# b)
y_predictions = logReg_model.predict(X_test)

# c)
disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_predictions))
disp.plot()
plt.show()
# broj TN je 89, TP 36, FN 18 i FP 11, model često osobe koje imaju dijabetes proglasi da nemaju - greška, nedovoljno komentirano

# d)
print(f'Tocnost: {accuracy_score(y_test, y_predictions)}')
print(f'Preciznost: {precision_score(y_test, y_predictions)}')
print(f'Odziv: {recall_score(y_test, y_predictions)}')
# Model točno klasificira ljude kao dijabetičare ili ne s 81% točnost, udio stvarnih dijabetičara u skupu ljudi koje je model proglasio dijabetičarima je 76,5% (preciznost), a model od svih ljudi koji jesu dijabetičari točno predviđa da jesu njih 66,6% (odziv)
# greška, nedovoljno komentirano 
##################################################
# 3. zadatak
##################################################

# učitavanje podataka:
data_df = pd.DataFrame(data, columns=['num_pregnant', 'plasma', 'blood_pressure',
                       'triceps', 'insulin', 'BMI', 'diabetes_function', 'age', 'diabetes']) #koriste se ocisceni podaci za dataframe
X = data_df.drop(columns=['diabetes']).to_numpy()
y = data_df['diabetes'].copy().to_numpy()

# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=5)

# a)
model = keras.Sequential()
model.add(layers.Input(shape=(8,)))
model.add(layers.Dense(units=12, activation="relu"))
model.add(layers.Dense(units=8, activation="relu"))
model.add(layers.Dense(units=1, activation="sigmoid"))
model.summary()

# b)
model.compile(loss="binary_crossentropy",
              optimizer="adam", metrics=["accuracy", ])

# c)
history = model.fit(X_train, y_train, batch_size=10,
                    epochs=150, validation_split=0.1)


# d)
model.save('Model/')

# e)
model = load_model('Model/')
score = model.evaluate(X_test, y_test, verbose=0)
for i in range(len(model.metrics_names)):
    print(f'{model.metrics_names[i]} = {score[i]}')

# f)
y_predictions = model.predict(X_test)
y_predictions = np.around(y_predictions).astype(np.int32)
cm = confusion_matrix(y_test, y_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()
# komentar u pdfu
