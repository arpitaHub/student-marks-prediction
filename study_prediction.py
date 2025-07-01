from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

hrs=[[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]
marks=[20,30,35,45,50,55,65,75,85,95]

x_train ,x_test, y_train, y_test = train_test_split(hrs, marks, test_size=0.2)
model=LinearRegression()
model.fit (x_train,y_train)
y_pred=model.predict(x_test)
print("actual marks:",y_test)
print("predicted marks:",y_pred)
my_prediction=model.predict([[11]])
print("If I study for 11 hours I will score", my_prediction, "marks.")
print("mean squared error:", model.score(x_test, y_test))


import matplotlib.pyplot as plt
plt.scatter(hrs,marks,color='blue')
plt.plot(hrs,model.predict(hrs),color='red')
plt.xlabel("Study hours")
plt.ylabel("Marks")
plt.title("Study hours vs marks")
plt.show()