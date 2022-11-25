import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

#matplotlib画图中中文显示会有问题，需要这两行设置默认字体
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

df=pd.read_csv('iris.csv',header=None,encoding='utf-8')#读入数据文件，前提要保证数据文件和程序文件在同一目录下
data=df.values#去掉数据前的索引并取值
x_data=[lines[0:4] for lines in data]#拿出数据的前四列作为输入
x_data=np.array(x_data,float)#转变成numpy格式
y_data=[lines[4] for lines in data]#取出最后一列的标签

#数据归一化
scaler=MinMaxScaler()
x_data=scaler.fit_transform(x_data)#最小最大归一化MinMaxScaler函数是对列进行处理的，这里对鸢尾花的每种性状进行归一化处理
x_data=x_data.T


#训练集90个样本，测试集60个样本
x_train = np.hstack((x_data[:, 0:30], x_data[:, 50:80],x_data[:, 100:130]))
y_train =y_data[0:30]+y_data[50:80]+y_data[100:130]
x_test = np.hstack((x_data[:, 30:50],x_data[:, 80:100],x_data[:, 130:150]))
y_test = y_data[30:50]+y_data[80:100]+y_data[130:150]
y_train1 =y_data[0:30]+y_data[50:80]+y_data[100:130]
y_test1 =y_data[30:50]+y_data[80:100]+y_data[130:150]



#将训练集和测试集的标签变成独热编码
y_train=np.array(y_train).reshape([-1,1])
Encoder=OneHotEncoder()
Encoder.fit(y_train)
y_train=Encoder.transform(y_train).toarray()
y_train=np.asarray(y_train,dtype=np.int32)
y_train=y_train.T

y_test=np.array(y_test).reshape([-1,1])
Encoder.fit(y_test)
y_test=Encoder.transform(y_test).toarray()
y_test=np.asarray(y_test,dtype=np.int32)
y_test=y_test.T




def sigmoid(z):                                #定义sigmoid函数
    return 1.0/(1+np.exp(-z))




def dsigmoid(z):                            #定义sigmoid导函数
    return z*(1-z)



def relu(z):                  #relu函数
    return np.maximum(0,z)



def drelu(z):               #relu导函数
  return np.where(z>0,1,0)




def tanh(z):                  #tanh函数
    return (np.exp(z)-np.exp(-z))/(np.exp(-z)+np.exp(z))



def dtanh(z):               #tanh导函数
    return 1-z*z



#softmax分类器
def softmax(x):
    k=x.shape[1]
    M=np.max(x, axis=0)
    c=x-M
    ex=np.exp(c.T)
    ex_sum=np.sum(ex,axis=1)
    count=[]
    for i in range(0,k):
        count.append(ex[i]/ex_sum[i])
    count=np.array(count).T
    return count

#准确个数统计函数
def accuracy_count(predict_value,real_label):
    n_cols=predict_value.shape[1]
    output = np.argmax(predict_value, axis=0)
    count = 0
    for i in range(0, n_cols):
        if output[i] ==real_label[i]:
            count = count + 1
    return count


n,m=np.shape(x_train)
n_x=4
ec=6#AE的神经元个数
n_y=4
y=3#输出层神经元个数
np.random.seed(2)
w1 = np.random.randn(ec, n_x)#6*4生成符合独立同分布的随机矩阵
b1 = np.zeros((ec, 1))#6*1
w2 = np.random.randn(n_y, ec)#4*6
b2 = np.zeros((n_y, 1))#4*1
w3=np.random.randn(y,ec)#3*6
b3=np.zeros((y,1))#3*1


def forward(x_train,w1, b1, w2, b2):
    z1 = np.dot(w1, x_train) + b1#6*120
    a1 =sigmoid(z1)#6*120
    z2 = np.dot(w2, a1) + b2#4*120
    a2 =sigmoid(z2)#4*120
    return z1, a1, z2, a2




def backward( a2, a1, w2, x_train):
    dz2 = (a2-x_train)*dsigmoid(a2)#4*120
    dw2 = np.dot(dz2, a1.T)#4*6
    db2 = np.sum(dz2, axis=1, keepdims=True)/m#4*1
    dz1 = np.dot(w2.T, dz2)*dsigmoid(a1)#6*120
    dw1 = np.dot(dz1, x_train.T)#6*4
    db1 = np.sum(dz1, axis=1, keepdims=True)/m#6*1
    return dz2, dw2, db2, dz1, dw1, db1


lr1 = 0.05
number =3000
error_data=[]
#预训练部分
for i in range(0, number):
    z1, a1, z2, a2 =forward(x_train,w1, b1, w2, b2)
    error = np.sum(0.5*(a2 - x_train)**2)/m
    error_data.append(error)
    if error<1e-5:
        break
    else:
        dz2, dw2, db2, dz1, dw1, db1 =backward( a2, a1, w2, x_train)
        w1 = w1 - lr1 * dw1
        w2 = w2 - lr1 * dw2
        b1 = b1 - lr1 * db1
        b2 = b2 - lr1 * db2
print('predict train loss is:{}:'.format(error))



#微调和预测的前向和反向结构
def forward1(x_train,w1, b1, w3, b3):
    z1 = np.dot(w1, x_train) + b1#6*120
    a1 =sigmoid(z1)#6*120
    z2 = np.dot(w3, a1) + b3#3*120
    a2 =softmax(z2)#3*120
    return z1, a1, z2, a2


def backward1( a2, a1, w3, y_train):
    dz2 = (a2-y_train)#3*120
    dw3 = np.dot(dz2, a1.T)#3*6
    db3 = np.sum(dz2, axis=1, keepdims=True)/m#3*1
    dz1 = np.dot(w3.T, dz2)*dsigmoid(a1)#6*120
    dw1 = np.dot(dz1, x_train.T)#6*4
    db1 = np.sum(dz1, axis=1, keepdims=True)/m#6*1
    return dw3,db3,dw1,db1


lr2=0.01
number1=7000
loss_data=[]
train_accuracy_rate=[]
#全局微调
for i in range(0,number1):
    z1_1, a1_1, z2_2, a2_2=forward1(x_train,w1, b1, w3, b3)
    M = np.max(z2_2, axis=0)
    loss=np.sum(-y_train*((z2_2-M)-np.log(np.sum(np.exp(z2_2-M),axis=0))))/m
    loss_data.append(loss)
    train_accuracy_rate.append((accuracy_count(a2_2, y_train1)/m)*100)
    dw3_3,db3_3,dw1_1,db1_1=backward1(a2_2, a1_1, w3, y_train)
    w1 = w1 - lr2 * dw1_1
    w3 = w3 - lr2 * dw3_3
    b1 = b1 - lr2 * db1_1
    b3 = b3 - lr2 * db3_3
print('global fine-tuning loss is:{}:'.format(loss))
print('global fine-tuning train accuracy rate  is:{}:'.format((accuracy_count(a2_2, y_train1)/m)*100))






#预测

z1_test,a1_test,z2_test,a2_test=forward1(x_test,w1, b1, w3, b3)
n_cols=a2_test.shape[1]
M = np.max(z2_test, axis=0)
loss1=np.sum(-y_test*((z2_test-M)-np.log(np.sum(np.exp(z2_test-M),axis=0))))/n_cols
print('forecast loss is:{}:'.format(loss1))
count=accuracy_count(a2_test,y_test1)
acc=(count/n_cols)*100
print('forecast accuracy rate is:{}%'.format(acc))






#绘制预训练loss，微调loss和微调精度曲线
fig1,axs1=plt.subplots(figsize=(10,6),dpi=100)#创建画布
fig2,axs2=plt.subplots(figsize=(10,6),dpi=100)
fig3,axs3=plt.subplots(figsize=(10,6),dpi=100)
fig4,axs4=plt.subplots(figsize=(10,6),dpi=100)
fig5,axs5=plt.subplots(figsize=(10,6),dpi=100)
fig6,axs6=plt.subplots(figsize=(10,6),dpi=100)
fig7,axs7=plt.subplots(figsize=(10,6),dpi=100)
x1=np.arange(number)
x2=np.arange(number1)
x3=np.arange(m)
axs3.set_yticks(range(0,101,5))
axs1.plot(x1,error_data,c='red',label='预训练loss')
axs2.plot(x2,loss_data,c='green',label='全局微调loss')
axs3.plot(x2,train_accuracy_rate,c='blue',label='微调精度')
axs4.plot(x3,a2[0],color='green',label='重构值')
axs4.plot(x3,x_train[0],color='red',label='真实值')
axs5.plot(x3,a2[1],color='green',label='重构值')
axs5.plot(x3,x_train[1],color='red',label='真实值')
axs6.plot(x3,a2[2],color='green',label='重构值')
axs6.plot(x3,x_train[2],color='red',label='真实值')
axs7.plot(x3,a2[3],color='green',label='重构值')
axs7.plot(x3,x_train[3],color='red',label='真实值')
axs4.set_ylabel('花萼长度')
axs5.set_ylabel('花萼宽度')
axs6.set_ylabel('花瓣长度')
axs7.set_ylabel('花瓣宽度')
axs1.legend(loc='best')
axs2.legend(loc='best')
axs3.legend(loc='best')
axs4.legend(loc='best')
axs5.legend(loc='best')
axs6.legend(loc='best')
axs7.legend(loc='best')
axs1.grid(True,linestyle='--',alpha=0.5)
axs2.grid(True,linestyle='--',alpha=0.5)
axs3.grid(True,linestyle='--',alpha=0.5)
axs4.grid(True,linestyle='--',alpha=0.5)
axs5.grid(True,linestyle='--',alpha=0.5)
axs6.grid(True,linestyle='--',alpha=0.5)
axs7.grid(True,linestyle='--',alpha=0.5)
plt.show()











