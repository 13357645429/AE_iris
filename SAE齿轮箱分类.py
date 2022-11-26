'分类器采用和自编码器同样结构的预训练方法，没有加入正则化'

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

#matplotlib画图中中文显示会有问题，需要这两行设置默认字体
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

#导入每种故障的数据集
normal880_3=np.loadtxt('D:\Desktop\Pycharm_Project\齿轮箱故障数据/880_3/normal880-3.txt')
dianmo880_3=np.loadtxt('D:\Desktop\Pycharm_Project\齿轮箱故障数据/880_3/dianmo880-3.txt')
dianshi880_3=np.loadtxt('D:\Desktop\Pycharm_Project\齿轮箱故障数据/880_3/dianshi880-3.txt')
duanmo880_3=np.loadtxt('D:\Desktop\Pycharm_Project\齿轮箱故障数据/880_3/duanmo880-3.txt')
duanchi880_3=np.loadtxt('D:\Desktop\Pycharm_Project\齿轮箱故障数据/880_3/duanchi880-3.txt')
mosun880_3=np.loadtxt('D:\Desktop\Pycharm_Project\齿轮箱故障数据/880_3/mosun880-3.txt')


#截取每个数据集的样本个数
ending_number1=400
ending_number2=600


extract_train_sample_number=400#从每个数据集中截取出来作为训练集的样本个数
extract_test_sample_number=200#从每个数据集中截取出来作为测试集的样本个数
fault_total_number=6#带上正常状态时，总共的故障类型种数


#为训练集和测试集的每个样本打标签
def label_setting(extract_number):
    count=[]
    for i in range(0,fault_total_number*extract_number):
        if i<extract_number:
            count.append(0)
        elif extract_number<=i<2*extract_number:
            count.append(1)
        elif 2*extract_number<=i<3*extract_number:
            count.append(2)
        elif 3*extract_number<=i<4*extract_number:
            count.append(3)
        elif 4*extract_number<=i<5*extract_number:
            count.append(4)
        else:
            count.append(5)
    return np.array(count).reshape(-1,1)


#生成训练集和测试集的标签
train_label=label_setting(extract_train_sample_number)#2400*1
test_label=label_setting(extract_test_sample_number)#1200*1


#从每种数据集中抽取等量样本组成训练集和测试集
train=np.vstack((normal880_3[:ending_number1],duanmo880_3[:ending_number1],dianshi880_3[:ending_number1],dianmo880_3[:ending_number1],duanchi880_3[:ending_number1],mosun880_3[:ending_number1]))#30000*9
test=np.vstack((normal880_3[ending_number1:ending_number2],duanmo880_3[ending_number1:ending_number2],dianshi880_3[ending_number1:ending_number2],dianmo880_3[ending_number1:ending_number2],duanchi880_3[ending_number1:ending_number2],mosun880_3[ending_number1:ending_number2]))#2400*9


#数据归一化
scaler=MinMaxScaler()
trainT=scaler.fit_transform(train).T#最小最大归一化MinMaxScaler函数是对列进行处理的，这里对鸢尾花的每种性状进行归一化处理
testT=scaler.fit_transform(test).T


#将训练集和测试集的标签变成独热编码
train_one_hot_label=np.array(train_label).reshape([-1,1])
Encoder=OneHotEncoder()
Encoder.fit(train_one_hot_label)
train_one_hot_label=Encoder.transform(train_one_hot_label).toarray()
train_one_hot_label=np.asarray(train_one_hot_label,dtype=np.int32)
train_one_hot_label=train_one_hot_label.T

test_one_hot_label=np.array(test_label).reshape([-1,1])
Encoder.fit(test_one_hot_label)
test_one_hot_label=Encoder.transform(test_one_hot_label).toarray()
test_one_hot_label=np.asarray(test_one_hot_label,dtype=np.int32)
test_one_hot_label=test_one_hot_label.T
#对训练集和测试集标签进行独热编码
# def one_hot(label):
#     count=[]
#     for i in range(0,label.shape[0]):
#         if label[i]==0:
#             count.append([1,0,0,0,0,0])
#         elif label[i]==1:
#             count.append([0,1,0,0,0,0])
#         elif label[i]==2:
#             count.append([0,0,1,0,0,0])
#         elif label[i]==3:
#             count.append([0,0,0,1,0,0])
#         elif label[i]==4:
#             count.append([0,0,0,0,1,0])
#         else:
#             count.append([0,0,0,0,0,1])
#     return np.array(count).T
#
# train_one_hot_label=one_hot(train_label)
# test_one_hot_label=one_hot(test_label)


def sigmoid(z):                                #定义sigmoid函数
    n_rows,n_cols=np.shape(z)
    indices_pos=np.nonzero(z>=0)
    indices_neg=np.nonzero(z<0)
    hu=np.zeros((n_rows,n_cols))
    hu[indices_pos]=1/(1+np.exp(-z[indices_pos]))
    hu[indices_neg]=np.exp(z[indices_neg])/(1+np.exp(z[indices_neg]))
    return hu


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


#准确个数统计器
def accuracy_count(predict_value,real_label):
    n_cols=predict_value.shape[1]
    output = np.argmax(predict_value, axis=0)
    count = 0
    for i in range(0, n_cols):
        if output[i] ==real_label[i]:
            count = count + 1
    return count


#网络结构参数
m,x=np.shape(train)#train是2400*9的矩阵，m是样本个数，x是输入维度
ec1=12#自编码器1是12个神经元
ec2=10#自编码器2是10个神经元
ec3=8#自编码器3是8个神经元
ec4=4#自编码器4是7个神经元
y=fault_total_number#输出神经元6个


#初始化权重和偏置
np.random.seed(1)
w1=np.random.randn(ec1,x)#11*9
b1=np.zeros((ec1,1))#11*1
np.random.seed(1)
w2=np.random.randn(x,ec1)#9*11
b2=np.zeros((x,1))#9*1
np.random.seed(1)
w3=np.random.randn(ec2,ec1)#8*11
b3=np.zeros((ec2,1))#8*1
np.random.seed(1)
w4=np.random.randn(ec1,ec2)#11*8
b4=np.zeros((ec1,1))#11*1
np.random.seed(1)
w5=np.random.randn(ec3,ec2)#7*11
b5=np.zeros((ec3,1))#7*1
np.random.seed(1)
w6=np.random.randn(ec2,ec3)#11*7
b6=np.zeros((ec2,1))#11*1
np.random.seed(1)
w7=np.random.randn(ec4,ec3)#6*7
b7=np.zeros((ec4,1))#6*1
w8=np.random.randn(ec3,ec4)#6*7
b8=np.zeros((ec3,1))#6*1
w9=np.random.randn(y,ec4)#6*7
b9=np.zeros((y,1))#6*1
w10=np.random.randn(ec4,y)#6*7
b10=np.zeros((ec4,1))

#AE
def AE(w1_1,w2_2,b1_1,b2_2,INPUT,lr1,iterations1):
    loss_data=[]
    loss=0
    for i in range(0,iterations1):
        z1_1=np.dot(w1_1,INPUT)+b1_1#12*30000
        a1_1=sigmoid(z1_1)#12*30000
        z2_2=np.dot(w2_2,a1_1)+b2_2#9*30000
        a2_2=sigmoid(z2_2)#9*30000
        loss=np.sum(0.5*(a2_2-INPUT)**2)/m
        loss_data.append(loss)
        if loss<0.1e-10:
            break
        else:
            dz2_2=(a2_2-INPUT)*dsigmoid(a2_2)#9*30000
            dw2=np.dot(dz2_2,a1_1.T)#9*12
            db2=np.sum(dz2_2,axis=1,keepdims=True)/m#9*1
            dz1_1=np.dot(w2_2.T,dz2_2)*dsigmoid(a1_1)#12*30000
            dw1=np.dot(dz1_1,INPUT.T)#12*9
            db1=np.sum(dz1_1,axis=1,keepdims=True)/m#12*1
            w1_1=w1_1-lr1*dw1
            b1_1=b1_1-lr1*db1
            w2_2=w2_2-lr1*dw2
            b2_2=b2_2-lr1*db2
    return w1_1,b1_1,w2_2,b2_2,a1_1,loss,loss_data


def fenleiqi_layer(w1_1,w2_2,b1_1,b2_2,INPUT,lr3,iterations3):
    loss_data=[]
    loss=0
    for i in range(0,iterations3):
        z1_1=np.dot(w1_1,INPUT)+b1_1#12*30000
        a1_1=sigmoid(z1_1)#12*30000
        z2_2=np.dot(w2_2,a1_1)+b2_2
        a2_2=softmax(z2_2)
        M = np.max(z2_2, axis=0)
        loss = np.sum(-INPUT * ((z2_2 - M) - np.log(np.sum(np.exp(z2_2 - M), axis=0)))) / m
        loss_data.append(loss)
        if loss<0.1e-10:
            break
        else:
            dz2_2=a2_2-INPUT
            dw2_2=np.dot(dz2_2,a1_1.T)#12*9
            db2_2=np.sum(dz2_2,axis=1,keepdims=True)/m#12*1
            dz1_1=np.dot(w2_2.T, dz2_2) * dsigmoid(a1_1)
            dw1_1=np.dot(dz1_1,INPUT.T)
            db1_1 = np.sum(dz1_1, axis=1, keepdims=True) / m
            w1_1=w1_1-lr3*dw1_1
            w2_2 = w2_2 - lr3 * dw2_2
            b2_2 = b2_2 - lr3 * db2_2
            b1_1=b1_1-lr3*db1_1
    return w1_1,b1_1,w2_2,b2_2,loss,loss_data


#全局微调
def global_fine_tuning(w1_1,w2_2,w3_3,w4_4,w5_5,b1_1,b2_2,b3_3,b4_4,b5_5,INPUT,lr2,iterations2):
    loss_data = []
    train_accuracy_rate = []
    loss=0
    for i in range(0,iterations2):
        z1_1 = np.dot(w1_1, INPUT) + b1_1
        a1_1 = sigmoid(z1_1)
        z2_2=np.dot(w2_2,a1_1)+b2_2
        a2_2=sigmoid(z2_2)
        z3_3=np.dot(w3_3,a2_2)+b3_3
        a3_3 = sigmoid(z3_3)
        z4_4 = np.dot(w4_4, a3_3) + b4_4
        a4_4 = sigmoid(z4_4)
        z5_5 = np.dot(w5_5, a4_4) + b5_5
        a5_5=softmax(z5_5)
        M = np.max(z5_5, axis=0)
        loss = np.sum(-train_one_hot_label * ((z5_5 - M) - np.log(np.sum(np.exp(z5_5 - M), axis=0)))) / m
        loss_data.append(loss)
        train_accuracy_rate.append((accuracy_count(a5_5, train_label) / m) * 100)
        if loss<0.1e-5:
            break
        else:
            dz5_5 = a5_5 - train_one_hot_label
            dw5_5 = np.dot(dz5_5, a4_4.T)
            db5_5 = np.sum(dz5_5, axis=1, keepdims=True) / m
            dz4_4=np.dot(w5_5.T, dz5_5) * dsigmoid(a4_4)
            dw4_4 = np.dot(dz4_4, a3_3.T)
            db4_4 = np.sum(dz4_4, axis=1, keepdims=True) / m
            dz3_3 = np.dot(w4_4.T, dz4_4) * dsigmoid(a3_3)
            dw3_3 = np.dot(dz3_3, a2_2.T)
            db3_3 = np.sum(dz3_3, axis=1, keepdims=True) / m
            dz2_2 = np.dot(w3_3.T, dz3_3) * dsigmoid(a2_2)
            dw2_2 = np.dot(dz2_2, a1_1.T)
            db2_2 = np.sum(dz2_2, axis=1, keepdims=True) / m
            dz1_1 = np.dot(w2_2.T, dz2_2) * dsigmoid(a1_1)
            dw1_1 = np.dot(dz1_1, INPUT.T)
            db1_1 = np.sum(dz1_1, axis=1, keepdims=True) / m
            w1_1=w1_1-lr2*dw1_1
            b1_1=b1_1-lr2*db1_1
            w2_2=w2_2-lr2*dw2_2
            b2_2=b2_2-lr2*db2_2
            w3_3=w3_3-lr2*dw3_3
            b3_3=b3_3-lr2*db3_3
            w4_4=w4_4-lr2*dw4_4
            b4_4=b4_4-lr2*db4_4
            w5_5=w5_5-lr2*dw5_5
            b5_5=b5_5-lr2*db5_5
    return w1_1,b1_1,w2_2,b2_2,w3_3,b3_3,w4_4,b4_4,w5_5,b5_5,loss,loss_data,train_accuracy_rate


#forecast
def forecast(w1,b1,w3,b3,w5,b5,w7,b7,w9,b9,testT,test_one_hot_label,test_label):
    z1 = np.dot(w1, testT) + b1
    a1 = sigmoid(z1)
    z2=np.dot(w3,a1)+b3
    a2=sigmoid(z2)
    z3=np.dot(w5,a2)+b5
    a3 = sigmoid(z3)
    z4 = np.dot(w7, a3) + b7
    a4 = sigmoid(z4)
    z5 = np.dot(w9, a4) + b9
    a5=softmax(z5)
    M = np.max(z5, axis=0)
    forecast_loss = np.sum(-test_one_hot_label * ((z5 - M) - np.log(np.sum(np.exp(z5 - M), axis=0)))) / m
    forecast_accuracy_rate=(accuracy_count(a5, test_label) / testT.shape[1]) * 100
    return forecast_loss,forecast_accuracy_rate,a5


#AE1预训练
w1,b1,w2,b2,AE1_code,loss_AE1,loss_data_AE1=AE(w1_1=w1,w2_2=w2,b1_1=b1,b2_2=b2,INPUT=trainT,lr1=0.001,iterations1=3000)
print('AE1 predict loss is:',loss_AE1)
fig1,axs1=plt.subplots(dpi=200)
axs1.plot(np.arange(3000),loss_data_AE1,c='red',label='AE1预训练loss')
axs1.legend(loc='best')


#AE2预训练
w3,b3,w4,b4,AE2_code,loss_AE2,loss_data_AE2=AE(w1_1=w3,w2_2=w4,b1_1=b3,b2_2=b4,INPUT=AE1_code,lr1=0.001,iterations1=3000)
print('AE2 predict loss is:',loss_AE2)
fig2,axs2=plt.subplots(dpi=200)
axs2.plot(np.arange(3000),loss_data_AE2,c='red',label='AE2预训练loss')
axs2.legend(loc='best')


#AE3预训练
w5,b5,w6,b6,AE3_code,loss_AE3,loss_data_AE3=AE(w1_1=w5,w2_2=w6,b1_1=b5,b2_2=b6,INPUT=AE2_code,lr1=0.001,iterations1=3000)
print('AE3 predict loss is:',loss_AE3)
fig3,axs3=plt.subplots(dpi=200)
axs3.plot(np.arange(3000),loss_data_AE3,c='red',label='AE3预训练loss')
axs3.legend(loc='best')


#AE4预训练
w7,b7,w8,b8,AE4_code,loss_AE4,loss_data_AE4=AE(w1_1=w7,w2_2=w8,b1_1=b7,b2_2=b8,INPUT=AE3_code,lr1=0.003,iterations1=3000)
print('AE4 predict loss is:',loss_AE4)
fig4,axs4=plt.subplots(dpi=200)
axs4.plot(np.arange(3000),loss_data_AE4,c='red',label='AE4预训练loss')
axs4.legend(loc='best')


#分类器层预训练
w9,b9,w10,b10,loss_fenleiqi,loss_data_fenleiqi=fenleiqi_layer(w1_1=w9,b1_1=b9,w2_2=w10,b2_2=b10,INPUT=AE4_code,lr3=0.00001,iterations3=3000)
print('fenleiqi predict loss is:',loss_fenleiqi)
fig5,axs5=plt.subplots(dpi=200)
axs5.plot(np.arange(3000),loss_data_fenleiqi,c='red',label='分类器预训练loss')
axs5.legend(loc='best')


#全局微调
w1,b1,w3,b3,w5,b5,w7,b7,w9,b9,loss_fine_tuning,loss_data_fine_tuning,train_accuracy_rate=global_fine_tuning(w1_1=w1,w2_2=w3,w3_3=w5,w4_4=w7,w5_5=w9,b1_1=b1,b2_2=b3,b3_3=b5,b4_4=b7,b5_5=b9,INPUT=trainT,lr2=0.001,iterations2=5000)
print('fine_tuning loss is:',loss_fine_tuning)
print('fine_tuning train accuracy rate is:',train_accuracy_rate[4999])
fig6,axs6=plt.subplots(dpi=200)
axs6.plot(np.arange(5000),loss_data_fine_tuning,c='red',label='全局微调loss')
axs6.legend(loc='best')
fig7,axs7=plt.subplots(dpi=200)
axs7.plot(np.arange(5000),train_accuracy_rate,c='red',label='训练精度')
axs7.legend(loc='best')


#预测
loss_data_forecast,forecast_accuracy_rate,a5_forecast=forecast(w1,b1,w3,b3,w5,b5,w7,b7,w9,b9,testT,test_one_hot_label,test_label)
print('forecast loss is:',loss_data_forecast)
print('forecast accuracy rate is:',forecast_accuracy_rate)


#+ 0.5*l*(np.sum(np.square(w1_1))+np.sum(np.square(w2_2))+np.sum(np.square(w3_3))+np.sum(np.square(w4_4))+np.sum(np.square(w5_5)))/m





