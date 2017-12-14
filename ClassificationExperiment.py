from sklearn import datasets as ds
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from random import randrange
import numpy as np

# svm classification

def get_data(path,n_features=None):
    data = ds.load_svmlight_file(path,n_features=n_features)
    return data[0], data[1]

# 损失函数
def loss(X,y,W,b,C=1):

	N,dim = X.shape

	hinge_loss = np.maximum(0, 1-y*(np.dot(X,W)+b) )
	hinge_loss = np.sum(hinge_loss)

	data_loss = 0.5*np.sum( W **2 ) + C * hinge_loss
	data_loss /= N
	return data_loss 

# 梯度函数
def gradient(X,y,w,b,C=1):

	N,dim = X.shape

	margin = 1 - y * (np.dot(X,w)+b ) 
	y_tmp = -y
	y_tmp[margin<0] = 0
	dw =  w + C / N *  np.dot(X.T,y_tmp)  
	db = C / N * np.sum(y_tmp)
	return dw, db


def predict(X,y,w,b,threshold=0):
	pred = np.dot(X,w) + b
	pred[pred>=threshold]=1
	pred[pred<threshold]=-1
	acc = (pred==y).sum() / X.shape[0]
	return acc


def gradientDecent(X_train,y_train,w,b,alpha,num_rounds,X_test,y_test,C=1,batch_size=1000):

	train_loss_history = []
	test_loss_history = []

	for i in range(num_rounds):

		random = list(set(np.random.randint(0,X_train.shape[0],size=batch_size)))
		dx = X_train[random]
		dy = y_train[random]

		w = w - alpha * gradient(dx,dy,w,b,C=C)[0] 
		b = b - alpha * gradient(dx,dy,w,b,C=C)[1] 

		train_loss_history.append( loss(X_train,y_train,w,b,C=C) )
		test_loss_history.append( loss(X_test,y_test,w,b,C=C) )
	return w,b,train_loss_history,test_loss_history

def NAG(X_train,y_train,w,b,alpha,num_rounds,X_test,y_test,C=1,batch_size=1000):
	train_loss_history = []
	test_loss_history = []

	yy = 0.9
	vw = np.zeros(w.shape)
	vb = 0

	for i in range(num_rounds):

		random = list(set(np.random.randint(0,X_train.shape[0],size=batch_size)))
		dx = X_train[random]
		dy = y_train[random]
		w_gt,b_gt = gradient(dx,dy,w-yy*vw ,b-yy*vb ,C=C)

		vw = yy*vw + alpha * w_gt
		w = w - vw

		vb = yy*vb + alpha * b_gt
		b = b - vb

		train_loss_history.append(loss(X_train,y_train,w,b,C=C) )
		test_loss_history.append( loss(X_test,y_test,w,b,C=C) )
	return w,b,train_loss_history,test_loss_history

def RMSProp(X_train,y_train,w,b,alpha,num_rounds,X_test,y_test,C=1,batch_size=1000):
	train_loss_history = []
	test_loss_history = []

	yy = 0.9
	w_Gt = np.zeros(w.shape)
	b_Gt = 0
	e = 1e-9
	gama = 0.001

	for i in range(num_rounds):

		random = list(set(np.random.randint(0,X_train.shape[0],size=batch_size)))
		dx = X_train[random]
		dy = y_train[random]
		w_gt,b_gt = gradient(dx,dy,w,b,C=C)

		w_Gt = yy * w_Gt + (1-yy) * ( w_gt**2) 
		w = w - gama / np.sqrt(w_Gt+e) * w_gt

		b_Gt = yy * b_Gt + (1-yy) * ( b_gt**2) 
		b = b - gama / np.sqrt(b_Gt+e) * b_gt

		train_loss_history.append(loss(X_train,y_train,w,b,C=C) )
		test_loss_history.append( loss(X_test,y_test,w,b,C=C) )
    
	return w,b,train_loss_history,test_loss_history

def AdaDelta(X_train,y_train,w,b,alpha,num_rounds,X_test,y_test,C=1,batch_size=1000):
	train_loss_history = []
	test_loss_history = []

	yy = 0.95
	w_Gt = np.zeros(w.shape)
	b_Gt = 0
	e = 1e-6
	wt = np.zeros(w.shape) 
	bt = 0
	for i in range(num_rounds):

		random = list(set(np.random.randint(0,X_train.shape[0],size=batch_size)))
		dx = X_train[random]
		dy = y_train[random]
		w_gt,b_gt = gradient(dx,dy,w,b,C=C)

		w_Gt = yy * w_Gt + (1-yy) * ( w_gt**2) 
		wdw = - ( np.sqrt(wt+e) / np.sqrt(w_Gt+e) ) * w_gt
		w = w + wdw
		wt = yy * wt + (1-yy) * ( wdw**2) 

		b_Gt = yy * b_Gt + (1-yy) * ( b_gt**2) 
		bdw = - ( np.sqrt(bt+e) / np.sqrt(b_Gt+e) ) * b_gt
		b = b + bdw
		bt = yy * bt + (1-yy) * ( bdw**2) 

		train_loss_history.append(loss(X_train,y_train,w,b,C=C) )
		test_loss_history.append( loss(X_test,y_test,w,b,C=C) )
    
	return w,b,train_loss_history,test_loss_history


def Adam(X_train,y_train,w,b,alpha,num_rounds,X_test,y_test,C=1,batch_size=1000):
	train_loss_history = []
	test_loss_history = []
	beta1 = 0.9
	yy = 0.999
	gama = 1e-3
	e = 1e-8	

	wm = np.zeros(w.shape)
	w_Gt = np.zeros(w.shape)

	bm = 0
	b_Gt = 0

	for i in range(num_rounds):

		random = list(set(np.random.randint(0,X_train.shape[0],size=batch_size)))
		dx = X_train[random]
		dy = y_train[random]
		w_gt,b_gt = gradient(dx,dy,w,b,C=C)

		wm = beta1 * wm + (1-beta1) * w_gt
		w_Gt = yy * w_Gt + (1-yy) * ( w_gt**2) 
		alp = gama * np.sqrt(1-yy**(i+1) ) / (1-beta1**(i+1) )
		w = w - alp * wm / np.sqrt(w_Gt + e)

		bm = beta1 * bm + (1-beta1) * b_gt
		b_Gt = yy * b_Gt + (1-yy) * ( b_gt**2) 
		alp = gama * np.sqrt(1-yy**(i+1) ) / (1-beta1**(i+1) )
		b = b - alp * bm / np.sqrt(b_Gt + e)


		train_loss_history.append(loss(X_train,y_train,w,b,C=C) )
		test_loss_history.append( loss(X_test,y_test,w,b,C=C) )
    
	return w,b,train_loss_history,test_loss_history


def train(X,y,X_test,y_test):
	m = X.shape[1]
	alpha=0.001
	num_rounds=700
	init_w = np.zeros(m).T
	init_b = 0
	C = 50
	batch_size = 3000
	w,b,train_loss_history,grad_loss_history = gradientDecent(X,y,init_w,init_b,alpha,num_rounds,X_test,y_test,C=C,batch_size=batch_size)
	print('grad acc' , predict(X_test,y_test,w,b))
	w,b,train_loss_history,NAG_loss_history = NAG(X,y,init_w,init_b,alpha,num_rounds,X_test,y_test,C=C,batch_size=batch_size)
	print('NAG acc' , predict(X_test,y_test,w,b))
	w,b,train_loss_history,RMSProp_loss_history = RMSProp(X,y,init_w,init_b,alpha,num_rounds,X_test,y_test,C=C,batch_size=batch_size)
	print('RMSProp acc' , predict(X_test,y_test,w,b))
	w,b,train_loss_history,AdaDelta_loss_history = AdaDelta(X,y,init_w,init_b,alpha,num_rounds,X_test,y_test,C=C,batch_size=batch_size)
	print('AdaDelta acc' , predict(X_test,y_test,w,b))
	w,b,train_loss_history,Adam_loss_history = Adam(X,y,init_w,init_b,alpha,num_rounds,X_test,y_test,C=C,batch_size=batch_size)
	print('Adam acc' , predict(X_test,y_test,w,b))
	plt.plot(np.arange(num_rounds),grad_loss_history, label='train_loss')
	plt.plot(np.arange(num_rounds),NAG_loss_history,label='NAG loss')
	plt.plot(np.arange(num_rounds),RMSProp_loss_history,label='RMSProp loss')
	plt.plot(np.arange(num_rounds),AdaDelta_loss_history,label='AdaDelta loss')
	plt.plot(np.arange(num_rounds),Adam_loss_history,label='Adam loss')

	plt.legend(loc=1)
	plt.xlabel('number_of_rounds')
	plt.ylabel('loss')
	plt.show()

X_train,y_train = get_data('a9a.txt',n_features=123)
X_train = X_train.toarray()
X_test,y_test = get_data('a9a.t',n_features=123)
X_test = X_test.toarray()

train(X_train,y_train,X_test,y_test)

