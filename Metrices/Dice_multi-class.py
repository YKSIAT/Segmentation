def forward(self, bottom, top):
	self.diff[...] = bottom[1].data
	top[0].data[...] = 1 - self.dice_coef_multi_class(bottom[0], bottom[1])

def backward(self, top, propagate_down,  bottom):
	if propagate_down[1]:
	 	raise Exception("label not diff")
	elif propagate_down[0]:
		a=(-2. * self.diff + self.dice) / self.sum
		bottom[0].diff[...] = a
	else:
	 	raise Exception("no diff")
	# =============================

def dice_coef_multi_class(self, y_pred, y_true):
	n_classes = 5
	smooth=np.float32(1e-7)
	y_true=y_true.data
	y_pred=y_pred.data
	y_pred = np.argmax(y_pred, 1)
	y_pred = np.expand_dims(y_pred,1)

	y_pred=np.ndarray.flatten(y_pred)
	y_true = np.ndarray.flatten(y_true)

	dice = np.zeros(n_classes)
	self.sum = np.zeros([n_classes])
	for i in range(n_classes):
		y_true_i = np.equal(y_true, i)
		y_pred_i = np.equal(y_pred, i)
		self.sum[i] = np.sum(y_true_i) + np.sum(y_pred_i) + smooth
		dice[i] = (2. * np.sum(y_true_i * y_pred_i) + smooth) / self.sum[i]
	self.sum=np.sum(self.sum)
	self.dice=np.sum(dice)
	return self.dice
  # reference website:https://github.com/Lasagne/Recipes/issues/99
