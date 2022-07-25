import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

x = np.load('training_loss_200.npy')
y = np.load('validation_loss_200.npy')
epochs = np.arange(len(x))
df = pd.DataFrame({'epochs':epochs,'Training Loss':x,'Validation Loss':y})
z = pd.melt(df,'epochs',value_name='Loss')
sns.set_style('whitegrid')
fig = plt.figure()
ax = plt.axes()
p = sns.lineplot(x='epochs',y='Loss',hue='variable',palette='Set2',data=z,ax=ax)
plt.title('Loss per pixel vs Epochs')
plt.show()