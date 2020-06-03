#!/usr/bin/env python
# coding: utf-8

# # DSP HW #4 (PART 1/3)
# ## Ali Sardarian
# ### UT 2020

# In[1]:


import numpy as np
import thinkdsp as dsp
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import glob
import nnplot as nnplt  # manipulated code for neural network structure visualization


# # :قسمت الف

# ## :پیش‌پردازش داده‌های آموزش

# In[2]:


training_inputs_list = []
training_labels_list = []

for i in range(10):
    files = glob.glob("SpeechRecognition\\TrainSet\\"+str(i)+"\\*.wav")
    for f in files: 
        wave_features = []
        wave = dsp.read_wave(f)
        wave_array = wave.ys
        wave_array_splits = np.array(np.array_split(wave_array, 10))
        for frame_array in wave_array_splits: # 10 iterate
            frame_energy = np.sum(np.absolute(frame_array) ** 2)
            frame_power = frame_energy/len(frame_array)
            frame_zcr = len(np.nonzero(np.diff(frame_array > 0))[0])
            wave_features.append(frame_power) # power of frame feature
            wave_features.append(frame_zcr) # zero crossing rate of frame feature
        training_inputs_list.append(np.array(wave_features))
        label = np.zeros((10,), dtype=int)
        label[i] = 1
        training_labels_list.append(label) # labeling


# ### :نرمال سازی داده‌های ورودی

# In[3]:


def bipolar_normalize(array):
    return (2*(array - array.min())/(array.max() - array.min()))-1


# In[4]:


training_inputs = np.array(training_inputs_list)
training_labels = np.array(training_labels_list)
# normalization:
training_inputs = bipolar_normalize(training_inputs)
training_labels = bipolar_normalize(training_labels)
print ("training data shape : ", training_inputs.shape)
print ("training labels shape : ", training_labels.shape)


# ### :مخلوط کردن داده‌های ورودی

# In[5]:


# shuffle:
combined = np.column_stack((training_inputs, training_labels))
np.random.shuffle(combined)
training_inputs = combined[:,:20]
training_labels = combined[:,20:]
print ("training data shape : ", training_inputs.shape)
print ("training labels shape : ", training_labels.shape)


# ### به علت حالت ترتیبی داده‌ها و تعداد زیاد داده‌های موجود در هر کلاس، مخلوط کردن داده‌های آموزشی به طور غیریکنواخت باعث بهبود فرآیند آموزش شبکه و جلوگیری از وابستگی آن به یک کلاس خاص می‌شود، به طوری که در این پیاده‌سازی باعث افزایش 20 درصدی دقت شد

# ## :تعیین ساختار شبکه

# In[6]:


nnplt.Plot([20, 10], size=15) # just structure visualization for better understanding


# ## :پیاده‌سازی شبکه

# In[7]:


np.random.seed(None) # based on system's time


# In[8]:


class AdalineNeuralNetwork():
    def __init__(self, input_neurons, output_neurons):
        self.weights = np.random.uniform(low=-0.1, high=0.1, size=(input_neurons, output_neurons)) # weights initialization
        self.bias = np.random.uniform(low=-0.1, high=0.1, size=(1, output_neurons)) # bias initialization
        self.learning_rate = 1
        self.input_neurons = input_neurons
    
    def train(self, inputs, labels, epochs):
        for iteration in range(epochs):
            inputs_count = len(inputs)
            for i in range(inputs_count):
                y_out = self.feed(inputs[i])
                error = labels[i] - y_out
                delta_w = np.dot(inputs[i].reshape(self.input_neurons,1), error*self._norm_d(y_out))*self.learning_rate
                delta_b = error*self._norm_d(y_out)*self.learning_rate
                self.weights += delta_w
                self.bias += delta_b
        print ("Trained!")
            
    def feed(self, input):
        return self._norm(np.dot(input.reshape((1,self.input_neurons)), self.weights)+self.bias) # normalize to prevent NaN values
    
    def _norm(self, x):
        return (1 - np.exp(-x)) / (1 + np.exp(-x)) # sigmoid (bipolar)
    
    def _norm_d(self, x): # gets sigmoid output
        return 0.5 * (1 + x) * (1 - x)
        


# ## :آموزش شبکه

# In[9]:


nn = AdalineNeuralNetwork(20, 10)
nn.train(training_inputs, training_labels, 1000)


# ## :ارزیابی شبکه

# ### :پیش‌پردازش داده‌های آزمون

# In[10]:


testing_inputs_list = []
testing_labels_list = []

for i in range(10):
    files = glob.glob("SpeechRecognition\\TestSet\\"+str(i)+"\\*.wav")
    for f in files: 
        wave_features = []
        wave = dsp.read_wave(f)
        wave_array = wave.ys
        wave_array_splits = np.array(np.array_split(wave_array, 10))
        for frame_array in wave_array_splits: # 10 iterate
            frame_energy = np.sum(np.absolute(frame_array) ** 2)
            frame_power = frame_energy/len(frame_array)
            frame_zcr = len(np.nonzero(np.diff(frame_array > 0))[0])
            wave_features.append(frame_power) # powerof frame feature
            wave_features.append(frame_zcr) # zero crossing rate of frame feature
        testing_inputs_list.append(np.array(wave_features))
        label = np.zeros((10,), dtype=int)
        label[i] = 1
        testing_labels_list.append(label) # labeling


# ### :نرمال سازی داده‌های آزمون

# In[11]:


testing_inputs = np.array(testing_inputs_list)
testing_labels = np.array(testing_labels_list)
# normalization:
testing_inputs = bipolar_normalize(testing_inputs)
testing_labels = bipolar_normalize(testing_labels)
print ("testing data shape : ", testing_inputs.shape)
print ("testing labels shape : ", testing_labels.shape)


# ### :دقت تشخیص برای مجموعه‌ی آزمون

# In[12]:


test_count = len(testing_inputs)
correct_recognitions = 0
for t in range(test_count):
    test_input = testing_inputs[t]
    test_label = testing_labels[t]
    output = nn.feed(test_input)
    if (np.argmax(test_label) == np.argmax(output)):
        correct_recognitions += 1
accuracy = correct_recognitions/test_count
print("Accuracy on Test Data = ", accuracy*100, "%")


# ### :دقت تشخیص برای مجموعه‌ی آموزش

# In[13]:


train_count = len(training_inputs)
correct_recognitions = 0
for t in range(train_count):
    train_input = training_inputs[t]
    train_label = training_labels[t]
    output = nn.feed(train_input)
    if (np.argmax(train_label) == np.argmax(output)):
        correct_recognitions += 1
accuracy = correct_recognitions/train_count
print("Accuracy on Train Data = ", accuracy*100, "%")


# # :قسمت ب

# ## :پیش‌پردازش داده‌های آموزش

# In[14]:


training_inputs_list_2 = []
training_labels_list_2 = []

for i in range(10):
    files = glob.glob("SpeechRecognition\\TrainSet\\"+str(i)+"\\*.wav")
    for f in files: 
        wave_features = []
        wave = dsp.read_wave(f)
        wave_array = wave.ys
        wave_array_splits = np.array(np.array_split(wave_array, 50))
        for frame_array in wave_array_splits: # 50 iterate
            frame_energy = np.sum(np.absolute(frame_array) ** 2)
            frame_power = frame_energy/len(frame_array)
            frame_zcr = len(np.nonzero(np.diff(frame_array > 0))[0])
            wave_features.append(frame_power) # powerof frame feature
            wave_features.append(frame_zcr) # zero crossing rate of frame feature
        training_inputs_list_2.append(np.array(wave_features))
        label = np.zeros((10,), dtype=int)
        label[i] = 1
        training_labels_list_2.append(label) # labeling


# ### :نرمال سازی داده‌های ورودی

# In[15]:


training_inputs_2 = np.array(training_inputs_list_2)
training_labels_2 = np.array(training_labels_list_2)
# normalization:
training_inputs_2 = bipolar_normalize(training_inputs_2)
training_labels_2 = bipolar_normalize(training_labels_2)
print ("training data shape : ", training_inputs_2.shape)
print ("training labels shape : ", training_labels_2.shape)


# ### :مخلوط کردن داده‌های ورودی

# In[16]:


# shuffle:
combined_2 = np.column_stack((training_inputs_2, training_labels_2))
np.random.shuffle(combined_2)
training_inputs_2 = combined_2[:,:100]
training_labels_2 = combined_2[:,100:]
print ("training data shape : ", training_inputs_2.shape)
print ("training labels shape : ", training_labels_2.shape)


# ## :آموزش شبکه

# In[17]:


nn2 = AdalineNeuralNetwork(100, 10)
nn2.train(training_inputs_2, training_labels_2, 1000)


# ## :ارزیابی شبکه

# ### :پیش‌پردازش داده‌های آزمون

# In[18]:


testing_inputs_list_2 = []
testing_labels_list_2 = []

for i in range(10):
    files = glob.glob("SpeechRecognition\\TestSet\\"+str(i)+"\\*.wav")
    for f in files: 
        wave_features = []
        wave = dsp.read_wave(f)
        wave_array = wave.ys
        wave_array_splits = np.array(np.array_split(wave_array, 50))
        for frame_array in wave_array_splits: # 50 iterate
            frame_energy = np.sum(np.absolute(frame_array) ** 2)
            frame_power = frame_energy/len(frame_array)
            frame_zcr = len(np.nonzero(np.diff(frame_array > 0))[0])
            wave_features.append(frame_power) # power of frame feature
            wave_features.append(frame_zcr) # zero crossing rate of frame feature
        testing_inputs_list_2.append(np.array(wave_features))
        label = np.zeros((10,), dtype=int)
        label[i] = 1
        testing_labels_list_2.append(label) # labeling


# ### :نرمال سازی داده‌های آزمون

# In[19]:


testing_inputs_2 = np.array(testing_inputs_list_2)
testing_labels_2 = np.array(testing_labels_list_2)
# normalization:
testing_inputs_2 = bipolar_normalize(testing_inputs_2)
testing_labels_2 = bipolar_normalize(testing_labels_2)
print ("testing data shape : ", testing_inputs_2.shape)
print ("testing labels shape : ", testing_labels_2.shape)


# ### :دقت تشخیص برای مجموعه‌ی آزمون

# In[20]:


test_count = len(testing_inputs_2)
correct_recognitions = 0
for t in range(test_count):
    test_input = testing_inputs_2[t]
    test_label = testing_labels_2[t]
    output = nn2.feed(test_input)
    if (np.argmax(test_label) == np.argmax(output)):
        correct_recognitions += 1
accuracy = correct_recognitions/test_count
print("Accuracy on Test Data = ", accuracy*100, "%")


# ### :دقت تشخیص برای مجموعه‌ی آموزش

# In[21]:


train_count = len(training_inputs_2)
correct_recognitions = 0
for t in range(train_count):
    train_input = training_inputs_2[t]
    train_label = training_labels_2[t]
    output = nn2.feed(train_input)
    if (np.argmax(train_label) == np.argmax(output)):
        correct_recognitions += 1
accuracy = correct_recognitions/train_count
print("Accuracy on Train Data = ", accuracy*100, "%")


# # :نتیجه

# ## با افزایش تعداد نرون‌های ورودی و همچنین ویژگی‌های هر فایل، دقت تشخیص شبکه به طور قابل توجهی کاهش یافت و شبکه دچار بیش برازش شد
# 
