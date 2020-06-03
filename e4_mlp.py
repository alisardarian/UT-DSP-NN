#!/usr/bin/env python
# coding: utf-8

# # DSP HW #4 (PART 2/3)
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
        wave_array_splits = np.array(np.array_split(wave_array, 50))
        for frame_array in wave_array_splits: # 50 iterate
            frame_energy = np.sum(np.absolute(frame_array) ** 2)
            frame_power = frame_energy/len(frame_array)
            frame_zcr = len(np.nonzero(np.diff(frame_array > 0))[0])
            wave_features.append(frame_power) # powerof frame feature
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
training_inputs = combined[:,:100]
training_labels = combined[:,100:]
print ("training data shape : ", training_inputs.shape)
print ("training labels shape : ", training_labels.shape)


# ### به علت حالت ترتیبی داده‌ها و تعداد زیاد داده‌های موجود در هر کلاس، مخلوط کردن داده‌های آموزشی به طور غیریکنواخت باعث بهبود فرآیند آموزش شبکه و جلوگیری از وابستگی آن به یک کلاس خاص می‌شود، به طوری که در این پیاده‌سازی باعث افزایش 40 درصدی دقت شد

# ## :تعیین ساختار شبکه

# In[6]:


nnplt.Plot([100, 50, 10], size=30) # just structure visualization for better understanding


# ## :پیاده‌سازی شبکه

# In[7]:


np.random.seed(None) # based on system's time


# In[8]:


class MLPNeuralNetwork():
    def __init__(self, input_neurons, hidden_neurons, output_neurons):
        self.weights_1 = np.random.uniform(low=-0.1, high=0.1, size=(input_neurons, hidden_neurons)) # weights initialization
        self.weights_2 = np.random.uniform(low=-0.1, high=0.1, size=(hidden_neurons, output_neurons)) # weights initialization
        self.delta_w_1 = np.zeros((input_neurons, hidden_neurons))
        self.delta_w_2 = np.zeros((hidden_neurons, output_neurons))
        self.bias_1 = np.random.uniform(low=-0.1, high=0.1, size=(1, hidden_neurons)) # bias initialization
        self.bias_2 = np.random.uniform(low=-0.1, high=0.1, size=(1, output_neurons)) # bias initialization
        self.delta_b_1 = np.zeros((1, hidden_neurons))
        self.delta_b_2 = np.zeros((1, output_neurons))
        self.x = np.zeros((1, input_neurons))
        self.z = np.zeros((1, hidden_neurons))
        self.y = np.zeros((1, output_neurons))
        self.z_in = np.zeros((1, hidden_neurons))
        self.y_in = np.zeros((1, output_neurons))
        self.t = np.zeros((1, output_neurons))
        self.input_neurons = input_neurons
        self.hidden_neurons = hidden_neurons
        self.learning_rate = 0.01
    
    def activation_function(self, x):
        return (1 - np.exp(-x)) / (1 + np.exp(-x)) # bipolar sigmoid
    
    def activation_function_d(self, x):
        return 0.5 * (1 + self.activation_function(x)) * (1 - self.activation_function(x))
    
    def train(self, inputs, labels, epochs):
        for iteration in range(epochs):
            inputs_count = len(inputs)
            for i in range(inputs_count):
                self.x = inputs[i]
                self.t = labels[i]
                # feed forward:
                self.feed_forward()
                # backward propagation:
                self.back_propagation()
                # update weights:
                self.update_weights()
                
        print ("Trained!")
            
    def feed_forward(self):
        self.z_in = np.dot(self.x.reshape((1,self.input_neurons)), self.weights_1)+self.bias_1
        self.z = self.activation_function(self.z_in)
        self.y_in = np.dot(self.z.reshape((1,self.hidden_neurons)), self.weights_2)+self.bias_2
        self.y = self.activation_function(self.y_in)
        
    def back_propagation(self):
        delta_2 = (self.t - self.y)*self.activation_function_d(self.y_in)
        self.delta_w_2 = np.dot(self.z.T, delta_2)*self.learning_rate
        self.delta_b_2 = delta_2*self.learning_rate
        delta_1 = (np.dot(delta_2, self.weights_2.T))*self.activation_function_d(self.z_in)
        self.delta_w_1 = np.dot(self.x.reshape((1,self.input_neurons)).T, delta_1)*self.learning_rate
        self.delta_b_1 = delta_1*self.learning_rate
        
    def update_weights(self):
        self.weights_2 += self.delta_w_2
        self.bias_2 += self.delta_b_2
        self.weights_1 += self.delta_w_1
        self.bias_1 += self.delta_b_1
        
    def evaluate(self, input):
        z_in = np.dot(input.reshape((1,self.input_neurons)), self.weights_1)+self.bias_1
        z = self.activation_function(z_in)
        y_in = np.dot(z.reshape((1,self.hidden_neurons)), self.weights_2)+self.bias_2
        y = self.activation_function(y_in)
        return y


# ## :آموزش شبکه

# In[9]:


nn = MLPNeuralNetwork(100, 50, 10)
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
        wave_array_splits = np.array(np.array_split(wave_array, 50))
        for frame_array in wave_array_splits: # 50 iterate
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
    output = nn.evaluate(test_input)
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
    output = nn.evaluate(train_input)
    if (np.argmax(train_label) == np.argmax(output)):
        correct_recognitions += 1
accuracy = correct_recognitions/train_count
print("Accuracy on Train Data = ", accuracy*100, "%")


# # :نتیجه

# ## همانطور که مشاهده می‌شود دقت تشخیص با استفاده از شبکه‌ی پس‌انتشار خطا نسبت به شبکه‌ی آدالاین در شرایط مشابه بیشتر است و با افزایش ویژگی‌ها و تعداد نرون‌های ورودی دیگر دچار مشکل نشده و همچنان همگرا می‌شود 

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
        wave_fft = np.fft.fft(wave_array, 512)
        training_inputs_list_2.append(wave_fft[0:256])
        
        label = np.zeros((10,), dtype=int)
        label[i] = 1
        training_labels_list_2.append(label) # labeling


# ### :نرمال سازی داده‌های ورودی

# In[15]:


training_inputs_2 = np.abs(np.array(training_inputs_list_2))
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
training_inputs_2 = combined_2[:,:256]
training_labels_2 = combined_2[:,256:]
print ("training data shape : ", training_inputs_2.shape)
print ("training labels shape : ", training_labels_2.shape)


# ## :آموزش شبکه

# In[17]:


nn2 = MLPNeuralNetwork(256, 120, 10)
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
        wave_fft = np.fft.fft(wave_array, 512)
        testing_inputs_list_2.append(wave_fft[0:256])

        label = np.zeros((10,), dtype=int)
        label[i] = 1
        testing_labels_list_2.append(label) # labeling


# ### :نرمال سازی داده‌های آزمون

# In[19]:


testing_inputs_2 = np.abs(np.array(testing_inputs_list_2))
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
    output = nn2.evaluate(test_input)
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
    output = nn2.evaluate(train_input)
    if (np.argmax(train_label) == np.argmax(output)):
        correct_recognitions += 1
accuracy = correct_recognitions/train_count
print("Accuracy on Train Data = ", accuracy*100, "%")


# # :نتیجه

# ## دقت شبکه به طور قابل توجهی کاهش می‌یابد، که نشان می‌دهد استفاده از نمونه‌های تبدیل فوریه به تنهایی، علاوه بر پیچیده‌تر شدن شبکه، ویژگی مناسبی برای این نوع شبکه نمی‌باشد

# # :قسمت ج

# In[22]:


total_data_inputs = np.concatenate((training_inputs, testing_inputs))
total_data_labels = np.concatenate((training_labels, testing_labels))
total_count = len(total_data_inputs)


# ## :محاسبه‌ی دقت کلی برای شبکه‌ها با تعداد نرون‌های لایه مخفی مختلف

# In[23]:


def get_total_accuracy(neuralNetwork):
    correct_recognitions = 0
    for t in range(total_count):
        input_data = total_data_inputs[t]
        real_output = total_data_labels[t]
        network_output = neuralNetwork.evaluate(input_data)
        if (np.argmax(real_output) == np.argmax(network_output)):
            correct_recognitions += 1
    return (correct_recognitions/total_count)*100


# ## :آموزش شبکه‌ها و انتخاب شبکه با بهترین دقت

# In[24]:


networks = []
for i in range(20,151,10): # 20 to 150 hidden neurons (10 steps)
    n = MLPNeuralNetwork(100, i, 10) 
    n.train(training_inputs, training_labels, 1000)
    networks.append(n)
networks_accuracy = np.zeros(len(networks))
for i in range(len(networks)):
    networks_accuracy[i] = get_total_accuracy(networks[i])
optimum_netwrok = networks[np.argmax(networks_accuracy)]
print("Optimum Hidden Layers = ", optimum_netwrok.hidden_neurons)


# ## :رسم ماتریس درهم ریختگی برای بهترین حالت

# In[25]:


# Confusion Matrix for optimum network:
confusion = np.zeros((10,10))
for d in range(total_count):
    input_data = total_data_inputs[d]
    real_output = total_data_labels[d]
    network_output = optimum_netwrok.evaluate(input_data)
    confusion[np.argmax(real_output)][np.argmax(network_output)] += 1
confusion = (confusion/total_count)*100 # accuracy
# Draw matrix (heatmap):
fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(confusion)
ax.set_xticks(np.arange(10))
ax.set_yticks(np.arange(10))
ax.set_xticklabels(np.arange(10))
ax.set_yticklabels(np.arange(10))
for i in range(10):
    for j in range(10):
        text = ax.text(j, i, "{:.2f}".format(confusion[i, j]),
                       ha="center", va="center", color="w")
ax.set_title("Confusion Matrix of MLP Network with "+str(optimum_netwrok.hidden_neurons)+" Hidden Neurons")
fig.tight_layout()
plt.show()


# ### ستون‌ها = خروجی شبکه | ردیف‌ها = مقدار واقعی
