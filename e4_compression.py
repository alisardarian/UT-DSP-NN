#!/usr/bin/env python
# coding: utf-8

# # DSP HW #4 (PART 3/3)
# ## Ali Sardarian
# ### UT 2020

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import glob
from math import log10, sqrt


# # :پیش‌پردازش داده‌های آموزش

# ### :تابع بلوک کردن تصویر

# In[2]:


def block_the_image(arr, nrows, ncols):
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols).swapaxes(1,2).reshape(-1, nrows, ncols))


# ### :تابع بازگردانی تصویر از بلوک‌ها

# In[3]:


def image_form_blocks(blocks, nrows, ncols):
    block_list = [] 
    i=-1
    for r in range(int(nrows/blocks.shape[1])):
        row = []
        for c in range(int(ncols/blocks.shape[2])):
            i+=1
            row.append(blocks[i])
        block_list.append(row)
    return np.block(block_list)


# In[4]:


training_inputs_list = []

files = glob.glob("ImageCompression\\TrainSet\\*.jpg") # 91 files
for f in files: 
    img_features = []
    image = plt.imread(f)
    image_blocks = block_the_image(image, 8, 8) # 1024 blocks per file
    training_inputs_list.append(np.array(image_blocks))


# ### :نرمال سازی داده‌های ورودی

# In[5]:


training_inputs = np.array(training_inputs_list)
# nomalization: (0->1)
training_inputs = training_inputs/256
print ("training data shape : ", training_inputs.shape)


# # :پیاده‌سازی شبکه

# In[6]:


np.random.seed(None) # based on system's time


# In[7]:


class CompressionNeuralNetwork():
    def __init__(self, input_neurons, hidden_neurons, output_neurons):
        self.weights_1 = np.random.uniform(low=-0.1, high=0.1, size=(input_neurons, hidden_neurons)) # weights initialization
        self.weights_2 = np.random.uniform(low=-0.1, high=0.1, size=(hidden_neurons, output_neurons)) # weights initialization
        self.delta_w_1 = np.zeros((input_neurons, hidden_neurons))
        self.delta_w_2 = np.zeros((hidden_neurons, output_neurons))
        self.x = np.zeros((1, input_neurons))
        self.z = np.zeros((1, hidden_neurons))
        self.y = np.zeros((1, output_neurons))
        self.z_in = np.zeros((1, hidden_neurons))
        self.y_in = np.zeros((1, output_neurons))
        self.t = np.zeros((1, output_neurons))
        self.input_neurons = input_neurons
        self.hidden_neurons = hidden_neurons
        self.learning_rate = 0.01
        self.errors = [] # along with epoch nums
    
    def activation_function(self, x):
        return 1 / (1 + np.exp(-x)) # sigmoid
    
    def activation_function_d(self, x):
        return self.activation_function(x) * (1 - self.activation_function(x))
    
    def train(self, inputs, epochs):
        for iteration in range(epochs):
            error = 0;
            for sample in inputs:
                for block in sample:   
                    self.x = block.reshape((1,self.input_neurons))
                    self.t = self.x
                    # feed forward:
                    self.feed_forward()
                    
                    error += self.t - self.y
                    
                    # backward propagation:
                    self.back_propagation()
                    # update weights:
                    self.update_weights()
            self.errors.append(np.mean(error))
                
        print ("Trained!")
            
    def feed_forward(self):
        self.z_in = np.dot(self.x.reshape((1,self.input_neurons)), self.weights_1)
        self.z = self.activation_function(self.z_in)
        self.y_in = np.dot(self.z.reshape((1,self.hidden_neurons)), self.weights_2)
        self.y = self.activation_function(self.y_in)
        
    def back_propagation(self):
        delta_2 = (self.t - self.y)*self.activation_function_d(self.y_in)
        self.delta_w_2 = np.dot(self.z.T, delta_2)*self.learning_rate
        delta_1 = (np.dot(delta_2, self.weights_2.T))*self.activation_function_d(self.z_in)
        self.delta_w_1 = np.dot(self.x.reshape((1,self.input_neurons)).T, delta_1)*self.learning_rate
        
    def update_weights(self):
        self.weights_2 += self.delta_w_2
        self.weights_1 += self.delta_w_1
        
    def evaluate(self, input):
        output = []
        for block in input:
            z_in = np.dot(block.reshape((1,self.input_neurons)), self.weights_1)
            z = self.activation_function(z_in)
            y_in = np.dot(z.reshape((1,self.hidden_neurons)), self.weights_2)
            y = self.activation_function(y_in) # 8x8 output
            output.append(y.reshape(8,8)*256)
        return np.array(output)


# # :حالت اول با 16 نرون مخفی

# ## :آموزش شبکه

# In[8]:


nn = CompressionNeuralNetwork(64, 16, 64)
epochs = 25
nn.train(training_inputs, epochs)


# ### :نمودار خطای شبکه در حین آموزش

# In[9]:


# draw error diagram:
plt.plot(np.arange(epochs), nn.errors)
plt.xlabel('epochs') 
plt.ylabel('error') 
plt.title('Notwork Error Diagram') 
plt.show()


# ## :ارزیابی شبکه

# ### :پیش‌پردازش داده‌های آزمون

# In[10]:


testing_inputs_list = []
test_images = []

files = glob.glob("ImageCompression\\TestSet\\*.jpg") # 5 files
for f in files: 
    img_features = []
    image = plt.imread(f)
    test_images.append(image)
    image_blocks = block_the_image(image, 8, 8) # 1024 blocks per file
    testing_inputs_list.append(np.array(image_blocks))


# ### :نرمال سازی داده‌های آزمون

# In[11]:


testing_inputs = np.array(testing_inputs_list)
# nomalization: (0->1)
testing_inputs = testing_inputs/256
print ("testing data shape : ", testing_inputs.shape)


# In[12]:


def psnr_image(original, compressed):
    mse = np.mean((original - compressed) ** 2) 
    if(mse == 0): 
        return 100
    max_pixel = 255.0
    psnr = 10 * log10(max_pixel**2 / mse) 
    return psnr     


# In[13]:


def show_images(img1, img2, psnr):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,figsize=(7, 4))
    ax1.imshow(img1, cmap = 'gray')
    ax2.imshow(img2, cmap = 'gray')
    ax1.set_title('original')
    ax2.set_title('decompressed') 
    ax1.axis('off')
    ax2.axis('off')
    fig.suptitle("PSNR = "+str(psnr), fontsize=15)
    plt.show()


# In[14]:


test_count = len(testing_inputs)
for t in range(test_count):
    test_input = testing_inputs[t]
    output_image_blocks = nn.evaluate(test_input)
    output_image = image_form_blocks(output_image_blocks, 256, 256)
    # psnr calculation:
    psnr = psnr_image(test_images[t], output_image)
    # show images:
    show_images(test_images[t], output_image, psnr)


# # :حالت دوم با 36 نرون مخفی

# ## :آموزش شبکه

# In[15]:


nn2 = CompressionNeuralNetwork(64, 36, 64)
epochs = 25
nn2.train(training_inputs, epochs)


# ### :نمودار خطای شبکه در حین آموزش

# In[16]:


# draw error diagram:
plt.plot(np.arange(epochs), nn2.errors)
plt.xlabel('epochs') 
plt.ylabel('error') 
plt.title('Notwork Error Diagram') 
plt.show()


# ## :ارزیابی شبکه

# In[17]:


test_count = len(testing_inputs)
for t in range(test_count):
    test_input = testing_inputs[t]
    output_image_blocks_2 = nn2.evaluate(test_input)
    output_image_2 = image_form_blocks(output_image_blocks_2, 256, 256)
    # psnr calculation:
    psnr = psnr_image(test_images[t], output_image_2)
    # show images:
    show_images(test_images[t], output_image_2, psnr)


# # :نتیجه

# ## استفاده از  تعداد نرون‌های مخفی بیشتر یا نرخ فشرده‌سازی کمتر باعث  بهبود معیار مقایسه و کیفیت ظاهری تصویر خروجی می‌شود
