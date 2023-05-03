from tkinter import Tk,StringVar,Label,Button,TOP,BOTH
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from keras.models import load_model
from keras.datasets.mnist import load_data
from matplotlib.pyplot import subplots
from numpy import random,argmax,array,max,zeros,arange,reshape
    
def to_categorical(y, num_classes=None, dtype="float32"):
    y = array(y, dtype="int")
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = max(y) + 1
    n = y.shape[0]
    categorical = zeros((n, num_classes), dtype=dtype)
    categorical[arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = reshape(categorical, output_shape)
    return categorical

class Test():
    def __init__(self):
        self.root = Tk()
        self.root.geometry("400x400")
        self.root.title("测试mnist数据集")
        self.text = StringVar()
        self.text.set("")
        self.label = Label(self.root, textvariable=self.text)
        self.button = Button(self.root, text="下一个数据", command=self.changeText)
        self.button.pack()
        self.label.pack()

        self.fig, self.ax = subplots(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.root)
        self.canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

        self.changeText()
        self.root.mainloop()

    def changeText(self):
        index = random.randint(0, test_images_origin.shape[0])
        data = test_images_origin[index]
        self.ax.cla()
        self.ax.imshow(data, cmap='jet', interpolation='nearest')
        predict = model.predict(test_images[index].reshape(1, 28* 28), verbose=0)
        answer = argmax(predict)
        self.text.set("识别出是：" + str(answer))
        self.canvas.draw()

(train_images_origin, train_labels_origin), (test_images_origin, test_labels_origin) = load_data(".\mnist.npz")

train_images = train_images_origin.reshape((60000, 28*28)).astype('float')
test_images = test_images_origin.reshape((10000, 28*28)).astype('float')
train_labels = to_categorical(train_labels_origin)
test_labels = to_categorical(test_labels_origin)

model = load_model('model.h5')
app = Test()
