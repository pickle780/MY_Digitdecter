from tkinter import Tk,Button,Label,StringVar,Frame
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from numpy import argmax,array,max,zeros,arange,reshape,frombuffer,uint8,mean,float32,rot90,flipud,fliplr
from keras.models import load_model

#模型字典
label_map = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
             10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I',
             19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R',
             28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z', 36: 'a',
             37: 'b', 38: 'c', 39: 'd', 40: 'e', 41: 'f', 42: 'g', 43: 'h', 44: 'i', 45: 'j',
             46: 'k', 47: 'l', 48: 'm', 49: 'n', 50: 'o', 51: 'p', 52: 'q', 53: 'r', 54: 's',
             55: 't', 56: 'u', 57: 'v', 58: 'w', 59: 'x', 60: 'y', 61: 'z'}
# 类别转化矩阵
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

class MyApp:
    def __init__(self, master):
        self.master = master
        self.create_frames()    #创建基本框架
        self.create_areas()     #创建区域
        self.create_headles()   #创建事件处理函数
        # 设置事件处理函数

        self.timer=False
        self.drawing = False
        self.lastx = None
        self.lasty = None

    def create_headles(self):
        self.DrawBoard.mpl_connect('button_press_event', self.on_button_press)
        self.DrawBoard.mpl_connect('motion_notify_event', self.on_motion_notify)
        self.DrawBoard.mpl_connect('button_release_event', self.on_button_release)

    def create_frames(self):
        # 创建画板的框架
        self.DrawBoard_frames = Figure(figsize=(2.8, 2.8), dpi=100)
        self.DrawBoard_frame = self.DrawBoard_frames.add_subplot(111)
        self.DrawBoard_frame.set_xlim([0, 1])
        self.DrawBoard_frame.set_ylim([0, 1])
        self.DrawBoard_frame.axis('off')
        # 创建展示区域的框架
        self.Display_frames = Figure(figsize=(2.8, 2.8), dpi=100)
        self.Display_frame = self.Display_frames.add_subplot(111)
        self.Display_frame.set_xlim([0, 25])
        self.Display_frame.set_ylim([0, 25])
        #创建一个label用于显示结果
        self.text = StringVar()
        self.text.set("")
        self.label = Label(self.master, fg="#00BFFF" , textvariable=self.text)
        self.label.pack()
    def create_areas(self):
        # 创建清空按钮
        self.frame_button=Frame(self.master)
        self.frame_button.pack(expand=True,side="bottom")

        self.clear_button = Button(self.frame_button, text="清空画布",width=10, command=self.clear)
        self.clear_button.pack(side="left",expand=True,padx=10)
        
        # 创建测试按钮
        self.test_button = Button(self.frame_button, text ="测试按钮",width=10,command=self.display)
        self.test_button.pack(side="right",expand=True,padx=10)
        # 创建画板并连接到Matplotlib图形
        self.frame_board=Frame(self.master)
        self.DrawBoard = FigureCanvasTkAgg(self.DrawBoard_frames, master=self.master)
        self.canvas_widget = self.DrawBoard.get_tk_widget()
        self.canvas_widget.pack(side="left",expand=True)
        # 创建画板并连接到Matplotlib图形
        self.display_area = FigureCanvasTkAgg(self.Display_frames, master=self.master)
        self.canvas_widget2 = self.display_area.get_tk_widget()
        self.canvas_widget2.pack(side="right",expand=True)

    def on_button_press(self, event):
        self.drawing = True
        self.lastx = event.xdata
        self.lasty = event.ydata
        
    def on_motion_notify(self, event):
        if self.drawing:
            x = event.xdata
            y = event.ydata
            self.DrawBoard_frame.plot([self.lastx, x], [self.lasty, y], linewidth=15, color='black')
            self.lastx = x
            self.lasty = y
            self.DrawBoard.draw()

    def on_button_release(self, event):
        self.drawing = False

    def clear(self):
        self.DrawBoard_frame.clear()
        self.DrawBoard_frame.set_xlim([0, 1])
        self.DrawBoard_frame.set_ylim([0, 1])
        self.DrawBoard_frame.axis('off')
        self.DrawBoard.draw()

    def display(self):
        # 将一维数组转换为二维数组
        image_np = 255 - (frombuffer(self.DrawBoard.tostring_rgb(), dtype=uint8).reshape(self.DrawBoard.get_width_height()[::] + (3,))[:,:,0])
        # 分块取平均值
        img_sampled = mean(image_np.reshape((28, 10, 28, 10)), axis=(1, 3))
        self.Display_frame.imshow(img_sampled[::-1,:])
        # 显示
        self.display_area.draw()

        img_sampled = (img_sampled.astype(float32)).reshape((28,28)).T
        #训练的时候忘了转置了...
        
        img_sampled = img_sampled.reshape(1,28,28)#CNN需要三维的输入

        #img_sampled = img_sampled.astype(float32).reshape(1,28*28)#MLP需要一维的输入

        img_sampled = img_sampled / 255 
        predict = model.predict(img_sampled, verbose=0)
        answer = argmax(predict)
        self.text.set("预测结果为："+label_map[answer])

model = load_model('.\emnist_model_final.h5')

root = Tk()
root.geometry("600x400")
root.title("画板")

my_canvas = MyApp(root)

root.mainloop()
