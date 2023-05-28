digit_decter: <br>
====
准备好requirements.txt的环境后 <br>
在anaconda promote下，进入本文件夹 <br>
输入 <br>
> pyinstaller test_mnist.spec --upx-dir=".\upx-4.0.2-win64" <br>
其中 <br>
> ".\upx-4.0.2-win64" <br>
这个是upx的路径 <br>
这里有一个细节，是在test_mnist.spec这个文件中有这么一行 <br>
> upx_exclude=['_pywrap_tensorflow_internal.pyd']	<br>
这个upx是不能压缩此pyd的，程序会发生问题，这个pyd来自于tensorflow安装产生的文件 <br>
其他的细节：<br>
> upx_args = ['--best']			#采用最大压缩，节约最多空间
> console=False				#因为是gui程序，所以不需要控制台
> sys.setrecursionlimit(sys.getrecursionlimit() * 5)
> block_cipher = None			#这个是我自己加的，因为pyinstaller最大递归上限不够，所以得自己括，不然程序做不出来
