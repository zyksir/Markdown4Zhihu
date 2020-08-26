本文试图探究的问题：

1. 什么是RNN？
2. 为什么RNN会出现梯度爆炸/梯度消失问题？
3. 什么是LSTM？
4. LSTM是如何保证梯度可以正常传递的？



本文是笔者学习过程中对博客的笔记，原作者在评论里说欢迎翻译为日文，我就试着翻译为中文吧hhh。但是感觉博客以及我提出了自己的看法。参考的几个博客在文末给出链接

#### RNN

##### 什么是RNN

![img](https://raw.githubusercontent.com/zyksir/Markdown4Zhihu/master/Data/LSTM是如何解决梯度消失问题的/RNN.png)

对于给定的输入序列 <img src="https://www.zhihu.com/equation?tex=<x_1, x_2, ...., x_k>" alt="<x_1, x_2, ...., x_k>" class="ee_img tr_noresize" eeimg="1">  , 网络将其编码成 <img src="https://www.zhihu.com/equation?tex=<h_1, ...., h_k>" alt="<h_1, ...., h_k>" class="ee_img tr_noresize" eeimg="1"> ，其中 <img src="https://www.zhihu.com/equation?tex=h_t" alt="h_t" class="ee_img tr_noresize" eeimg="1"> 代表 <img src="https://www.zhihu.com/equation?tex=x_t" alt="x_t" class="ee_img tr_noresize" eeimg="1"> 以及其之前信息的编码。整个网络有两个参数矩阵 <img src="https://www.zhihu.com/equation?tex=W_{rec}， W_{in}" alt="W_{rec}， W_{in}" class="ee_img tr_noresize" eeimg="1"> ,  <img src="https://www.zhihu.com/equation?tex=h_t" alt="h_t" class="ee_img tr_noresize" eeimg="1"> 的计算公式如下：

<img src="https://www.zhihu.com/equation?tex=h_{t}=\sigma\left(W_{r e c} \cdot h_{t-1}+W_{i n} \cdot x_{t}\right)
" alt="h_{t}=\sigma\left(W_{r e c} \cdot h_{t-1}+W_{i n} \cdot x_{t}\right)
" class="ee_img tr_noresize" eeimg="1">
整个网络如上图所示， <img src="https://www.zhihu.com/equation?tex=c_t" alt="c_t" class="ee_img tr_noresize" eeimg="1"> 就是 <img src="https://www.zhihu.com/equation?tex=h_t" alt="h_t" class="ee_img tr_noresize" eeimg="1"> ,  <img src="https://www.zhihu.com/equation?tex=c_0=h_0" alt="c_0=h_0" class="ee_img tr_noresize" eeimg="1"> 是隐层的初始化状态，一般是零向量。

##### 为什么RNN会出现梯度消失/爆炸问题

记 <img src="https://www.zhihu.com/equation?tex=W = [W_{rec}, W_{in}]" alt="W = [W_{rec}, W_{in}]" class="ee_img tr_noresize" eeimg="1"> ； <img src="https://www.zhihu.com/equation?tex=L_t" alt="L_t" class="ee_img tr_noresize" eeimg="1"> 是编码 <img src="https://www.zhihu.com/equation?tex=h_t" alt="h_t" class="ee_img tr_noresize" eeimg="1"> 对应的损失，是关于 <img src="https://www.zhihu.com/equation?tex=W" alt="W" class="ee_img tr_noresize" eeimg="1"> 的函数； <img src="https://www.zhihu.com/equation?tex=L" alt="L" class="ee_img tr_noresize" eeimg="1"> 为总损失；则：

<img src="https://www.zhihu.com/equation?tex=\frac{\partial L}{\partial W}=\sum_{t=1}^{T} \frac{\partial L_{t}}{\partial W}
" alt="\frac{\partial L}{\partial W}=\sum_{t=1}^{T} \frac{\partial L_{t}}{\partial W}
" class="ee_img tr_noresize" eeimg="1">
计算完之后我们可以通过 <img src="https://www.zhihu.com/equation?tex=W \leftarrow W-\alpha \frac{\partial E}{\partial W}" alt="W \leftarrow W-\alpha \frac{\partial E}{\partial W}" class="ee_img tr_noresize" eeimg="1"> 来更新参数。

其中 <img src="https://www.zhihu.com/equation?tex=\frac{\partial L_{t}}{\partial W}" alt="\frac{\partial L_{t}}{\partial W}" class="ee_img tr_noresize" eeimg="1"> 又可以写成如下形式：

<img src="https://www.zhihu.com/equation?tex=\frac{\partial L_{k}}{\partial W}=\frac{\partial L_{k}}{\partial h_{k}} \frac{\partial h_{k}}{\partial h_{k-1}} \cdots \frac{\partial h_{2}}{\partial h_{1}} \frac{\partial h_{1}}{\partial W} = \frac{\partial L_{k}}{\partial h_{k}} \left(\prod_{t=2}^{k} \frac{\partial h_{t}}{\partial h_{t-1}}\right) \frac{\partial h_{1}}{\partial W}
" alt="\frac{\partial L_{k}}{\partial W}=\frac{\partial L_{k}}{\partial h_{k}} \frac{\partial h_{k}}{\partial h_{k-1}} \cdots \frac{\partial h_{2}}{\partial h_{1}} \frac{\partial h_{1}}{\partial W} = \frac{\partial L_{k}}{\partial h_{k}} \left(\prod_{t=2}^{k} \frac{\partial h_{t}}{\partial h_{t-1}}\right) \frac{\partial h_{1}}{\partial W}
" class="ee_img tr_noresize" eeimg="1">
将 <img src="https://www.zhihu.com/equation?tex=h_t" alt="h_t" class="ee_img tr_noresize" eeimg="1"> 的计算公式带入，得到 <img src="https://www.zhihu.com/equation?tex=\frac{\partial h_{t}}{\partial h_{t-1}}=\sigma^{\prime}\left(W_{r e c} \cdot h_{t-1}+W_{i n} \cdot x_{t}\right) \cdot W_{r e c}" alt="\frac{\partial h_{t}}{\partial h_{t-1}}=\sigma^{\prime}\left(W_{r e c} \cdot h_{t-1}+W_{i n} \cdot x_{t}\right) \cdot W_{r e c}" class="ee_img tr_noresize" eeimg="1"> ，那么 <img src="https://www.zhihu.com/equation?tex=\frac{\partial L_{t}}{\partial W}" alt="\frac{\partial L_{t}}{\partial W}" class="ee_img tr_noresize" eeimg="1"> 又可以写成：

<img src="https://www.zhihu.com/equation?tex=\frac{\partial L_{k}}{\partial W}=\frac{\partial L_{k}}{\partial h_{k}} \left(\prod_{t=2}^{k} \sigma^{\prime}\left(W_{r e c} \cdot h_{t-1}+W_{i n} \cdot x_{t}\right) \cdot W_{r e c}\right) \frac{\partial h_{1}}{\partial W}
" alt="\frac{\partial L_{k}}{\partial W}=\frac{\partial L_{k}}{\partial h_{k}} \left(\prod_{t=2}^{k} \sigma^{\prime}\left(W_{r e c} \cdot h_{t-1}+W_{i n} \cdot x_{t}\right) \cdot W_{r e c}\right) \frac{\partial h_{1}}{\partial W}
" class="ee_img tr_noresize" eeimg="1">
但是一般激活函数 <img src="https://www.zhihu.com/equation?tex=\sigma(x) = \tanh (x)=\frac{e^{x}-e^{-x}}{e^{x}+e^{-x}}" alt="\sigma(x) = \tanh (x)=\frac{e^{x}-e^{-x}}{e^{x}+e^{-x}}" class="ee_img tr_noresize" eeimg="1"> , 其导数 <img src="https://www.zhihu.com/equation?tex=\sigma^{\prime}(x) = 1 - (\sigma(x))^2" alt="\sigma^{\prime}(x) = 1 - (\sigma(x))^2" class="ee_img tr_noresize" eeimg="1"> ，是一个在区间 <img src="https://www.zhihu.com/equation?tex=[0, 1)" alt="[0, 1)" class="ee_img tr_noresize" eeimg="1"> 之间的数，这个连乘就会导致梯度消失；如果激活函数是sigmoid问题会更大，sigmoid导数绝对值的上界是0.25，一下子就没了；而 <img src="https://www.zhihu.com/equation?tex=W_{rec}" alt="W_{rec}" class="ee_img tr_noresize" eeimg="1"> 要是特征值比较大，连乘还会导致梯度爆炸

 

#### LSTM

##### 什么是LSTM 

![img](https://raw.githubusercontent.com/zyksir/Markdown4Zhihu/master/Data/LSTM是如何解决梯度消失问题的/LSTM.png)

<img src="https://www.zhihu.com/equation?tex=\begin{aligned}
\text{forget gate} \quad f_{t} &=\sigma\left(W_{f} \cdot\left[h_{t-1}, x_{t}\right]+b_{f}\right) \\
\text{input gate} \quad i_{t} &=\sigma\left(W_{i} \cdot\left[h_{t-1}, x_{t}\right]+b_{i}\right) \\
\text{tmp cell state} \quad \tilde{c}_{t} &=\tanh \left(W_{C} \cdot\left[h_{t-1}, x_{t}\right]+b_{C}\right) \\
\text{cell state} \quad c_{t} &=f_{t} * c_{t-1}+i_{t} * \tilde{c}_{t} \\
\text{output gate} \quad o_{t} &=\sigma\left(W_{o}\left[h_{t-1}, x_{t}\right]+b_{o}\right) \\
\text{hidden state} \quad h_{t} &=o_{t} * \tanh \left(c_{t}\right)
\end{aligned}
" alt="\begin{aligned}
\text{forget gate} \quad f_{t} &=\sigma\left(W_{f} \cdot\left[h_{t-1}, x_{t}\right]+b_{f}\right) \\
\text{input gate} \quad i_{t} &=\sigma\left(W_{i} \cdot\left[h_{t-1}, x_{t}\right]+b_{i}\right) \\
\text{tmp cell state} \quad \tilde{c}_{t} &=\tanh \left(W_{C} \cdot\left[h_{t-1}, x_{t}\right]+b_{C}\right) \\
\text{cell state} \quad c_{t} &=f_{t} * c_{t-1}+i_{t} * \tilde{c}_{t} \\
\text{output gate} \quad o_{t} &=\sigma\left(W_{o}\left[h_{t-1}, x_{t}\right]+b_{o}\right) \\
\text{hidden state} \quad h_{t} &=o_{t} * \tanh \left(c_{t}\right)
\end{aligned}
" class="ee_img tr_noresize" eeimg="1">
这里引入了三个门(forget gate、input gate、output gate)，每个门其实就是[0, 1]之间的一个数，可以调整比例；另外引入 <img src="https://www.zhihu.com/equation?tex=W_C" alt="W_C" class="ee_img tr_noresize" eeimg="1"> 来做对新信息的提取

- forget gate调整之前信息所占的比例
- input gate调整新信息所占的比例
- output gate做了一个过滤，将当前状态中的信息有选择的进行输出。

而 <img src="https://www.zhihu.com/equation?tex=c_t" alt="c_t" class="ee_img tr_noresize" eeimg="1"> 所编码是 <img src="https://www.zhihu.com/equation?tex=<x_1, ..., x_t>" alt="<x_1, ..., x_t>" class="ee_img tr_noresize" eeimg="1"> 这个序列所包含的特征，例如序列之间的依赖关系等(这里我自己按照半个上下文特征理解了，也就是只有上文特征；目前也就是个黑匣子，仁者见仁智者见智吧)



值得注意的是，LSTM其实有很多变种，下面介绍几个常见变种

-  [Gers & Schmidhuber (2000)](https://ieeexplore.ieee.org/document/861302) 在计算gate的时候还考虑了old cell state，具体可以看下图：

![img](https://raw.githubusercontent.com/zyksir/Markdown4Zhihu/master/Data/LSTM是如何解决梯度消失问题的/LSTM3-var-peepholes.png)

- GRU，最为常见的变种，将forget gate和input gate合并为一个update gate  <img src="https://www.zhihu.com/equation?tex=z_t" alt="z_t" class="ee_img tr_noresize" eeimg="1"> ，并且在计算tmp cell state的时候对来自过去的信息增加reset gate  <img src="https://www.zhihu.com/equation?tex=r_t" alt="r_t" class="ee_img tr_noresize" eeimg="1"> 做一个过滤
  ![img](https://raw.githubusercontent.com/zyksir/Markdown4Zhihu/master/Data/LSTM是如何解决梯度消失问题的/LSTM3-var-GRU.png)



##### LSTM是如何保证梯度可以正常传递的？

和上面分析类似，我们可以得到：

<img src="https://www.zhihu.com/equation?tex=\frac{\partial E_{k}}{\partial W} =\frac{\partial E_{k}}{\partial h_{k}} \frac{\partial h_{k}}{\partial c_{k}} \cdots \frac{\partial c_{2}}{\partial c_{1}} \frac{\partial c_{1}}{\partial W} =\frac{\partial E_{k}}{\partial h_{k}} \frac{\partial h_{k}}{\partial c_{k}}\left(\prod_{t=2}^{k} \frac{\partial c_{t}}{\partial c_{t-1}}\right) \frac{\partial c_{1}}{\partial W}
" alt="\frac{\partial E_{k}}{\partial W} =\frac{\partial E_{k}}{\partial h_{k}} \frac{\partial h_{k}}{\partial c_{k}} \cdots \frac{\partial c_{2}}{\partial c_{1}} \frac{\partial c_{1}}{\partial W} =\frac{\partial E_{k}}{\partial h_{k}} \frac{\partial h_{k}}{\partial c_{k}}\left(\prod_{t=2}^{k} \frac{\partial c_{t}}{\partial c_{t-1}}\right) \frac{\partial c_{1}}{\partial W}
" class="ee_img tr_noresize" eeimg="1">
在LSTM中，我们展开可以得到：

<img src="https://www.zhihu.com/equation?tex=c_{t}=c_{t-1} *  \sigma\left(W_{f} \cdot\left[h_{t-1}, x_{t}\right]\right) + \tanh \left(W_{c} \cdot\left[h_{t-1}, x_{t}\right]\right) * \sigma\left(W_{i} \cdot\left[h_{t-1}, x_{t}\right]\right)
" alt="c_{t}=c_{t-1} *  \sigma\left(W_{f} \cdot\left[h_{t-1}, x_{t}\right]\right) + \tanh \left(W_{c} \cdot\left[h_{t-1}, x_{t}\right]\right) * \sigma\left(W_{i} \cdot\left[h_{t-1}, x_{t}\right]\right)
" class="ee_img tr_noresize" eeimg="1">
求导得到：

<img src="https://www.zhihu.com/equation?tex=\begin{aligned}
\frac{\partial c_{t}}{\partial c_{t-1}} =&\frac{\partial f_{t}}{\partial c_{t-1}} \cdot c_{t-1}+\frac{\partial c_{t-1}}{\partial c_{t-1}} \cdot f_{t}+\frac{\partial i_{t}}{\partial c_{t-1}} \cdot \tilde{c}_{t}+\frac{\partial \tilde{c}_{t}}{\partial c_{t-1}} \cdot i_{t} \\
=&\sigma^{\prime}\left(W_{f} \cdot\left[h_{t-1}, x_{t}\right]\right) \cdot W_{f} \cdot o_{t-1} \otimes \tanh ^{\prime}\left(c_{t-1}\right) \cdot c_{t-1} \\
&+f_{t} \\
&+\sigma^{\prime}\left(W_{i} \cdot\left[h_{t-1}, x_{t}\right]\right) \cdot W_{i} \cdot o_{t-1} \otimes \tanh ^{\prime}\left(c_{t-1}\right) \cdot \tilde{c}_{t} \\
&+\sigma^{\prime}\left(W_{c} \cdot\left[h_{t-1}, x_{t}\right]\right) \cdot W_{c} \cdot o_{t-1} \otimes \tanh ^{\prime}\left(c_{t-1}\right) \cdot i_{t}
\end{aligned}
" alt="\begin{aligned}
\frac{\partial c_{t}}{\partial c_{t-1}} =&\frac{\partial f_{t}}{\partial c_{t-1}} \cdot c_{t-1}+\frac{\partial c_{t-1}}{\partial c_{t-1}} \cdot f_{t}+\frac{\partial i_{t}}{\partial c_{t-1}} \cdot \tilde{c}_{t}+\frac{\partial \tilde{c}_{t}}{\partial c_{t-1}} \cdot i_{t} \\
=&\sigma^{\prime}\left(W_{f} \cdot\left[h_{t-1}, x_{t}\right]\right) \cdot W_{f} \cdot o_{t-1} \otimes \tanh ^{\prime}\left(c_{t-1}\right) \cdot c_{t-1} \\
&+f_{t} \\
&+\sigma^{\prime}\left(W_{i} \cdot\left[h_{t-1}, x_{t}\right]\right) \cdot W_{i} \cdot o_{t-1} \otimes \tanh ^{\prime}\left(c_{t-1}\right) \cdot \tilde{c}_{t} \\
&+\sigma^{\prime}\left(W_{c} \cdot\left[h_{t-1}, x_{t}\right]\right) \cdot W_{c} \cdot o_{t-1} \otimes \tanh ^{\prime}\left(c_{t-1}\right) \cdot i_{t}
\end{aligned}
" class="ee_img tr_noresize" eeimg="1">
这里一方面有一个forget gate在导数里，另外剩下的几项是相加关系，那么 <img src="https://www.zhihu.com/equation?tex=\frac{\partial E_{k}}{\partial W}" alt="\frac{\partial E_{k}}{\partial W}" class="ee_img tr_noresize" eeimg="1"> 就不太会趋向于0了。另外的博客里我找到这么一个直观，由于 <img src="https://www.zhihu.com/equation?tex=f_t" alt="f_t" class="ee_img tr_noresize" eeimg="1"> 控制着之前的信息有多少会被记住，因此这个数字应该是趋向于1的。事实上，LSTM并不能算是完全解决了这个问题，但是它很好地缓解了这个问题。

我这里对第一个参考blog的解释提出质疑，他认为只要 <img src="https://www.zhihu.com/equation?tex=\sum_{t=1}^{T} \frac{\partial E_{t}}{\partial W} \rightarrow 0" alt="\sum_{t=1}^{T} \frac{\partial E_{t}}{\partial W} \rightarrow 0" class="ee_img tr_noresize" eeimg="1">  不成立就可了，但是这个感觉不太对，哪怕是RNN，t比较小时候对W的梯度也不会增加；但是RNN的问题在于，对于比较大的t,  <img src="https://www.zhihu.com/equation?tex=\frac{\partial E_{k}}{\partial W}" alt="\frac{\partial E_{k}}{\partial W}" class="ee_img tr_noresize" eeimg="1"> 必然会趋向于0，也就说RNN无法利用一个很长的序列后半部分的信息，这个是它梯度消失带来的问题；



另外LSTM除了可以限制梯度必须小于某个阈值，如果大于这个阈值，就取这个阈值进行更新。可以限制梯度必须小于某个阈值，如果大于这个阈值，就取这个阈值进行更新！(这一小段话引用自知乎用户[Bill](https://www.zhihu.com/people/xiao-nu-43)的回答)



##### 关于梯度爆炸的补充

解决梯度爆炸还有一个小技巧，可以限制梯度必须小于某个阈值，如果大于这个阈值，就取这个阈值进行更新！



参考链接：

[Nir Arbel: How LSTM networks solve the problem of vanishing gradients](https://medium.com/datadriveninvestor/how-do-lstm-networks-solve-the-problem-of-vanishing-gradients-a6784971a577)

