
# CVFX HW2 Report  - Team 16 

### 1. Training MUNIT
### Training 
Start
![](https://i.imgur.com/IFP2hxW.png)

Finish
![](https://i.imgur.com/uzHg6vX.png)



---

### 2. Inference one image in multiple style - MUNIT
### Introduction
本篇 paper 的主要貢獻在於他允許一張圖片輸出多張圖片，並保留原圖的 content，只轉變成不同styles。

因此在這篇中所提出的 model 架構可以分為 content code 和 style code 兩部分。

首先是 content code，因為每種 class 都有特有的共同特徵 (例如:狗和貓都有眼睛、鼻子和嘴巴，而靴子和高跟鞋都有鞋底、鞋跟和鞋頭)，所以本篇於某幾層 convolution layer 中使用 Instance Normalization 來保存這些共同特徵。

而 style code 部分就是單純為了保存 style 的特徵，例如白貓和灰貓在顏色上就有很大差異，而哈士奇跟柯基外表有差異很大，因此本篇依然於某幾層 convolution layer 中使用 Batch Normalization 來保存 style 的特性分布。而不使用 Instance Normalization 是因為此方式會使分布的 Mean 和 Variance 消除掉，所以不用。

### Model
Model 中對圖片的轉換方式如下。
![](https://i.imgur.com/99tdZ9H.png)

- X1 — input image
- X2 — output image
- C — conten code，每張圖片映射到一個content code, 記錄共有的特徵。
- S — style code，X2的圖片產生是基於S2的style code(這部分為random產生)，不同的style code產生不同的圖片。

此為 Model 的大致架構 (圖非原 paper 的圖示，而是原 UNIT 的架構圖示，因為兩者差異不大，只差在前幾層 convolution layer 有使用 Instance Normalization 和 Batch Normalization )
![](https://i.imgur.com/2E798Pp.png)

### Loss function
![](https://i.imgur.com/jDQJNMQ.png)
![](https://i.imgur.com/uFMza8V.png)

我們希望自己的 reconstruction error 要最小，意思是一張圖片 x 進入 encoder 再進去 decoder 出來的結果 y，希望 x 和 y 可以很接近。

![](https://i.imgur.com/DVg2YNh.png)
![](https://i.imgur.com/aTispdH.png)

我們希望自己的 latent reconstruction error 要最小，
- 意思是大眼睛的狗狗丟進model也要生出大眼睛的貓咪希望x和y的content code相近。
- 意思是柯基狗丟進decoder解碼後再編碼還能夠知道這style code是柯基狗
希望 x 和 y 的 style code 相近。

### Inference
|Style<br>\\<br>Content | ![](https://i.imgur.com/ERwTuD8.png) | ![](https://i.imgur.com/o8abqOL.png) | ![](https://i.imgur.com/oGE9XYZ.png) |
|---|---|---|---|
| ![](https://i.imgur.com/DkOJSMu.jpg) | ![](https://i.imgur.com/4AeOXwn.jpg) | ![](https://i.imgur.com/ra3ZZ5G.jpg) | ![](https://i.imgur.com/fAfEorp.jpg) |
| ![](https://i.imgur.com/kzq1CbX.jpg) | ![](https://i.imgur.com/dtV8e55.jpg) | ![](https://i.imgur.com/9AqpMlp.jpg) | ![](https://i.imgur.com/aD503OB.jpg) |
| ![](https://i.imgur.com/0x8ySEM.jpg) | ![](https://i.imgur.com/d2kdfMi.jpg) | ![](https://i.imgur.com/wYhb3ut.jpg) | ![](https://i.imgur.com/3x3jsxi.jpg) 

由於MUNIT將整個轉換流程拆成2個步驟，分別為content與style，在轉換的過程中可以針對不同的部位來進行，所以可以很清楚地根據這種特徵去轉換不同部位的風格，尤其是在最右邊的黑色轉換，可以很明顯看出物件原本的輪廓，又不失想要轉換的風格，甚至一些反光的部分效果也都不錯。相較下來，中間橘色的轉換效果就並不是很好，有一些地方的上色並不是很均勻，可以看出明顯的色差，這讓結果顯得有一些不自然，不過造成這樣的成果也有可能是因為input的style顏色較為複雜，右邊是單純的黑色，但是中間的則有明顯的漸層，鞋子內部的顏色也有明顯的差別，間接可能也就造成了這個結果。


---

### 3. Compare with other method - BicycleGAN
Reference - Toward Multimodal Image-to-Image Translation [[Paper]](https://arxiv.org/abs/1711.11586) [[Code]](https://github.com/junyanz/BicycleGAN)

* Analysis of its structure:
BicycleGAN是一種Supervised way Multimodal image-to-image translation structure，其貢獻在於接著CycleGAN的成功之下，算是很早期挑戰Multimodal的方法，並能有效解決mode collapse problem。

    * 何謂mode collapse: 
        在VAE中，是透過將目標圖片先轉成低維度的Latent vector，再將其轉換回Output photo，於早期的pix2pix structure中，是透過簡單的混入noise於Laten vector中，希望藉由加上隨機的參數，產生多變化的結果，但根據實驗發現，添加的noise易被Generator給忽略，導致長時間訓練過後依然會趨向單一結果。

    BicycleGAN為了解決這個問題，提出了一個結合兩種GAN結構的方式，讓Latent vector & Output形成一個Bijection(即不同的Latent vector必連接著不同的Output)，架構圖如下：<br>
    ![](https://i.imgur.com/PFvVpY7.png)
    * First part - cVAE-GAN (B -> z -> predict B)
        在pix2pix中loss是拿Input & noise合成的結果，與Ground truth做比較，Truth與Latent本身並無直接關聯，會容易出現mode collapse，因此cVAE-GAN這邊為了達成Bijection，直接將Ground truth透過Encoder得到其Latent vector，再與Input經Generator生成結果，保證了B -> z，z為單一，而延伸至Conditional scenario，藉由讓E(B)、即中間產生z結果趨向Gaussian distribution (By加入KL-divergence loss)，保證實際Inference(不知道Ground truth B)時，使用的隨機Latent vector，能代表合理的Style latent。<br>
        ![](https://i.imgur.com/82ilxpR.png)

    * Second part - cLR-GAN (z -> predict B -> predict z)
        與上述相對應，cLR則是先讓Input & noise經Generator產生一預測結果圖，再拿noise與預測結果Encoder得回的Latent vector作比較，保證z -> B的方向也只存在一種結果。另外不必考慮Ground truth和預測結果的loss，因為我們希望能有多樣性、Style不局限於Training dataset，但依舊要考慮discriminator loss，如此才能讓結果盡可能的"real"。<br>
        ![](https://i.imgur.com/rGAoqp3.png)

    
    以此，結合兩種Structure，我們就能同時達成z & B彼此Bijection的特性，KL-divergence的加入使得Latent保證為Style latent，cLR則使圖片"realistic"，並可實踐Multimodal的目標。
    ![](https://i.imgur.com/cZbAID3.png)


* Result: 

|Style<br>\\<br>Content | ![](https://i.imgur.com/ERwTuD8.png) | ![](https://i.imgur.com/o8abqOL.png) | ![](https://i.imgur.com/oGE9XYZ.png) |
|---|---|---|---|
| ![](https://i.imgur.com/DkOJSMu.jpg) | ![](https://i.imgur.com/PbPmBf3.png) | ![](https://i.imgur.com/opwxsbJ.png) | ![](https://i.imgur.com/fxeFr8w.png) |
| ![](https://i.imgur.com/kzq1CbX.jpg) | ![](https://i.imgur.com/RWXeVX5.png) | ![](https://i.imgur.com/618xeYQ.png) | ![](https://i.imgur.com/2g7gAd6.png) |
| ![](https://i.imgur.com/0x8ySEM.jpg) | ![](https://i.imgur.com/XtfnAIB.png) | ![](https://i.imgur.com/9REIm3A.png)| ![](https://i.imgur.com/qqtcFEL.png) |

由圖可知，其實後出的MUNIT，結果並不比BicycleGAN優秀，特別是細節的表現處，MUNIT在填補鞋子的接縫處，都有明顯的缺陷(可能被試作鞋子內部，導致填色上不自然)，這點跟BicycleGAN是Supervised way、Training時就有實際鞋子結構參考應該有很大的關係，畢竟在Edge photo細節不夠下，Unsupervised way預測的一定會與現實有出入，可以看到在細節處比較完善的第三種鞋款，MUNIT的表現就不輸BicycleGAN，加上一般圖片Ground truth難以取得，擁有Unsupervised優勢的MUNIT，其發展性仍比BicycleGAN高。


