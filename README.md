# LMCS-Net: Low-Rank Multi-Axial Cross-Layer Similarity-Guided Network
LMCS-Net is a low-rank multi-axial cross-layer similarity-guided network designed for electric power anomalous object detection, which addresses the problems of missed detection, false detection, and inaccurate localization in anomalous object detection under power scenarios. 

## Overall architecture
<div align="center">
<img src="LMCS-Net/images/total.png" width="88%">
</div>

## Low-Rank Multi-Axial Wavelet Enhancement Mechanism(LMWEM)
<div align="center">
<img src="LMCS-Net/images/LMWEM.png" width="88%">
</div>

## Cross-Layer Similarity-Guided Dynamic Kernel Modulator(CSDKM)
<div align="center">
<img src="LMCS-Net/images/CSDKM.png" width="88%">
</div>

## Quantitative results 🔥
<div align="center" style="font-size:20px; font-weight:bold;">
COMPARISON OF OUR METHOD AGAINST OTHERS ON THE SAOD DATASET.
</div>
<br>
<div align="center">
<img src="LMCS-Net/images/SAOD.png" width="88%">
</div>

<br>

<div align="center" style="font-size:20px; font-weight:bold;">
COMPARISON OF OUR METHOD AGAINST OTHERS ON THE IDID DATASET.
</div>
<br>
<div align="center">
<img src="LMCS-Net/images/IDID.png" width="88%">
</div>

## Qualitative results 🔥
<div align="center">
<img src="LMCS-Net/images/saoddataset.png" width="88%">
</div>
<div align="center" style="font-size:18px; font-weight:bold;">
Results on the self-built SAOD dataset for substation anomalous object detection
</div>

<br>

<div align="center">
<img src="LMCS-Net/images/ididdataset.png" width="88%">
</div>
<div align="center" style="font-size:18px; font-weight:bold;">
Results on the public IDID dataset for transmission line anomalous object detection
</div>


## 🦄 Dependencies
To run the code, make sure you have the following dependencies installed:

| Dependency | Version |
|------------|---------|
| PyTorch    | 1.13.1   |
| Python     | 3.9.19  |
| CUDA       | 11.7    |
| Ubuntu     | 22.04   |

---
## 📦 Install
Install the ultralytics package, including all requirements, in a Python>=3.8 environment with PyTorch>=1.8.

```bash
pip install ultralytics
pip install -r requirements.txt
pip install -e .
```
## 🚀 Train
```bash
python train.py
```
## 🧪 Test
```bash
python test.py
```

[//]: # (## 📁 Resource Download)

[//]: # ()
[//]: # (All resources &#40;public IDID dataset + self-built SAOD dataset&#41; are packaged on Baidu Netdisk. )

[//]: # ()
[//]: # ()
[//]: # ([//]: # &#40;**Baidu Netdisk:**&#41;)
[//]: # ()
[//]: # ([//]: # &#40;- Link: https://pan.baidu.com/s/1rjGXCyEE8Vc4TMF62MxpJA&#41;)
[//]: # ()
[//]: # ([//]: # &#40;- Password: H519 &#41;)
[//]: # ()
[//]: # (## 📜 Citation)

[//]: # (If our work assists your research, feel free to give us a star ⭐ or cite us using:)
