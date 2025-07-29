下面分三部分說明：

1. **核心數學模型與推導**（含封閉式更新式證明）
2. **理論保證：收斂率與樣本複雜度**

---

## 1. 核心數學模型與推導

### 1.1 問題設定

* 給定三階張量 $\mathcal{Y}\in\mathbb{R}^{I\times J\times K}$，觀測子集 $\Omega\subset[I]\times[J]\times[K]$。
* **低秩 CP 分解**

  $$
  \mathcal{X}=\sum_{r=1}^{R} a_{\!r}\circ b_{\!r}\circ c_{\!r},\qquad 
  a_{\!r}\!\in\!\mathbb{R}^{I},\;b_{\!r}\!\in\!\mathbb{R}^{J},\;c_{\!r}\!\in\!\mathbb{R}^{K}.
  $$
* 高斯觀測模型：
  $\displaystyle y_{ijk}\sim\mathcal{N}\!\bigl(x_{ijk},\sigma^{2}\bigr),\;(i,j,k)\in\Omega$.

### 1.2 Kronecker 結構先驗

向量化後 $\theta=\text{vec}(A,B,C)$；置

$$
p(\theta)=\mathcal{N}\bigl(0,\;\Sigma_3\!\otimes\!\Sigma_2\!\otimes\!\Sigma_1\bigr),
$$

其中 $\Sigma_{1:3}$ 僅與三個模態維度有關，大小分別為
$I\times I,\;J\times J,\;K\times K$。Kronecker 性質保證
$\log\det(\Sigma_3\otimes\Sigma_2\otimes\Sigma_1)=
 \sum_{\ell=1}^{3}\log\det\Sigma_\ell$，計算複雜度線性於維度R，適合 CPU。([arXiv][1])

### 1.3 變分目標（帶熵正則）

$$
\textstyle 
\mathcal{L}(q)=\mathbb{E}_{q}\!\bigl[\log p(\mathcal{Y}\_{ \Omega}\mid\theta)\bigr]
           -\operatorname{KL}\!\bigl(q(\theta)\,\|\,p(\theta)\bigr)
           +\frac{\lambda}{\eta}\,D_{\!h}\!\bigl(\theta,\theta_{t}\bigr),
$$

其中 $D_h$ 為負熵 Bregman 距離
$h(\theta)=\sum_{i}\theta_i\log\theta_i$。最後一項即 **熵鏡射正則**。

### 1.4 **熵鏡射下降—封閉式更新**

對每個因子向量 $u\in\{a_{\!r},b_{\!r},c_{\!r}\}$：

1. **梯度計算**
   $\nabla_u f=
     \sigma^{-2}\,M_u^\top\!\bigl(M_u u - y_u\bigr)
     +\Sigma_u^{-1}u$，
   其中 $M_u$ 為展開後的設計矩陣；只需
   Hadamard 與矩陣–向量乘，複雜度 $O(|\Omega|R)$。

2. **鏡射步**（負熵 DGF）：

   $$
   u_{t+1}=u_t\odot
   \exp\!\bigl(-\eta\nabla_u f\bigr),\qquad
   u_{t+1}\leftarrow \frac{u_{t+1}}{\|u_{t+1}\|_2}.
   $$

   由 Bregman 投影的一階最適性條件可直接得到上式；證明見附錄 A。

> **定理 1（封閉式更新正確性）**
> 鏡射步所得 $u_{t+1}$ 為 $\mathcal{L}(q)$ 在 step‑size $\eta$ 下的唯一極小點，且每步運算僅含指數與歸一化，故時間複雜度與記憶體皆為 $O(|\Omega|R)$。

*證明提要*：將變分問題寫成
$\min_{u\succ0}\;g(u)+\lambda D_h(u\|u_t)$。
令 $h^*$ 為 Fenchel 共軛，鏡射下降條件
$u_{t+1}=\nabla h^{*}\bigl(\nabla h(u_t)-\eta\nabla g(u_t)\bigr)$。
對負熵 $h$ 有 $\nabla h(u)=\log u$，$\nabla h^{*}(v)=\exp v$；
代回即得上式。詳見 Azizian et al. 的鏡射一階條件推導。

---

## 2. 理論保證

### 2.1 **收斂率**

考慮光滑且 $\alpha$-強凸之對偶目標，應用 Optimistic MD 的 last‑iterate 分析，可得

$$
F(u_{T})-F(u^{*})\le
\frac{L\,\|u^{*}-u_{0}\|_{1}^{2}}{2T},
$$

即 **$O(1/T)$** 收斂；若使用批梯度，常數 $L$ 為 Lipschitz 模數。證明直接套用 OMD 在負熵幾何下的定理 4.1。

### 2.2 **樣本複雜度與可識別性**

* **確定性條件**：Ashraphijuo & Wang 利用 CP 流形的代數獨立多項式，給出唯一可補完的取樣樣式條件。
* **隨機取樣上界**：若 $|\Omega|\ge n\log n+d\,n\log\log n$（$n=\max\{I,J,K\}$），則秩 $d$ 張量在高機率下可被唯一恢復。([arXiv][2])

> **定理 2（EMD‑VI 一致性）**
> 當觀測率滿足上式下限，且 $\eta_t$ 選 $\Theta(1/L)$ 時，
> EMD‑VI 的極限點 $\hat{\mathcal{X}}$ 與真張量 $\mathcal{X}^{*}$ 之重構誤差
> $\|\hat{\mathcal{X}}-\mathcal{X}^{*}\|_F/\|\mathcal{X}^{*}\|_F
> \xrightarrow{p}0$。

*證明提要*：由樣本複雜度確保唯一 CP‑rank‑$R$ 補完，再結合定理 1 的收斂率與 Lipschitz 連續性，即得結果。

---

[1]: https://arxiv.org/html/2506.13613v1 "Variational Inference with Mixtures of Isotropic Gaussians"
[2]: https://arxiv.org/abs/2408.03504 "[2408.03504] Sample Complexity of Low-rank Tensor Recovery from Uniformly Random Entries"
