## 目录

      - 哈里斯角的检测
      - MOPS特征描述
      - 特征匹配
      - 仿射变换/图像扭曲

给定图像1和图像2，先各自进行哈里斯角点的检测，然后对每个角点提取描述符，再使用匹配算法进行匹配，最后通过单应矩阵验证匹配正确率。

#### 哈里斯角的检测

1.判断一个像素是否是哈里斯角，依照窗口滑动的思想，计算其在一定区域$(u,v)$内的灰度变化量$S(u,v)$：

$$S(u, v)=\sum_u \sum_v \omega(x, y)(I(x+u, y+v)-I(x, y))^2$$

接下来对其进行化简，将$I(x+u,y+v)$进行泰勒分解：

$$I(x+u,y+v) \approx I(x, y)+I_x(x, y) \cdot u+I_y(x, y) \cdot v$$

代入原式可得：

$$S(u,v)=\sum_u \sum_v \omega(x, y)\left(I_x(x, y) \cdot u+I_y(x, y) \cdot v\right)^2$$

展开得：

$S(u,v)=\sum \sum \omega(x, y)\left[\left(I_f(x, y) \cdot u\right)^2+2 \cdot I_x(x, y) \cdot u \cdot I_y(x, y) \cdot v+\left(I_y(x, y) \cdot v\right)^2\right]$

化成矩阵形式：

$$S(u, v) \approx[u, v] \sum_u \sum_v \omega(u, v)\left[\begin{array}{cc}I_x(x, y)^2 & I_x(x, y) I_y(x, y) \\ I_x(x, y) I_y(x, y) & I_y(x, y)^2\end{array}\right]\left[\begin{array}{l}u \\ v\end{array}\right]$$

定义海森矩阵$M$为：

$$M=\sum_u \sum_v \omega(u, v)\left[\begin{array}{cc}I_x(x, y)^2 & I_x(x, y) I_y(x, y) \\ I_x(x, y) I_y(x, y) & I_y(x, y)^2\end{array}\right]=\left[\begin{array}{cc}A & C \\ C & B\end{array}\right]$$

其中$\lambda_{max}$和$\lambda_{min}$分别为海森矩阵的特征值。

因此灰度变化量$S(u,v)$可以写为：

$$S(u,v) \approx [u, v]\left[\begin{array}{cc}A & C \\ C & B\end{array}\right]\left[\begin{array}{l}u \\ v\end{array}\right] = Au^2+2Cuv+Bv^2$$

2.只有灰度变化量还不够，定义一个响应函数$R$，更好地判断角点、边缘、平坦区域：

$$R=\operatorname{det} (M)-k( trace(M) )^2$$

其中：

$$\operatorname{det} (M)=\lambda_1 \lambda_2=A B-C^2$$

$$trace(M)=\lambda_1+\lambda_2=A+B$$

常数$k$根据经验选取，范围为$0.02 \sim 0.06$。最终根据$R$的值来判断点的种类。$R$很大则为角点，$R<0$则为边缘，$|R|$很小则为平坦区域。

在实际应用中，某个像素的$R$只需要大于等于$Max(R) * th$即可判定为角点，其中$Max(R)$为所有$R$中的最大值，$th$为常数参数，一般为$0.1$。

函数法：

```py
R_matrix = cv2.cornerHarris(img_src,blockSize= 2,ksize= 3,k= 0.04) 
# Harris角点检测函数，返回响应函数结果矩阵。
# blockSize为窗口大小，ksize为sobel算子的核大小，k为harris算子参数。
```

机理法（手动实现法）：

#### MOPS特征描述

![19642dadc531c9d51d8a7fbc16f12794.png](/_resources/19642dadc531c9d51d8a7fbc16f12794.png)

选取$40*40$（偶数实现时不太好，使用$45*45$）窗口作为补丁，下采样为$8*8$（使用$9*9$）的尺寸，然后平移图像使得窗口中心点为接下来的旋转操作的中心点，然后将其旋转至水平方向。（这样无论带比较的图片旋转了怎样的角度，可以相互匹配的点的特征描述符都是同一方向的）

旋转的方向由海森矩阵$H$的较大的那个特征值（$\lambda_{max}$）对应的特征向量（$\vec{x}_{max}$）决定（也就是通过旋转这个窗口的图像，将中心点变化最快的方向，$\vec{x}_{max}$指向的方向与x轴重合，因此能够有旋转不变性）。

旋转后对整个窗口进行归一化（每个值减去窗口内所有值的平均值并除以标准差）。

最后一次平移只是为了方便取出$8*8$窗口中的点。这就是MOPS算法的描述子。

#### 特征匹配

误差平方和算法（SSD）：

$$D(i, j)=\sum_{s=1}^M \sum_{t=1}^N[S(i+s-1, j+t-1)-T(s, t)]^2$$

待搜索图像为$S$，以$(i,j)$为起始点，目标特征图像为$T$，目标是从图像$S$中找到与$T$最匹配的子图。$M \times N$为子图的大小。实质上，误差平方和就是两个尺寸相同的窗口内的位置两两对应的像素值的差值的平方之和。

#### 仿射变换/图像扭曲

如果已知两幅图像相互之间转换的单应矩阵$H$，那么可以在特征匹配时检查是否匹配到了对应的坐标点上，以此来判断匹配是否成功。

把一个坐标系中的某个坐标点的值填充到另一个坐标系的某个坐标点中，对每个坐标点都这么做，单应矩阵则蕴含了这个坐标系转换过程的信息（也是每次坐标点进行转换的过程的工具）。

如果未知两幅图像相互转换的单应矩阵，则通常可以通过两幅图像各给定的具有对应关系的4个点即可得到单应矩阵（单应矩阵自由度为8）。

图像2：

![623b6e6e53cd19b63ea8488b0c92613f.png](/_resources/623b6e6e53cd19b63ea8488b0c92613f.png)

图像1：

![2f6f8f1e9234afab9d1fe3139feb6e2c.png](/_resources/2f6f8f1e9234afab9d1fe3139feb6e2c.png)

经过单应矩阵$H$将图像2的每个坐标点映射到图像1的坐标系中：

![d88462e5e9eee45d3f566313238697aa.png](/_resources/d88462e5e9eee45d3f566313238697aa.png)

哈里斯角点的检测与匹配（简单窗口和MOPS方法）完整项目代码：

```py
from numpy import *
from math import *
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import auc

def Harris_Corner_Detect_Return_R(img,block_size=2,Sobel_size=3,k=0.04,borderType=cv2.BORDER_DEFAULT):
# 整个函数的返回值为响应函数矩阵R。
# img的数据类型需要为单通道8bit或者浮点数。
# block_size为扫描时候的窗口大小。
# Sobel_size为Sobel算子的大小（需为奇数），定义了角点检测的敏感度，值越大则越容易检测出角点。
# k为响应函数的常数。
	
	R=np.zeros(img.shape,dtype=np.float32)
	# 初始化响应函数矩阵。
	Angle=np.zeros(img.shape,dtype=np.float32)
	# 初始化较大特征向量方向矩阵。
	img_f=img.astype(np.float32)
	# 获得浮点数数据类型的img。
	# scale=1.0/( (Sobel_size-1)*2*block_size*255)
	# Ix=cv2.Sobel(img_f,-1,dx=1,dy=0,ksize=Sobel_size,scale=scale,borderType=borderType)
	# Iy=cv2.Sobel(img_f,-1,dx=0,dy=1,ksize=Sobel_size,scale=scale,borderType=borderType)
	Ix=cv2.Sobel(img_f,-1,dx=1,dy=0,ksize=Sobel_size,borderType=cv2.BORDER_DEFAULT)
	Iy=cv2.Sobel(img_f,-1,dx=0,dy=1,ksize=Sobel_size,borderType=cv2.BORDER_DEFAULT)
	Ixx=Ix**2
	Iyy=Iy**2
	Ixy=Ix*Iy
	Radius=int((block_size-1)/2)
	# 求滑动窗口半径
	N_pre=Radius
	N_post=block_size-N_pre-1
	row_s=col_s=N_pre
	row_e=img_f.shape[0]-N_post
	col_e=img_f.shape[1]-N_post
	# 使用boxFilter函数求和：
	# f_xx = cv2.boxFilter(Ixx,ddepth=-1,ksize=(block_size,block_size) ,anchor =(-1,-1),normalize=False, borderType=borderType)
	# f_yy = cv2.boxFilter(Iyy,ddepth=-1,ksize=(block_size,block_size),anchor =(-1,-1),normalize=False,borderType=borderType)
	# f_xy = cv2.boxFilter(Ixy, ddepth=-1,ksize=(block_size,block_size),anchor =(-1,-1),normalize=False,borderType=borderType)
	for r in range(row_s,row_e):
		for c in range(col_s,col_e):
			# 手动求和：
			sum_xx = Ixx[r-N_pre:r+N_post+1,c-N_pre:c+N_post+1].sum()
			sum_yy = Iyy[r-N_pre:r+N_post+1,c-N_pre:c+N_post+1].sum()
			sum_xy = Ixy[r-N_pre:r+N_post+1,c-N_pre:c+N_post+1].sum()
			# 使用boxFilter函数求和：
			# sum_xx = f_xx[r,c]
			# sum_yy = f_yy[r, c]
			# sum_xy = f_xy[r, c]
            # 经实验，使用boxFilter函数求和与手动求和的结果会有极其细微的差别（检测到的特征点会相差大概几十个）
			result=((sum_xx*sum_yy)-(sum_xy**2)) - k * (sum_xx+sum_yy)**2
            # 得到响应函数的结果。
			M=np.array([[sum_xx,sum_xy],[sum_xy,sum_yy]])
			eig,vec=np.linalg.eig(M)
            # 得到海森矩阵的特征值和特征向量。
			angle=Angle_compute(np.array([0,1]),vec[:,np.argmax(eig)])
			Angle[r,c]=float(-angle)
			R[r,c]=result
	return R,Angle

def Angle_compute(vector_1,vector_2):
    L_1=np.sqrt(vector_1.dot(vector_1))
    L_2=np.sqrt(vector_2.dot(vector_2))
    product=vector_1.dot(vector_2)
    cos_value=product/(L_1*L_2)
    radian=np.arccos(cos_value)
    result=radian*180/np.pi
    cross_result=np.cross(vector_1,vector_2)
    # 叉乘结果为正，则为顺时针的角度，结果为负，则为逆时针的角度。
    # [0,1]为x轴，[1,0]为y轴。
    if (cross_result>=0):
        return float(result)
    if (cross_result<0):
        return float(-result)

def Show_Harris_Points_Return_Number(img,R,threshold=0.01):
    img_show=img.copy()
    if (len(img_show.shape)==2):
        img_show=cv2.cvtColor(img_show,cv2.COLOR_GRAY2BGR)
    R_2=R
    R_2[R<=threshold*R.max()]=0
    img_show[R_2!=0]=(255,0,0)
    print("当前图像检测出的候选哈里斯角点数为：")
    print(len(np.where(R_2!=0)[0]))
    # 输出当前图像检测出的哈里斯角点数量。
    # print(np.max(R_2))
    # 输出矩阵R值中的最大值。
    plt.figure()
    plt.title('Show_Harrus_Points')
    plt.imshow(img_show,cmap=cm.gray)
    # plt.show()

def Get_Harris_Points_Coordinate(R,min_dist=10,threshold=0.01):
# min_dist为分割角点和图像边界的像素数目。
    R_t=(R > R.max() * threshold) * 1
	# 获得代表特征点的布尔矩阵，'*1'将布尔矩阵转化为0/1矩阵（类型转换）。

    coords=array(R_t.nonzero()).T
    # nonzero()可以得到numpy对象的非零元素的索引。
    # 它的返回值是一个元组，元组的每个元素都是一个整数数组，其值为对象的非零元素的索引。
    # .T为转置。
	# 获得代表特征点的坐标。
    
    values=[R[i[0],i[1]] for i in coords]
    # i[0]:h,i[1]:w,coords内的坐标遵循(h,w)。
	# 根据坐标获得代表特征点的R值。

    index=argsort(values)[::-1]
	# 根据value中的R值之间的大小进行排序，并把排序结果（以索引的形式）赋值给index。
    # 加上[::-1]代表从大到小的顺序进行排序，如果去掉则从小到大的顺序进行排序。

    allowed_locations=zeros(R.shape)
    allowed_locations[min_dist:-min_dist,min_dist:-min_dist]=1
	# 视为边缘的像素赋值为0不参与接下来的运算。

    filtered_coords=[]
    # 待输出的符合要求的坐标列表。
    
    for i in index:
    # 在index中，R值越大越靠前，越容易先被遍历。
        if (allowed_locations[coords[i,0],coords[i,1]] == 1):
        # 判定是否边缘的像素。
            filtered_coords.append(coords[i])
            # 将第i个特征点的坐标添加进去。
            allowed_locations[(coords[i,0]-min_dist):(coords[i,0]+min_dist), (coords[i,1]-min_dist):(coords[i,1]+min_dist)] = 0
            # 使第i个特征点周围半径为min_dist的窗口内的像素都不进行特征点判定。
    # 返回坐标遵循(h,w)。
    return filtered_coords

def Get_Descriptors_Windows(img,filtered_coords,wid=2):
# 对于给定的图像，返回filtered_coords中的每个特征点周围半径为wid的窗口内的像素的降维数组作为描述子。
	descr=[]
	for coords in filtered_coords:
		patch=img[coords[0]-wid:coords[0]+wid+1,coords[1]-wid:coords[1]+wid+1].flatten()
        # .flatten()将多维数组降维为一维数组（铺平）。
		descr.append(patch)
        # 将降维后得到的一维数组作为此点的描述子。
	return descr

def Get_Descriptors_MOPS(img,Angle,filtered_coords,wid=20):
    descr=[]
    for coords in filtered_coords:
    # coords遵循(h,w)。
        angle=Angle[coords[0],coords[1]]
        img_r,(H_New,W_New),M=Image_Rotation_Entire_Size(img,angle=angle)
        coord_td=Coordination_After_Rotation([coords[1],coords[0]],M)
        # coord_td遵循(w,h)。
        patch=img_r[coord_td[1]-wid:coord_td[1]+wid+1,coord_td[0]-wid:coord_td[0]+wid+1]
        patch=cv2.resize(patch,(8,8)).flatten()
        norms=np.linalg.norm(patch)
        patch=patch/norms
        # print(patch)
        descr.append(patch)
    return descr

def Image_Rotation_Entire_Size(img,center=None,angle=0,scale=1.0):
# 求仿射变化后原中心点在新图像中的坐标，遵循(w,h)。
# 对图像的索引遵守[h,w]。
    H,W=img.shape[0:2]
    if center==None:
        center=(W//2,H//2)
    rotate_Matrix=cv2.getRotationMatrix2D(center,float(angle),scale=scale)
    # 遵循(w,h)。
    H_New = int(W * fabs(sin(radians(angle))) + H * fabs(cos(radians(angle))))
    # H_New=Wsin(theta)+Hcos(theta)，为新的中心点的坐标。
    W_New = int(H * fabs(sin(radians(angle))) + W * fabs(cos(radians(angle))))
    # W_New=Wcos(theta)+Hsin(theta)，为新的中心点的坐标。
    rotate_Matrix[0,2] += (W_New - W)/2
    rotate_Matrix[1,2] += (H_New - H)/2
    # 平移原来的中心点。
    img_rotate=cv2.warpAffine(img,rotate_Matrix,(W_New,H_New),borderValue=(0,0,0))
    # warpAffine遵循(w,h)。
    return img_rotate,(H_New,W_New),rotate_Matrix

def Match_SSD_Ratio_test(desc1,desc2,threshold_1,threshold_2=1):
# 当threshold_1=1时，就是没有阈值的SSD匹配。
# 当threshold_2=1时，就是没有比率测试的SSD匹配。
    n = len(desc1[0])
    # 描述子窗口的降维前的size。
    d = -ones((len(desc1), len(desc2)))
    # 全-1矩阵。size分别为图像1和图像2的特征点数量。
    # d[i,j]代表图像1的第i个特征点和图像2的第j个特征点的代表相关性的数值。
    for i in range(len(desc1)):
        for j in range(len(desc2)):
            SSD=sum((desc1[i]-desc2[j])**2)
            # 对于某一对特征点而言，SSD越小，可信度越高，则阈值的设置为系数*SSD中的最大值，SSD的最大值越贴近SSD的最小值，则此最小值越不可信。
            # 因此SSD的最小值需要小于系数*SSD中的最大值（使用的是当前的图1的特征点计算得到的最大值和最小值）。
            d[i,j]=SSD
    bool_t=np.ones(len(desc1),dtype=bool)
    # 初始化全为True。
    for i in range(len(desc1)):
        L_i=d[i,:]
        if (min(L_i) <= threshold_1 * max(L_i)):
            bool_t[i]=True
            # 判定为阳性。
        else:
            bool_t[i]=False
            # 判定为阴性。
    index = argsort(d)
    matchscores_1 = index[:, 0]
    matchscores_2 = index[:, 1]
    bool_r=np.ones(len(matchscores_1),dtype=bool)
    for i in range(len(matchscores_1)):
            if ((d[i,matchscores_1[i]]/d[i,matchscores_2[i]])<threshold_2):
                bool_r[i]=True
                # 通过比率测试。
            else:
                bool_r[i]=False
                # 不通过比率测试。
    return matchscores_1,bool_t,bool_r

def Image_Combination(im1, im2):
    row1 = im1.shape[0]
    row2 = im2.shape[0]
    if row1 < row2:
        im1 = concatenate( (im1 , zeros((row2 - row1, im1.shape[1])) ) , axis=0)
    elif row1 > row2:
        im2 = concatenate( (im2 , zeros((row1 - row2, im2.shape[1])) ) , axis=0)
    # 如果图片行数不一致，则将较少的补零到一致。
    # concatenate()函数可以将两张图片直接进行合并到一个窗口中。
    return concatenate((im1, im2), axis=1)

def Plot_Matches_Points(img1, img2, filtered_coords1, filtered_coords2, matchscores ,bool_t,bool_r,figsize):
    plt.figure(figsize=(figsize[0],figsize[1]))
    img3 = Image_Combination(img1, img2)
    plt.imshow(img3)
    col_1 = img1.shape[1]
    # 图像2的点在合并窗口的坐标要加上图像1的列数。
    for i, m in enumerate(matchscores):
    # enumerate(迭代对象)会使得迭代对象的元素变成(索引值，元素值)的形式。
    # 前面的匹配相关的函数返回的数组是以索引表示图像1的序数，元素值表示相匹配的图像2的序数。
        if (bool_t[i]==True) and (bool_r[i]==True):
            plt.plot([filtered_coords1[i][1], filtered_coords2[m][1] + col_1], [filtered_coords1[i][0], filtered_coords2[m][0]], 'r')
            # 图像的两个坐标点连线。
    plt.axis('off')
    plt.show()

def Match_Correct_Rate_Return_FPR_and_TPR(T,img,img_t,filtered_coords1,filtered_coords2,matchscores,bool_t,bool_r=None):
    if bool_r.all()==None:
        bool_r=np.ones(matchscores,dtype=bool)
    TP=0
    FP=0
    bool_c=np.ones(len(matchscores),dtype=bool)
    for i,j in enumerate(matchscores):
        crood=[filtered_coords1[i][1],filtered_coords1[i][0]]
        # 遵循(w,h)。图1第i个特征点的坐标。
        crood_t=[filtered_coords2[j][1],filtered_coords2[j][0]]
        # 遵循(w,h)。图2第j个特征点的坐标。
        crood_td=Coordination_After_Rotation(crood,T)
        # 遵循(w,h)。图1第i个特征点在图2中的实际坐标。
        if ((abs(crood_t[0]-crood_td[0])<15) and (abs(crood_t[1]-crood_td[1])<15)):
            bool_c[i]=True
            # 实际为阳性。
        else:
            bool_c[i]=False
            # 实际为阴性。
        if (bool_t[i]==True) and ((bool_r[i]==True) and (bool_c[i]==True)):
            TP=TP+1
            # 为真阳性。
        if (bool_t[i]==True) and ((bool_r[i]==True) and (bool_c[i]==False)):
            FP=FP+1
            # 为假阳性。
    bool_c=list(bool_c)
    P=bool_c.count(True)
    # 实际为阳性的个数。
    N=len(bool_c)-P
    # 实际为阴性的个数。
    if P!=0:
      TPR=TP/P
    else:
      TPR=0
    if N!=0:
      FPR=FP/N
    else:
      FPR=0
    return [FPR,TPR]

# 一个特征点对另一图像的所有特征点计算得到的最小SSD值通过此阈值则假定为阳性，不通过此阈值则假定为阴性。
# 应用比率测试时，则还需要同时通过比率测试才假定为阳性。
# 通过阈值，则假定它是匹配正确的（阳性），反之则假定它是匹配错误的（阴性）。
# 即使是对另一图像的所有特征点中的最小SSD了，但没通过阈值则也假定为阴性。
# 真阳性：假定为阳性，实际上匹配正确（阳性）。
# 假阳性：假定为阴性，实际上匹配错误（阴性）。
# 通过单应矩阵，可以确认其匹配实际上是否正确（获得实际的阳性和阴性）。
# 真阳性率（TPR）：假定为阳性且实际上为阳性的样本个数与所有实际为阳性的样本个数之比。
# 假阳性概率（FPR）：假定为阳性且实际上为阴性的样本个数与所有实际为阴性的样本个数之比。
# 选取不同的阈值能够获得不同的(FPR,TPR)的坐标点，多个阈值的坐标点连接起来即为ROC曲线。
# 通过ROC曲线可以评估应用不同阈值时的算法表现。
# ROC曲线需要有真实的样本数据才能绘制，因此ROC曲线仅能用作参考，其可信度与样本数据有关。

def Coordination_After_Rotation(crood,rotate_Matrix):
# 图像旋转，返回原图像的某点坐标在旋转后的坐标。
# 因为opencv旋转后不会自动进行平移和窗口自适应，因此本函数仅适用于完整尺寸的旋转。
# 因为rotate_Matrix遵循(w,h)，因此整个函数的结果遵循(w,h)。
    crood = np.float32(crood).reshape([-1, 2])
    crood = np.hstack([crood, np.ones([1,1])]).T
    target_point = np.dot(rotate_Matrix, crood)
    # 图1的坐标经过旋转矩阵左乘后得到在图2的坐标。
    target_point = [int(target_point[0][0]),int(target_point[1][0])]
    # 返回结果遵循(w,h)。
    return target_point

def Get_ROC_Coordnation_t(d1,d2,threshold_1,threshold_2,T,img_1,img_2,filtered_coords1,filtered_coords2):
    d=threshold_1/10
    Coords=[]
    for i in range(10):
        matches,bool_t,bool_r = Match_SSD_Ratio_test(d1,d2,d+(i*d),threshold_2)
        Coords.append(Match_Correct_Rate_Return_FPR_and_TPR(T,img_1,img_2,filtered_coords1,filtered_coords2,matches,bool_t,bool_r))
    print(Coords)
    return Coords

def Get_ROC_Coordnation_r(d1,d2,threshold_1,threshold_2,T,img_1,img_2,filtered_coords1,filtered_coords2):
    d=threshold_1/10
    Coords=[]
    for i in range(10):
        matches,bool_t,bool_r = Match_SSD_Ratio_test(d1,d2,threshold_1,d+(i*d))
        Coords.append(Match_Correct_Rate_Return_FPR_and_TPR(T,img_1,img_2,filtered_coords1,filtered_coords2,matches,bool_t,bool_r))
    print(Coords)
    return Coords

def Draw_ROC_Return_AUC(Croods):
    Croods=np.array(Croods)
    FPR=Croods[:,0]
    TPR=Croods[:,1]
    plt.figure(figsize=(10,10))
    plt.plot([0,1],[0,1],lw=2,linestyle='--')
    plt.plot(FPR,TPR,lw=2,label='ROC Curve (AUC=%0.2f)' % auc(FPR,TPR))
    plt.xlim([0.0,1.0])
    plt.xlabel('False Positive Rate')
    plt.ylim([0.0,1.0])
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc="lower right")
    plt.show()

k=0.04
# 响应函数的常数。

th=0.01
# 响应函数的阈值系数。

wid=2
# 5x5描述子提取窗口半径。

t_1=0.5
t_2=0.03
# SSD匹配的阈值。

r_1=0.9
r_2=0.85
# 比率测试的阈值。

T=np.array(([1.065366,-0.001337,-299.163870],
            [0.027334,1.046342,-11.093753],
            [0.000101,0.000002,1]),dtype=np.float32)
# 图像1到图像2的单应矩阵（仿射变换）。

img_path_1='yosemite1.jpg'
img_path_2='yosemite2.jpg'

img_src_1 = cv2.imread(img_path_1,cv2.IMREAD_GRAYSCALE)
img_src_1_C = cv2.imread(img_path_1,1)
img_src_1_C = cv2.cvtColor(img_src_1_C,cv2.COLOR_BGR2RGB)

img_src_2 = cv2.imread(img_path_2,cv2.IMREAD_GRAYSCALE)
img_src_2_C = cv2.imread(img_path_2,1)
img_src_2_C = cv2.cvtColor(img_src_2_C,cv2.COLOR_BGR2RGB)
# cv2.IMREAD_GRAYSCALE将三通道变为单通道并以灰度值表示图像，也可以直接用参数0。

R_1,A_1=Harris_Corner_Detect_Return_R(img_src_1,block_size=2,Sobel_size=3,k=k)
Show_Harris_Points_Return_Number(img_src_1,R_1,threshold=th)

R_2,A_2=Harris_Corner_Detect_Return_R(img_src_2,block_size=2,Sobel_size=3,k=k)
Show_Harris_Points_Return_Number(img_src_2,R_2,threshold=th)

filtered_coords1 = Get_Harris_Points_Coordinate(R_1, min_dist=10,threshold=th)
filtered_coords2 = Get_Harris_Points_Coordinate(R_2, min_dist=10,threshold=th)

d1_windows = Get_Descriptors_Windows(img_src_1_C,filtered_coords1, wid)
d2_windows = Get_Descriptors_Windows(img_src_2_C,filtered_coords2, wid)

matches,bool_t,bool_r = Match_SSD_Ratio_test(d1_windows,d2_windows,threshold_1=t_1,threshold_2=r_1)
Plot_Matches_Points(img_src_1_C, img_src_2_C,filtered_coords1, filtered_coords2, matches,bool_t,bool_r,(30, 20))
Match_Correct_Rate_Return_FPR_and_TPR(T,img_src_1_C,img_src_2_C,filtered_coords1,filtered_coords2,matches,bool_t,bool_r)

d1_mops = Get_Descriptors_MOPS(img_src_1_C,A_1,filtered_coords1, wid)
d2_mops = Get_Descriptors_MOPS(img_src_2_C,A_2,filtered_coords2, wid)

matches,bool_t,bool_r = Match_SSD_Ratio_test(d1_mops,d2_mops,threshold_1=t_2,threshold_2=r_2)
Plot_Matches_Points(img_src_1_C, img_src_2_C,filtered_coords1, filtered_coords2, matches,bool_t,bool_r,(30, 20))
Match_Correct_Rate_Return_FPR_and_TPR(T,img_src_1_C,img_src_2_C,filtered_coords1,filtered_coords2,matches,bool_t,bool_r)

Crood_1_1=Get_ROC_Coordnation_t(d1_windows,d2_windows,1,1,T,img_src_1_C,img_src_2_C,filtered_coords1,filtered_coords2)
Crood_1_2=Get_ROC_Coordnation_r(d1_windows,d2_windows,1,1,T,img_src_1_C,img_src_2_C,filtered_coords1,filtered_coords2)

Crood_2_1=Get_ROC_Coordnation_t(d1_mops,d2_mops,0.1,1,T,img_src_1_C,img_src_2_C,filtered_coords1,filtered_coords2)
Crood_2_2=Get_ROC_Coordnation_r(d1_mops,d2_mops,1,1,T,img_src_1_C,img_src_2_C,filtered_coords1,filtered_coords2)

Draw_ROC_Return_AUC(Crood_1_1)
Draw_ROC_Return_AUC(Crood_1_2)

Draw_ROC_Return_AUC(Crood_2_1)
Draw_ROC_Return_AUC(Crood_2_2)

# 5x5窗口：SSD测试ROC曲线坐标：
# 参数r=1,t=1.0,(0.9979508196721312, 1.0)
# 参数r=1,t=0.9,(0.9979508196721312, 1.0)
# 参数r=1,t=0.8,(0.9979508196721312, 1.0)
# 参数r=1,t=0.7,(0.9979508196721312, 1.0)
# 参数r=1,t=0.6,(0.9364754098360656, 1.0)
# 参数r=1,t=0.5,(0.13524590163934427, 0.6739130434782609)
# 参数r=1,t=0.4,(0.022540983606557378, 0.30434782608695654)
# 参数r=1,t=0.3,(0.004098360655737705, 0.06521739130434782)
# 参数r=1,t=0.2,(0.0, 0.0)
# 参数r=1,t=0.1,(0.0, 0.0)

# 5x5窗口:比率测试ROC曲线坐标：
# 参数t=1,r=1.0,(0.9979508196721312, 1.0)
# 参数t=1,r=0.9,(0.11065573770491803, 0.5434782608695652)
# 参数t=1,r=0.8,(0.022540983606557378, 0.34782608695652173)
# 参数t=1,r=0.7,(0.010245901639344262, 0.15217391304347827)
# 参数t=1,r=0.6,(0.004098360655737705, 0.10869565217391304)
# 参数t=1,r=0.5,(0.0020491803278688526, 0.0)
# 参数t=1,r=0.4,(0.0, 0.0)
# 参数t=1,r=0.3,(0.0, 0.0)
# 参数t=1,r=0.2,(0.0, 0.0)
# 参数t=1,r=0.1，(0.0, 0.0)

# MOPS：SSD测试ROC曲线坐标：
# 参数r=1,t=0.1,(0.9108910891089109, 1.0)
# 参数r=1,t=0.09,(0.8693069306930693, 1.0)
# 参数r=1,t=0.08,(0.8158415841584158, 1.0)
# 参数r=1,t=0.07,(0.7465346534653465, 1.0)
# 参数r=1,t=0.06,(0.6356435643564357, 0.9655172413793104)
# 参数r=1,t=0.05,(0.5108910891089109, 0.9655172413793104)
# 参数r=1,t=0.04,(0.33663366336633666, 0.9310344827586207)
# 参数r=1,t=0.03,(0.18613861386138614, 0.9310344827586207)
# 参数r=1,t=0.02,(0.07722772277227723, 0.5862068965517241)
# 参数r=1,t=0.01,(0.011881188118811881, 0.27586206896551724)

# MOPS:比率测试ROC曲线坐标：
# 参数t=1,r=1.0,(1.0, 1.0)
# 参数t=1,r=0.9,(0.5683168316831683, 0.9310344827586207)
# 参数t=1,r=0.8,(0.2871287128712871, 0.7241379310344828)
# 参数t=1,r=0.7,(0.14455445544554454, 0.6206896551724138)
# 参数t=1,r=0.6,(0.06930693069306931, 0.5517241379310345)
# 参数t=1,r=0.5,(0.02574257425742574, 0.4482758620689655)
# 参数t=1,r=0.4,(0.011881188118811881, 0.3793103448275862)
# 参数t=1,r=0.3,(0.007920792079207921, 0.2413793103448276)
# 参数t=1,r=0.2,(0.0039603960396039604, 0.20689655172413793)
# 参数t=1,r=0.1,(0.0, 0.10344827586206896)

"""
def Match_Correlation_Cross_Validation(desc1, desc2, threshold=0.8):
# 根据相关性进行匹配，并且交叉验证。
    matches_12 = Match_Correlation(desc1, desc2, threshold)
    matches_21 = Match_Correlation(desc2, desc1, threshold)
    index_12 = where(matches_12 >= 0)[0]
    # where(条件语句)返回对应数组的满足条件语句的所有值的对应索引值的二维数组。
    # 此二维数组长度为2，第一个元素为行索引，第二个元素为列索引。
    # [0]代表行索引，[1]代表列索引。
    # 此处index_12即为所有matches_12中满足值大于0（存在最佳匹配）的元素的索引值的数组，即为图像1的特征点的序数。
    
    for n in index_12:
        if matches_21[ matches_12[n] ] != n:
        # matches_12[n]代表图像1的序数为n的特征点在图像2的特征点的最佳匹配的序数。
        # matches_21[matches_12[n]]代表图像2中的序数为matches_12[n]的特征点在图像1的特征点的最佳匹配的序数。
            matches_12[n] = -1
            # 将没有通过交叉验证的点消去。
    return matches_12

def Match_Correlation(desc1, desc2, threshold=0.8):
# 根据相关性进行匹配。
    n = len(desc1[0])
    # 描述子窗口的降维前的size。
    d = -ones((len(desc1), len(desc2)))
    # 全-1矩阵。size分别为图像1和图像2的特征点数量。
    # d[i,j]代表图像1的第i个特征点和图像2的第j个特征点的代表相关性的数值。
    for i in range(len(desc1)):
        for j in range(len(desc2)):
            d1 = (desc1[i] - mean(desc1[i])) / std(desc1[i])
            d2 = (desc2[j] - mean(desc2[j])) / std(desc2[j])
            # 对每个描述子进行归一化。（每个描述子都是一个多维数组被降维成的一维数组）
            ncc_value = (sum(d1 * d2)) / (n - 1)
            # 计算两个特征点之间代表相关性的一个数值。
            if ncc_value > threshold:
                d[i, j] = ncc_value
            # 大于给定的参数阈值则判定为成功匹配。
    index = argsort(-d)
    # 对参数d取负，也就是返回数组值从大到小排序的索引值。
    # 参数d为二维数组，则argsort()返回一个二维数组，每个元素代表对应的行的排序。
    # 注意当排序二维数组时，argsort()只会对每一行数据单独排序。
    matchscores = index[:, 0]
    # 此处，因为d是一个矩阵，有len(descr1)行，每一行都代表了某个图像1的特征点对图像2中的每个特征点的相关性数据（因为一行有len(desc2)列）。
    # 因此截取排序后的[:,0]的一列其实已经蕴含了图像1中每个特征点对图像2中每个特征点的最佳匹配。
    # matchscores虽然是一维数组，但是其中的元素的索引即为图像1的特征点的序数，元素的值即为图像2的特征点的序数。
    return matchscores
"""
```










