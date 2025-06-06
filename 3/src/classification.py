import cv2
import numpy as np
import math
import os
from segmentation import *
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops, hog
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(os.path.dirname(SRC_DIR), 'dateset')
TRAIN_IMGS_DIR = os.path.join(DATASET_DIR, 'train', 'imgs')
TRAIN_MASK_DIR = os.path.join(DATASET_DIR, 'train', 'mask')
TEST_IMGS_DIR = os.path.join(DATASET_DIR, 'test', 'imgs')
TEST_MASK_DIR = os.path.join(DATASET_DIR, 'test', 'mask')


class position:
    '''
    记录一维坐标到二维坐标的映射关系，避免重复计算。
    '''
    def __init__(self, height, width):
        self.t = np.zeros(shape=(height * width), dtype=tuple)
        for y in range(height):
            for x in range(width):
                self.t[y * width + x] = (y, x)


def generate_histogram(img, bin=8, mask=None):
    '''
    提取归一化RGB颜色直方图特征，题目要求每种颜色是8维，故直方图的interval为256/8=32。
    :param img:待处理图像
    :param bin:每种颜色的维度
    :param mask:掩模
    :return:归一化的RGB颜色直方图特征
    '''
    # 若没有mask，则将图像变为(height*width,3)的一列像素
    if mask is None:
        height, width, _ = img.shape
        img = np.reshape(img, newshape=(height * width, 3))
    # 若有mask，则只处理mask覆盖区域，其他区域像素不参与直方图的计算
    else:
        assert img.shape[:2] == mask.shape  # 判断尺寸是否一致
        front_size = len(mask[mask == 255])  # mask大小
        ret = np.zeros(shape=(front_size, 3), dtype=np.uint8)
        height, width = img.shape[:2]
        i = 0
        for h in range(height):
            for w in range(width):
                if mask[h, w] == 255:
                    ret[i] = img[h, w]
                    i += 1
        img = ret
    # 计算直方图的相关信息
    length, channel = img.shape
    assert channel == 3
    interval = 256 / bin  # 直方图间距
    colorspace = np.zeros(shape=(bin, bin, bin), dtype=float)
    # 计算直方图
    for p in range(length):  # 依次处理每个像素点
        v = img[p, :]
        # 计算三个通道归属的区间
        i = math.floor(v[0] / interval)
        j = math.floor(v[1] / interval)
        k = math.floor(v[2] / interval)
        colorspace[i, j, k] += 1
    res = np.reshape(colorspace, newshape=int(math.pow(bin, 3)))  # 转化为8*8*8=512的直方图特征
    res = res / length  # 归一化
    return res


def extract_lbp(img_gray, P=8, R=1):
    '''
    提取灰度图的LBP直方图特征
    '''
    lbp = local_binary_pattern(img_gray, P, R, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
    hist = hist.astype('float')
    hist /= (hist.sum() + 1e-6)
    return hist


def extract_glcm_features(region, distances=[1], angles=[0], levels=256):
    # 若区域全为0或太小，返回全零特征
    if np.count_nonzero(region) == 0 or region.shape[0] < 2 or region.shape[1] < 2:
        return np.zeros(6)
    glcm = graycomatrix(region, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    asm = graycoprops(glcm, 'ASM')[0, 0]
    return np.array([contrast, dissimilarity, homogeneity, energy, correlation, asm])


def extract_hog(region):
    # 若区域太小，返回全零特征（9维）
    if region.shape[0] < 8 or region.shape[1] < 8:
        return np.zeros(9)
    features = hog(region, pixels_per_cell=(8, 8), cells_per_block=(1, 1), visualize=False, feature_vector=True)
    # 若HOG特征长度不为9，补齐或截断
    if len(features) < 9:
        features = np.pad(features, (0, 9 - len(features)), 'constant')
    elif len(features) > 9:
        features = features[:9]
    return features


def extract_shape_features(mask):
    area = np.sum(mask > 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        perimeter = cv2.arcLength(contours[0], True)
    else:
        perimeter = 0
    aspect_ratio = mask.shape[1] / mask.shape[0] if mask.shape[0] > 0 else 0
    return np.array([area, perimeter, aspect_ratio])


def cal_fmat(img, bin, res, t):
    '''
    计算特征矩阵：获取区域颜色直方图和全图颜色直方图，并进行拼接
    :param img: 待处理图像
    :param bin:每种颜色的维度
    :param res:图像分割后的结果
    :param t:一维坐标到二维坐标的映射关系
    :return:各个区域的特征矩阵
    '''
    height, width, _ = img.shape
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 提取归一化全图RGB颜色直方图
    district_histogram = generate_histogram(img, bin)
    m = []
    # 图像分割后的每一个区域
    for comp in res.all_area():  # 遍历图像分割后每一个区域的根节点
        mask = np.zeros(shape=(height, width), dtype=np.uint8)
        v = res.one_area(comp)  # 找到该区域的所有像素点
        # 制作覆盖该区域的mask
        for i in v:
            pix = t.t[i]
            mask[pix[0], pix[1]] = 255
        full_histogram = generate_histogram(img, bin, mask)
        # 新增：提取区域的LBP特征
        region = cv2.bitwise_and(img_gray, img_gray, mask=mask)
        region_uint8 = region.astype(np.uint8)
        lbp_hist = extract_lbp(region_uint8)
        glcm_feat = extract_glcm_features(region_uint8)
        hog_feat = extract_hog(region_uint8)
        shape_feat = extract_shape_features(mask)
        # 拼接颜色直方图、全图直方图和LBP特征
        fvec = np.concatenate((full_histogram, district_histogram, lbp_hist, glcm_feat, hog_feat, shape_feat))
        m.append(fvec)
    m = np.array(m)
    return m


def cal_ytrain(t, comps, img_mask):
    '''
    生成训练集的标签数据。
    :param t:一维坐标到二维坐标的映射关系
    :param comps:图像分割的各区域根节点
    :param img_mask:图像执行题目二后生成的mask
    :return:训练集的标签（记录各个分类后的区域是否是前景）
    '''
    y_train = []
    for comp in comps:
        (y, x) = t.t[comp]
        if img_mask[y, x] == 255:
            y_train.append(1)
        else:
            y_train.append(0)
    return y_train


def data_generate():
    '''
    调用题目二的代码，生成分类模型的训练集和测试集
    :return:有70维特征的训练集与测试集
    '''
    x_train, y_train = [], []
    x_test, y_test = [], []
    # 提取训练集数据的特征
    print("Train Set:")
    for i in range(5, 1001, 5):
        img_number = str(i)
        # 打印进度
        print(f"Processing {str(i)}.png ...")
        # 读取相应编号训练原图和前景图
        img = cv2.imread(os.path.join(TRAIN_IMGS_DIR, img_number + "_origin.png"))
        img_mask = cv2.imread(os.path.join(TRAIN_MASK_DIR, img_number + "_mask.png"))
        if img is None or img_mask is None:
            print(f"Error: {img_number} origin or mask not found!")
            continue
        img_mask = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
        # 调用之前写的图像分割
        res = segment(img)
        t = position(img.shape[0], img.shape[1])
        # 计算特征矩阵
        fmat = cal_fmat(img, 8, res, t)
        for fvec in fmat:
            x_train.append(fvec)
        y_train = y_train + cal_ytrain(t, res.all_area(), img_mask)

    # 提取测试集数据的特征:末尾是学号
    print("Test Set:")
    for i in range(19, 1000, 100):
        print(f"Processing {str(i)}.png ...")
        img_number = str(i)
        # 读取相应编号测试原图和前景图进行相同操作
        img = cv2.imread(os.path.join(TEST_IMGS_DIR, img_number + "_origin.png"))
        img_mask = cv2.imread(os.path.join(TEST_MASK_DIR, img_number + "_mask.png"))
        if img is None or img_mask is None:
            print(f"Error: {img_number} test origin or mask not found!")
            continue
        img_mask = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
        res = segment(img)
        t = position(img.shape[0], img.shape[1])
        fmat = cal_fmat(img, 8, res, t)
        for fvec in fmat:
            x_test.append(fvec)
        y_test = y_test + cal_ytrain(t, res.all_area(), img_mask)

    # 处理训练集
    # PCA降维
    x_train = PCA(n_components=20).fit_transform(x_train)
    # 构建 visual bag of words dictionary
    trainer1 = cv2.BOWKMeansTrainer(50)  # 使用K-Means聚类方法获得50个聚类中心点
    trainer1.add(np.float32(x_train))
    v1 = trainer1.cluster()
    # np.dot(x_train, v1.T))是颜色对比度特征和各个 visual word特征的点积结果
    x_train = np.hstack((x_train, np.dot(x_train, v1.T)))  # 将参数元组的元素数组按水平方向进行叠加，得到70维特征
    x_train, y_train = np.array(x_train), np.array(y_train)

    # 处理测试集数据
    # PCA降维
    x_test = PCA(n_components=20).fit_transform(x_test)
    # 构建 visual bag of words dictionary
    trainer2 = cv2.BOWKMeansTrainer(50)
    trainer2.add(np.float32(x_test))
    v2 = trainer2.cluster()
    x_test = np.hstack((x_test, np.dot(x_test, v2.T)))
    x_test, y_test = np.array(x_test), np.array(y_test)

    return x_train, y_train, x_test, y_test


def test(x_train, y_train, x_test, y_test):
    # SVM
    svm = make_pipeline(StandardScaler(), SVC(gamma='auto', probability=True))
    # 随机森林
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    # 集成学习（软投票）
    clf = VotingClassifier(estimators=[('svm', svm), ('rf', rf)], voting='soft')
    clf.fit(x_train, y_train)
    y_train_predict = clf.predict(x_train)
    y_test_predict = clf.predict(x_test)
    print("训练集的准确率：", accuracy_score(y_train, y_train_predict))
    print("测试集的准确率：", accuracy_score(y_test, y_test_predict))


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = data_generate()
    test(x_train, y_train, x_test, y_test)
