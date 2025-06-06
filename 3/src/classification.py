import cv2
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
from skimage.feature import local_binary_pattern
from skimage.feature import graycomatrix
from skimage.feature import graycoprops

def extract_features(image):
    """提取图像特征"""
    # 颜色特征
    color_features = []
    for channel in cv2.split(image):
        color_features.extend([
            np.mean(channel),
            np.std(channel),
            np.median(channel),
            np.percentile(channel, 25),
            np.percentile(channel, 75)
        ])
    
    # 纹理特征
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, 8, 1, method='uniform')
    lbp_hist, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)
    
    # GLCM特征
    glcm = graycomatrix(gray, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], 256, symmetric=True, normed=True)
    glcm_features = [
        graycoprops(glcm, 'contrast').ravel(),
        graycoprops(glcm, 'dissimilarity').ravel(),
        graycoprops(glcm, 'homogeneity').ravel(),
        graycoprops(glcm, 'energy').ravel(),
        graycoprops(glcm, 'correlation').ravel()
    ]
    
    # 形状特征
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        shape_features = [
            cv2.contourArea(contour),
            cv2.arcLength(contour, True),
            cv2.contourArea(contour) / (cv2.arcLength(contour, True) ** 2)
        ]
    else:
        shape_features = [0, 0, 0]
    
    # 组合所有特征
    features = np.concatenate([
        color_features,
        lbp_hist,
        np.array(glcm_features).ravel(),
        shape_features
    ])
    
    return features

def train_model(X_train, y_train):
    """训练模型"""
    # 特征选择
    selector = SelectKBest(f_classif, k=50)
    X_train_selected = selector.fit_transform(X_train, y_train)
    
    # RFE特征选择
    svc = SVC(kernel='rbf')
    rfe = RFE(estimator=svc, n_features_to_select=10, step=1)
    X_train_rfe = rfe.fit_transform(X_train_selected, y_train)
    
    # 模型参数优化
    param_grid = {
        'svc__C': [0.1, 1, 10, 100],
        'svc__gamma': [0.01, 0.1, 1, 10],
        'svc__kernel': ['rbf', 'poly', 'sigmoid']
    }
    
    # 使用交叉验证进行参数优化
    svc = SVC(probability=True)
    pipe = Pipeline([('svc', svc)])
    grid_search = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_rfe, y_train)
    
    # 获取最佳模型
    best_model = grid_search.best_estimator_
    
    return best_model, selector, rfe

def main():
    # 加载数据
    X, y = load_data()
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 训练模型
    model, selector, rfe = train_model(X_train, y_train)
    
    # 评估模型
    X_test_selected = selector.transform(X_test)
    X_test_rfe = rfe.transform(X_test_selected)
    
    # 训练集性能
    train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_pred)
    print(f"训练集的准确率：{train_accuracy}")
    
    # 测试集性能
    test_pred = model.predict(X_test_rfe)
    test_accuracy = accuracy_score(y_test, test_pred)
    print(f"测试集的准确率：{test_accuracy}")

if __name__ == "__main__":
    main() 
