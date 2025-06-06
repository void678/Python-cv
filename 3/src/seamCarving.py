def getDynamicEnergyMap(image, mask):
    """计算动态能量图和路径图"""
    # 获取图像尺寸
    height, width = image.shape[:2]
    
    # 初始化能量图和路径图
    energy_map = np.zeros((height, width))
    route_map = np.zeros((height, width), dtype=np.int32)
    
    # 计算初始能量图
    for i in range(height):
        for j in range(width):
            if mask[i, j] == 1:  # 如果像素在mask中
                energy_map[i, j] = float('inf')  # 设置高能量值
            else:
                # 计算RGB差异
                if j > 0 and j < width - 1:
                    dx = np.sum(np.abs(image[i, j+1] - image[i, j-1]))
                else:
                    dx = 0
                if i > 0 and i < height - 1:
                    dy = np.sum(np.abs(image[i+1, j] - image[i-1, j]))
                else:
                    dy = 0
                energy_map[i, j] = dx + dy
    
    # 动态规划寻找最小能量路径
    for i in range(1, height):
        for j in range(width):
            # 获取上方三个像素的能量值
            up = energy_map[i-1, j]
            up_left = energy_map[i-1, j-1] if j > 0 else float('inf')
            up_right = energy_map[i-1, j+1] if j < width-1 else float('inf')
            
            # 找到最小能量值及其对应的方向
            min_energy = min(up_left, up, up_right)
            if min_energy == up_left:
                route_map[i, j] = -1  # 左上
            elif min_energy == up:
                route_map[i, j] = 0   # 上
            else:
                route_map[i, j] = 1   # 右上
            
            # 更新当前像素的能量值
            energy_map[i, j] += min_energy
    
    return energy_map, route_map

def getSeam(route_map, last_pixel_idx):
    """根据路径图获取seam"""
    height = route_map.shape[0]
    seam = np.zeros(height, dtype=np.int32)
    
    # 从底部开始追踪路径
    current_idx = last_pixel_idx
    for i in range(height-1, -1, -1):
        seam[i] = current_idx
        if i > 0:  # 不是第一行
            current_idx += route_map[i, current_idx]
    
    return seam

def removeSeam(image, seam):
    """移除seam并重构图像"""
    height, width = image.shape[:2]
    new_image = np.zeros((height, width-1, 3), dtype=np.uint8)
    
    # 重构图像
    for i in range(height):
        # 跳过seam对应的像素
        new_image[i, :seam[i]] = image[i, :seam[i]]
        new_image[i, seam[i]:] = image[i, seam[i]+1:]
    
    return new_image

def changeShape(image, mask, new_width, new_height):
    """改变图像尺寸"""
    height, width = image.shape[:2]
    
    # 确保新尺寸小于原始尺寸
    if new_width >= width or new_height >= height:
        raise ValueError("新尺寸必须小于原始尺寸")
    
    # 计算需要移除的seam数量
    seams_to_remove_width = width - new_width
    seams_to_remove_height = height - new_height
    
    # 水平方向移除seam
    for _ in range(seams_to_remove_width):
        # 计算能量图和路径图
        energy_map, route_map = getDynamicEnergyMap(image, mask)
        
        # 找到能量最小的seam
        last_row_energy = energy_map[-1]
        last_pixel_idx = np.argmin(last_row_energy)
        seam = getSeam(route_map, last_pixel_idx)
        
        # 移除seam
        image = removeSeam(image, seam)
        mask = removeSeam(mask, seam)
    
    # 垂直方向移除seam
    if seams_to_remove_height > 0:
        # 转置图像和mask
        image = np.transpose(image, (1, 0, 2))
        mask = np.transpose(mask, (1, 0))
        
        for _ in range(seams_to_remove_height):
            energy_map, route_map = getDynamicEnergyMap(image, mask)
            last_row_energy = energy_map[-1]
            last_pixel_idx = np.argmin(last_row_energy)
            seam = getSeam(route_map, last_pixel_idx)
            image = removeSeam(image, seam)
            mask = removeSeam(mask, seam)
        
        # 转置回原始方向
        image = np.transpose(image, (1, 0, 2))
        mask = np.transpose(mask, (1, 0))
    
    return image 