from pennylane import numpy as np
import cv2
from skimage.feature import local_binary_pattern
import random
import pymrmr
import pandas as pd


#这里面放一些特征选择常用的函数


def extract_data_with_label(origin_data, origin_label, target_label_list):
    new_data = []
    new_labels = []
    for i in range(len(origin_label)):
        for j in range(len(target_label_list)):
            if(origin_label[i] == target_label_list[j]):
                new_data.append(origin_data[i,:,:]) 
                new_labels.append(origin_label[i])
    new_data = np.array(new_data)
    new_labels = np.array(new_labels)  
    return new_data, new_labels

def normalization(image):
	image -= image.min()
	image = image / (image.max() - image.min())
	image *= 255
	image = image.astype(np.uint8)
	return image


def extract_HOG_features(origin_image):
    #origin_image = cv2.cvtColor(origin_image, cv2.COLOR_RGB2GRAY) #由于MNIST数据集本身就是灰度图，所以不需要再转灰度图
    origin_image = normalization(origin_image)
    #cell_size = (6,6)   #(6,6) = 324 dim; (5,5) = 576 dim; (7,7) = 324 dim;  (8,8) = 144 dim
    cell_size = (6,6)    
    num_cells_per_block = (2,2)
    block_size = (num_cells_per_block[0] * cell_size[0], num_cells_per_block[1] * cell_size[1])
    x_cells = origin_image.shape[1] // cell_size[0]
    y_cells = origin_image.shape[0] // cell_size[1]
    h_stride = 1
    v_stride = 1
    block_stride = (cell_size[0] * h_stride, cell_size[1] * v_stride)
    num_bins = 9
    win_size = (x_cells * cell_size[0], y_cells * cell_size[1])
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)
    hog_descriptor = hog.compute(origin_image)
    return hog_descriptor

def fs_with_HOG(ori_imgs):
	new_images = []
	for idx, img in enumerate(ori_imgs):
		new_images.append(extract_HOG_features(img))
	new_images = np.array(new_images)
	return new_images


def get_random_selected_list(selected_feature_num, feature_number_sum, file_path, load_state=False): #生成随机列表，用于选择特定的像素
	selected_index_list = []

	if load_state == True:
	 	selected_index_list = np.load(file_path)
	 	selected_index_list = selected_index_list.tolist()
	else: 
		for i in range(selected_feature_num):
			selected_index_list.append(random.randint(0,feature_number_sum-1))

		selected_index_list = np.array(selected_index_list, requires_grad=False)
		np.save(file_path, selected_index_list)

	return selected_index_list


def get_mRMR_selected_list_with_train_data(selected_feature_num, origin_data, labels, file_path, load_state=False):
	selected_index_list = []

	if load_state == True:
		selected_index_list = np.load(file_path)
	else:
	    labels = np.reshape(labels, (np.shape(labels)[0], -1))
	    #mRMR应该是对训练集提取特征子集，然后在测试集上选取一样的特征子集来测试效果
	    origin_data = np.reshape(origin_data,(np.shape(origin_data)[0],-1))
	    data = np.concatenate((labels,origin_data), axis=1)

	    row_index_list = []
	    row_name = []
	    for i in range(len(labels)):
	        row_name = 'Row_' + str(i+1)
	        row_index_list.append(row_name)

	    column_index_list = []
	    column_name = []

	    for i in range(np.shape(origin_data)[1] + 1):
	        column_name = 'Colum_' + str(i+1)
	        column_index_list.append(column_name)

	    data_df = pd.DataFrame(data, index=row_index_list, columns = column_index_list)
	    selected_index_list = pymrmr.mRMR(data_df, 'MID', selected_feature_num)
	    np.save(file_path, selected_index_list)
	return selected_index_list

def fs_with_random(ori_data, selected_index_list): # 随机选择若干个像素
	new_data = []
	selected_index_list.sort()
	ori_data = np.reshape(ori_data,(np.shape(ori_data)[0],-1))

	for i in range(len(selected_index_list)):
		new_data.append(ori_data[:,selected_index_list[i]])

	new_data = np.array(new_data, requires_grad=False)
	new_data = np.transpose(new_data)
	return new_data 

def fs_with_LBP(ori_imgs, radius): # radius为LBP算法中范围半径的取值
	n_points = 8 * radius
	new_images = []
	for idx, img in enumerate(ori_imgs):
		new_images.append(local_binary_pattern(img, n_points, radius))
	new_images = np.array(new_images, requires_grad=False)
	return new_images

def fs_with_mRMR(ori_data, labels, selected_index_list):
	#mRMR应该是对训练集提取特征子集，然后在测试集上选取一样的特征子集来测试效果
    labels = np.reshape(labels, (np.shape(labels)[0], -1))
    ori_data = np.reshape(ori_data,(np.shape(ori_data)[0],-1))
    data = np.concatenate((labels,ori_data), axis=1)

    row_index_list = []
    row_name = []
    for i in range(len(labels)):
        row_name = 'Row_' + str(i+1)
        row_index_list.append(row_name)

    column_index_list = []
    column_name = []

    for i in range(np.shape(ori_data)[1] + 1):
        column_name = 'Colum_' + str(i+1)
        column_index_list.append(column_name)

    data_df = pd.DataFrame(data, index=row_index_list, columns = column_index_list)
    new_data = data_df[selected_index_list]
    return np.array(new_data, dtype='float64')


#当维数特别低时候，比如少于20维时，有些数据就全为0，需要剔除,label要同步处理
def remove_invalid_image_and_labels(data, labels):
	remove_index = data.sum(axis=1)!=0
	data = data[remove_index,:]

	#process labels
	labels = np.reshape(labels, (np.shape(labels)[0], -1))
	labels = labels[remove_index,:]
	labels = labels.reshape(-1)
	return data, labels
