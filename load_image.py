import cv2
import os



class Load_img():
    """
    初始化加载图像，通过getitem方法可以直接将所有的图片通过cv2将图片加载
    返回获取的图像和图像名称
    """
    def __init__(self, root_dir):
        """:param root_dir: 选择图形所在目录
        """
        self.root_dir = root_dir
        self.path = os.path.join(self.root_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, item):
        img_name = self.img_path[item]
        img_item_path = os.path.join(self.root_dir, img_name)
        # print("选择的图像为",img_name)
        # 直接获取灰度图
        img = cv2.imread(img_item_path, 0)
        return img, img_name


if __name__ == '__main__':

    # 指定图片所在的文件夹路径
    folder_path = r'D:\A_works\python_works\ceshitu'
    pic = Load_img(folder_path)
    for i in enumerate(pic):
        j,k=i
        print(j.shape,k)



