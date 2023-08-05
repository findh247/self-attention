
from PIL import Image
old_path = r"D:\App\Tencent\im0001.ah.ppm"
save_path = r'D:\article\c.jpg'

# def PPM_to_JPG(old_path,save_path):#old_path为原始ppm文件地址，save_path为保存的jpg文件地址
#     val=''
#     with open(old_path,'rb') as f:#二进制读取原始图像数据
#         val=f.read()
#     with open(save_path,'wb') as f:#将二进制数据流转换为JPG文件
#         f.write(val)
#
# PPM_to_JPG(old_path,save_path)
ppm_image = Image.open(old_path,mode='r')
ppm_image.save(save_path, 'JPEG')
