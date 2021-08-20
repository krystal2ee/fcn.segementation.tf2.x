from PIL import Image
from matplotlib import pyplot as plt

root = 'E:\Segmentation\Datasets\PASKAL_VOC_11-May-2012\VOCtrainval\VOCdevkit\VOC2012/'
filename = '2007_000129'


class_filename = root + 'SegmentationClass/'+ filename +'.png'
object_filename = root + 'SegmentationObject/' + filename + '.png'

class_image = Image.open(class_filename)
object_image = Image.open(object_filename)

ax1 = plt.subplot(1,2,1)
ax1.set_title('Semantic Segmentation')
plt.imshow(class_image)
ax2 = plt.subplot(1,2,2)
ax2.set_title('Instance Segmentation')
plt.imshow(object_image)
plt.savefig('./example/pascal-voc.png')
plt.show()
