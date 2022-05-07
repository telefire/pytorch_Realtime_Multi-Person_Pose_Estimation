# get COCO dataset
mkdir dataset
mkdir dataset/COCO/
cd dataset/COCO/
git clone https://github.com/pdollar/coco.git
cd ../../

mkdir dataset/COCO/images
mkdir dataset/COCO/images/mask2017
mkdir dataset/COCO/mat
mkdir dataset/COCO/json

wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/zips/test2017.zip

unzip annotations_trainval2017.zip.zip -d dataset/COCO/
unzip val2017.zip -d dataset/COCO/images
unzip test2017.zip -d dataset/COCO/images
unzip train2017.zip -d dataset/COCO/images

rm -f person_keypoints_trainval2017.zip
rm -f test2017.zip
rm -f train2017.zip
rm -f val2017.zip