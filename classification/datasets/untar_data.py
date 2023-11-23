import tarfile
import os

def un_tar(path, sve_path):
	list_tar = os.listdir(path)
	for tar_i in list_tar:
		save_name_dir = tar_i.split('.')[0]
		tar_path = os.path.join(path, tar_i)
		tar = tarfile.open(tar_path)
		names = tar.getnames()
		save_path = os.path.join(sve_path, save_name_dir)
		os.mkdir(save_path)
		for name in names:
			tar.extract(name, save_path)
		tar.close()

if __name__ == '__main__':
	path = './ILSVRC2012_img_train'
	sve_path = './Imagenet/train'
	un_tar(path, sve_path)