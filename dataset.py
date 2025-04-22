from config import*
# from torch.utils.data import Dataset  # Thêm dòng này
class SegmentationDataset(Dataset):
	def __init__(self, imagePaths, maskPaths, transforms):
		# store the image and mask filepaths, and augmentation
		# transforms
		self.imagePaths = imagePaths
		self.maskPaths = maskPaths
		self.transforms = transforms
	def __len__(self):
		# return the number of total samples contained in the dataset
		return len(self.imagePaths)
	def __getitem__(self, idx):
		# grab the image path from the current index
		imagePath = self.imagePaths[idx]
		# load the image from disk, swap its channels from BGR to RGB,
		# and read the associated mask from disk in grayscale mode
		image = cv2.imread(imagePath)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		mask = cv2.imread(self.maskPaths[idx], 0)
		# check to see if we are applying any transformations
		if self.transforms is not None:
			# apply the transformations to both image and its mask
			image = self.transforms(image)
			mask = self.transforms(mask)
		# return a tuple of the image and its mask
		return (image, mask)
# load the image and mask filepaths in a sorted manner
trainImagesPaths = sorted(list(paths.list_images(IMAGE_TRAIN_PATH)))
trainMasksPaths = sorted(list(paths.list_images(MASK_TRAIN_PATH)))

validImagesPaths = sorted(list(paths.list_images(IMAGE_VALID_PATH)))
validMasksPaths = sorted(list(paths.list_images(MASK_VALID_PATH)))

transforms = transforms.Compose([transforms.ToPILImage(),
 	transforms.Resize((INPUT_IMAGE_HEIGHT,
		INPUT_IMAGE_WIDTH)),
	transforms.ToTensor()])
# create the train and test datasets
trainDS = SegmentationDataset(imagePaths=trainImagesPaths, maskPaths=trainMasksPaths,
	transforms=transforms)
validDS = SegmentationDataset(imagePaths=validImagesPaths, maskPaths=validMasksPaths,
    transforms=transforms)

print(f"[INFO] found {len(trainDS)} examples in the training set...")
print(f"[INFO] found {len(validDS)} examples in the valid set...")
# create the training and test data loaders

trainLoader = DataLoader(trainDS, shuffle=True,
	batch_size=bach_size, pin_memory=PIN_MEMORY,
	num_workers=4)
validLoader = DataLoader(validDS, shuffle=False,
	batch_size=bach_size, pin_memory=PIN_MEMORY,
	num_workers=4)
