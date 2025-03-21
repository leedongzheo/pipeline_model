from config import*
from train import get_args
def dice_coef_loss(inputs, target, smooth=1e-6):
    """
    Dice Loss: Thước đo sự chồng lấn giữa output và ground truth.
    """
    inputs = torch.sigmoid(inputs)  # Chuyển logits về xác suất
    intersection = (inputs * target).sum()
    union = inputs.sum() + target.sum()
    dice_score = (2.0 * intersection + smooth) / (union + smooth)
    return 1 - dice_score  # Dice loss
def bce_dice_loss(inputs, target):
    dice_score = dice_coef_loss(inputs, target)
    bce_loss = nn.BCELoss()
    bce_score = bce_loss(inputs, target)
    return bce_score + dice_score
def tensor_to_float(value):
    if isinstance(value, torch.Tensor):
        return value.cpu().item()  # Chuyển tensor về CPU và lấy giá trị float
    elif isinstance(value, list):
        return [tensor_to_float(v) for v in value]  # Xử lý danh sách các tensor
    return value  # Nếu không phải tensor, giữ nguyên
def to_numpy(tensor):
    # Move tensor to CPU and convert to NumPy array
    return tensor.cpu().detach().item()
def dice_coeff(self, pred, target, smooth=1e-5):
    intersection = torch.sum(pred * target)
    return (2. * intersection + smooth) / (torch.sum(pred) + torch.sum(target) + smooth)
# def inan():
def loss_func(inputs, target):
    args = get_args()
    if args.loss == "Dice_loss":
        x=dice_coef_loss(inputs,target)
        return x
    elif args.loss == "BCEDice_loss":
        x=bce_dice_loss(inputs,target)
        return x



    
